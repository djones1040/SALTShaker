import os
import tempfile
import unittest

import jax
import numpy as np
import sncosmo
from jax import numpy as jnp
from scipy import integrate

from saltshaker.training import saltresids
from saltshaker.training.TrainSALT import _SALT2COLORLAW, RunTraining, TrainSALT
from saltshaker.training.masscorrection import (
    MASSCORRECTIONFILE, massknotweights, masscorrection, masscorrectionbases, masscorrectionsubspace,
    readmasscorrection, sampleknotweights, surfacesatmass, writemodelatmass)
from saltshaker.util import readutils
from saltshaker.util.salt3_sncosmo import SALT3Source
from saltshaker.validation.masscorrectionplots import BANDS, deltamagnitudes, plotmasscorrection, readtemplates


class test_masscorrection_functions(unittest.TestCase):
    knots = np.array([8.5, 9.5, 10.5, 11.5])

    def test_knot_weights_match_numerical_average(self):
        """Closed-form weights equal the clamped interpolation averaged numerically over the Gaussian, and sum to one."""
        for mass, err in [(10.99, 0.05), (9.0, 0.4), (10.66, 4.0), (12.3, 0.2)]:
            def integrand(m, k):
                return np.interp(m, self.knots, np.eye(4)[k]) * np.exp(-0.5 * ((m - mass) / err) ** 2) / (err * np.sqrt(2 * np.pi))
            numerical = [integrate.quad(integrand, mass - 12 * err, mass + 12 * err, args=(k,),
                                        points=self.knots, limit=200)[0] for k in range(4)]
            weights = massknotweights(mass, err, self.knots)
            np.testing.assert_allclose(weights, numerical, rtol=0, atol=1e-10)
            self.assertAlmostEqual(weights.sum(), 1, places=12)

    def test_point_mass_gives_interpolation_weights(self):
        """A zero or NaN uncertainty gives linear interpolation, constant beyond the end knots."""
        np.testing.assert_allclose(massknotweights(10.2, 0, self.knots), [0, 0.3, 0.7, 0], atol=1e-12)
        np.testing.assert_allclose(massknotweights(10.2, np.nan, self.knots), [0, 0.3, 0.7, 0], atol=1e-12)
        np.testing.assert_array_equal(massknotweights(7.0, 0, self.knots), [1, 0, 0, 0])
        self.assertTrue(np.isnan(massknotweights(np.nan, 0.1, self.knots)).all())

    def test_imprecise_masses_follow_population(self):
        """Missing and very imprecise masses get the population weights; precise masses keep their own."""
        rng = np.random.default_rng(0)
        masses = np.concatenate([rng.normal(10.5, 0.5, 200), [9.0, 10.66, np.nan]])
        errs = np.concatenate([np.full(200, 0.05), [0.01, 4.0, np.nan]])
        weights = sampleknotweights(masses, errs, self.knots)
        np.testing.assert_allclose(weights[-2], weights[-1], atol=0.01)
        np.testing.assert_allclose(weights[-3], massknotweights(9.0, 0.01, self.knots), atol=1e-3)
        np.testing.assert_allclose(weights.sum(axis=1), 1)

    def test_bases(self):
        """Wavelength-basis rows are orthonormal, phase-basis columns are orthogonal with M0's norm, and zero input rows stay zero."""
        rng = np.random.default_rng(1)
        m0, m1 = rng.normal(size=(20, 30)), rng.normal(size=(20, 30))
        m0[:2], m1[:2] = 0, 0
        phasebasis, wavebasis = masscorrectionbases([m0, m1], 3)
        np.testing.assert_allclose(wavebasis @ wavebasis.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(phasebasis.T @ phasebasis, np.eye(3) * np.linalg.norm(m0) ** 2, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(phasebasis[:2], 0, atol=1e-12)

    def test_subspace(self):
        """The subspace is orthonormal, has one dimension fewer per reference, and its corrections are orthogonal to every reference."""
        rng = np.random.default_rng(5)
        phasebasis, wavebasis = rng.normal(size=(20, 3)), np.linalg.qr(rng.normal(size=(30, 3)))[0].T
        references = [rng.normal(size=(20, 30)) for _ in range(3)]
        subspace = masscorrectionsubspace(phasebasis, wavebasis, references)
        self.assertEqual(subspace.shape, (9, 6))
        np.testing.assert_allclose(subspace.T @ subspace, np.eye(6), atol=1e-12)
        for coords in subspace.T:
            correction = phasebasis @ coords.reshape(3, 3) @ wavebasis
            for ref in references:
                self.assertAlmostEqual(np.sum(correction * ref) / np.linalg.norm(correction) / np.linalg.norm(ref), 0, places=12)
        with self.assertRaisesRegex(ValueError, 'masscorrection_rank'):
            masscorrectionsubspace(phasebasis[:, :1], wavebasis[:1], references)

    def test_correction_matches_explicit_sum(self):
        """Each component's correction is U (sum_k w_k W_k) V, flattened phase-major, with W_k given by its subspace coordinates and W_0 = 0."""
        rng = np.random.default_rng(2)
        phasebasis, wavebasis = rng.normal(size=(5, 2)), rng.normal(size=(2, 7))
        subspace = np.linalg.qr(rng.normal(size=(4, 3)))[0]
        weights, perknot = np.array([0.2, 0.5, 0.3]), rng.normal(size=(2, 2, 3))
        expected = [phasebasis @ (subspace @ np.tensordot(weights[1:], perknot[c], axes=1)).reshape(2, 2) @ wavebasis
                    for c in range(2)]
        result = masscorrection(weights, perknot.reshape(2, -1), phasebasis, wavebasis, subspace)
        np.testing.assert_allclose(result, np.reshape(expected, (2, -1)), rtol=1e-12)


class test_masscorrection_training(unittest.TestCase):

    @staticmethod
    def configuredtrainer(**options):
        salt = TrainSALT()
        RunTraining().get_config_options(salt, 'testdata/test.conf', [])
        salt.options.masscorrection_rank = 3
        salt.options.masscorrection_knots = [9.5, 10.5, 11.5]
        for name, value in options.items():
            setattr(salt.options, name, value)
        return salt

    @classmethod
    def setUpClass(cls):
        salt = cls.configuredtrainer()
        cls.datadict = readutils.rdAllData(
            'testdata/data/SALT3TRAIN_K21_CSPDR3/SALT3TRAIN_K21_CSPDR3.LIST', False, dospec=True,
            peakmjdlist='testdata/test_pkmjd.LIST', binspecres=None, snparlist='testdata/test_pars.LIST',
            maxsn=None, specrecallist='testdata/test_spcrl.LIST')
        salt.kcordict = readutils.rdkcor(['CSP'], salt.options)
        x, config = salt.initialParameters(cls.datadict)
        cls.salt, cls.config = salt, config
        cls.res = saltresids.SALTResids(cls.datadict, salt.kcordict, config, salt.options)
        rng = np.random.default_rng(3)
        cls.xinit = np.asarray(x)
        cls.x = np.asarray(x).copy()
        cls.x[cls.res.imasscorr.flatten()] = rng.normal(scale=0.1, size=cls.res.imasscorr.size)
        cls.x[cls.res.ix1] = rng.normal(0.3, 1.5, size=cls.res.ix1.size)

    def explicitlycorrected(self, sn):
        """Parameters with sn's correction added to the components and the correction parameters zeroed"""
        res = self.res
        delta = np.asarray(masscorrection(res.massknotweights[sn], self.x[res.imasscorr],
                                          res.masscorrphasebasis, res.masscorrwavebasis, res.masscorrsubspace))
        explicit = self.x.copy()
        explicit[res.imasscorr.flatten()] = 0
        for i in range(res.n_components):
            explicit[res.icomponents[i]] += delta[i]
        return explicit

    def writeoutput(self, outdir, result=None):
        self.salt.options.yamloutputfile = os.path.join(outdir, 'output.yaml')
        if result is None:
            result = self.res.processoptimizedparametersforoutput(self.x, self.x)
        self.salt.wrtoutput(outdir, result, None, None, self.datadict)

    def resumedparameters(self, outdir, **options):
        salt = self.configuredtrainer(resume_from_outputdir=outdir, **options)
        salt.kcordict = self.salt.kcordict
        return salt.initialParameters(self.datadict)

    def test_flux_matches_explicitly_corrected_components(self):
        """Light-curve and spectrum fluxes equal those of the uncorrected model with each SN's corrected components."""
        res = self.res
        for sn in ['2004ef', '2006et']:
            explicit = self.explicitlycorrected(sn)
            for data in list(res.datadict[sn].photdata.values()) + list(res.datadict[sn].specdata.values()):
                np.testing.assert_allclose(data.modelflux(self.x), data.modelflux(explicit), rtol=1e-12, atol=0)

    def test_batched_likelihoods_use_each_sn_weights(self):
        """Batched likelihoods equal the sum over light curves and spectra, each with its own SN's mass weights."""
        res = self.res
        weights = np.array(list(res.massknotweights.values()))
        self.assertEqual(len(np.unique(weights.round(12), axis=0)), len(weights))
        for batched, kind in [(res.batchedphotlikelihood, 'photdata'), (res.batchedspeclikelihood, 'specdata')]:
            unbatched = sum(data.modelloglikelihood(self.x) for sn in res.datadict.values()
                            for data in getattr(sn, kind).values())
            np.testing.assert_allclose(batched(self.x, None, False, False), unbatched, rtol=1e-12)

    def test_photometry_recovers_injected_correction(self):
        """With other parameters fixed, light-curve fluxes are linear in the correction and determine all of it."""
        res = self.res
        icorr = res.imasscorr.flatten()
        lcs = [lc for sn in res.datadict.values() for lc in sn.photdata.values()]

        def fluxes(corr):
            x = jnp.array(self.x).at[icorr].set(corr)
            return jnp.concatenate([lc.modelflux(x) for lc in lcs])

        # Small enough that no model flux is clipped at zero
        injected = self.x[icorr] / 200
        jacobian = np.asarray(jax.jacfwd(fluxes)(jnp.zeros(icorr.size)))
        self.assertEqual(np.linalg.matrix_rank(jacobian), icorr.size)
        recovered = np.linalg.lstsq(jacobian, np.asarray(fluxes(injected) - fluxes(jnp.zeros(icorr.size))), rcond=None)[0]
        np.testing.assert_allclose(recovered, injected, rtol=0, atol=1e-10)

    def test_final_definitions_preserve_photometry(self):
        """Rescaling and mixing M0 and M1 to meet the SALT definitions carries the corrections along."""
        res = self.res
        redefined = res.constraints.enforcefinaldefinitions(self.x, res.SALTModel(self.x), checkerrors=False)
        self.assertFalse(np.allclose(redefined[res.imasscorr], self.x[res.imasscorr]))
        np.testing.assert_allclose(res.batchedphotlikelihood(redefined), res.batchedphotlikelihood(self.x), rtol=1e-10)

    def test_bband_constraint_preserves_model_flux(self):
        """fixbbandfluxes changes components and corrections together, leaving model fluxes unchanged."""
        res = self.res
        constrained = np.asarray(res.constraints.constraints['fixbbandfluxes'](jnp.array(self.x)))
        for lc in res.datadict['2004ef'].photdata.values():
            np.testing.assert_allclose(lc.modelflux(constrained), lc.modelflux(self.x), rtol=1e-10, atol=0)

    def test_corrections_meet_definitions_at_each_knot(self):
        """Each knot's correction to each component has zero B-band flux at peak and no component along the initial M1 or the color law times M0."""
        res = self.res
        shape = (len(res.phaseknotloc) - res.bsorder - 1, len(res.waveknotloc) - res.bsorder - 1)
        m0, m1 = self.xinit[res.im0].reshape(shape), self.xinit[res.im1].reshape(shape)
        centers = np.array([res.waveknotloc[j + 1:j + res.bsorder + 1].mean() for j in range(shape[1])])
        colored = _SALT2COLORLAW(centers)[np.newaxis, :] * m0
        bflux = np.asarray(res.constraints.__maximumlightpcderiv__.todense())
        for k in range(1, len(res.masscorrection_knots)):
            knot = np.asarray(masscorrection(np.eye(len(res.masscorrection_knots))[k], self.x[res.imasscorr],
                                             res.masscorrphasebasis, res.masscorrwavebasis, res.masscorrsubspace))
            for correction in knot:
                np.testing.assert_allclose(bflux @ correction, 0, atol=1e-12 * np.abs(bflux @ m0.flatten()).max())
                for ref in [m1, colored]:
                    cosine = correction @ ref.flatten() / np.linalg.norm(correction) / np.linalg.norm(ref)
                    self.assertAlmostEqual(cosine, 0, places=12)

    def test_correction_averages_to_zero(self):
        """The correction averages to zero over the training sample."""
        res = self.res
        corrections = [np.asarray(masscorrection(w, self.x[res.imasscorr], res.masscorrphasebasis,
                                                 res.masscorrwavebasis, res.masscorrsubspace))
                       for w in res.massknotweights.values()]
        np.testing.assert_allclose(np.mean(corrections, axis=0), 0, atol=1e-12 * np.abs(corrections).max())

    def test_regularization_covers_correction_at_each_knot(self):
        """The gradient regularizations add the same terms for each knot's correction as for the components, which vanish for a zero correction."""
        res = self.res
        uncorrected = self.x.copy()
        uncorrected[res.imasscorr.flatten()] = 0
        for func in [res.phaseGradientRegularization, res.waveGradientRegularization]:
            full, base = np.asarray(func(self.x, res.neff)), np.asarray(func(uncorrected, res.neff))
            ncomponents = base.size // (len(res.masscorrection_knots) + 1)
            np.testing.assert_allclose(full[:ncomponents], base[:ncomponents], rtol=1e-12)
            np.testing.assert_array_equal(base[ncomponents:], 0)
            self.assertGreater(np.abs(full[ncomponents:]).max(), 0)

    def test_output_gives_model_at_each_host_mass(self):
        """The written correction rebuilds each SN's model as a standard SALT3 model directory."""
        res = self.res
        with tempfile.TemporaryDirectory() as outdir:
            self.writeoutput(outdir)
            for sn in ['2004ef', '2006et']:
                modeldir = os.path.join(outdir, sn)
                writemodelatmass(outdir, modeldir, self.datadict[sn].hostlogmass,
                                 self.datadict[sn].hostlogmasserr)
                expected = res.SALTModel(self.explicitlycorrected(sn), res.phaseout, res.waveout)
                for i in range(res.n_components):
                    phase, wave, flux = np.loadtxt(os.path.join(modeldir, f'salt3_template_{i}.dat'), unpack=True)
                    np.testing.assert_allclose(flux, expected[i].ravel(), rtol=1e-10, atol=1e-12 * np.abs(expected[i]).max())
                    np.testing.assert_array_equal(
                        np.loadtxt(os.path.join(outdir, f'salt3_template_{i}.dat'), usecols=(0, 1)),
                        np.column_stack([phase, wave]))
            self.assertEqual(
                sorted(os.listdir(modeldir)),
                sorted(name for name in os.listdir(outdir)
                       if name == 'SALT3.INFO' or name.startswith(('salt3_template', 'salt3_lc', 'salt3_color'))))

    def test_plotted_magnitudes_match_model_at_mass(self):
        """Magnitude changes in the validation plot equal sncosmo photometry of the model written at that mass."""
        with tempfile.TemporaryDirectory() as outdir:
            self.writeoutput(outdir)
            result, phase, wave = readmasscorrection(outdir)
            average = readtemplates(outdir, self.res.n_components, phase, wave)
            peak = np.argmin(np.abs(phase))
            for logmass in [9.0, 10.2, 11.5]:
                modeldir = os.path.join(outdir, str(logmass))
                writemodelatmass(outdir, modeldir, logmass)
                models = [sncosmo.Model(SALT3Source(modeldir=d)) for d in (modeldir, outdir)]
                for model in models:
                    model.set(z=0, t0=0, x0=1, x1=0, c=0)
                expected = [-2.5 * np.log10(models[0].bandflux(band, 0) / models[1].bandflux(band, 0)) for band in BANDS]
                plotted = deltamagnitudes(surfacesatmass(result, logmass)[0], average[0], wave)[:, peak]
                np.testing.assert_allclose(plotted, expected, rtol=0, atol=1e-5)
            plotfile = os.path.join(outdir, 'SALTmasscomp.pdf')
            plotmasscorrection(plotfile, outdir, logmasses=[sn.hostlogmass for sn in self.datadict.values()])
            self.assertGreater(os.path.getsize(plotfile), 0)

    def test_resume_reads_saved_correction(self):
        """Resuming reads the bases and corrections from the output, and starts at zero when they are not there."""
        res = self.res
        with tempfile.TemporaryDirectory() as outdir:
            result = res.processoptimizedparametersforoutput(self.x, self.x)
            # Bases unlike the initialization's, as from different initial templates
            saved = result.masscorrection._replace(phasebasis=2 * result.masscorrection.phasebasis)
            self.writeoutput(outdir, result._replace(masscorrection=saved))
            guess, config = self.resumedparameters(outdir)
            np.testing.assert_array_equal(config.masscorrphasebasis, saved.phasebasis)
            # Parameters are written to 16 significant digits
            np.testing.assert_allclose(guess[res.imasscorr], self.x[res.imasscorr], rtol=1e-14)
            with self.assertRaises(ValueError):
                self.resumedparameters(outdir, masscorrection_knots=[9.5, 10.5, 11.0])

            os.remove(os.path.join(outdir, MASSCORRECTIONFILE))
            for name in ['salt3_parameters.dat', 'salt3_parameters_unscaled.dat']:
                with open(os.path.join(outdir, name)) as fin:
                    lines = [line for line in fin if not line.startswith('masscorr')]
                with open(os.path.join(outdir, name), 'w') as fout:
                    fout.writelines(lines)
            guess, config = self.resumedparameters(outdir)
            np.testing.assert_array_equal(config.masscorrphasebasis, res.masscorrphasebasis)
            np.testing.assert_array_equal(guess[res.imasscorr], 0)
            np.testing.assert_allclose(guess[res.icomponents], self.x[res.icomponents], rtol=1e-14)

    def test_smoothness_prior(self):
        """masscorrsmooth vanishes for corrections linear in mass."""
        res = self.res
        x = self.x.copy()
        knots = np.array(res.masscorrection_knots)
        pattern = np.random.default_rng(4).normal(size=(res.n_components, 1, res.masscorrsubspace.shape[1]))
        offsets = (knots[1:] - knots[0])[np.newaxis, :, np.newaxis]
        x[res.imasscorr] = (offsets * pattern).reshape(res.n_components, -1)
        np.testing.assert_allclose(res.priors.priors['masscorrsmooth'](1e-3, x), 0, atol=1e-10)
        x[res.imasscorr] = (offsets ** 2 * pattern).reshape(res.n_components, -1)
        self.assertGreater(np.abs(res.priors.priors['masscorrsmooth'](1e-3, x)).max(), 1)


if __name__ == "__main__":
    unittest.main()
