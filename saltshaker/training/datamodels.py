"""
Data model classes for SALT3 training.

This module provides JAX-compatible data structures for photometric and
spectroscopic training data, enabling efficient batched model evaluation
and residual computation.

Classes
-------
SALTparameters
    Container for SALT model parameters (x0, x1, c, components, etc.).
modeledtrainingdata
    Abstract base class for modeled training data.
modeledtraininglightcurve
    Light curve data with model evaluation methods.
modeledtrainingspectrum
    Spectrum data with model evaluation methods.
SALTfitcacheSN
    Cached data for a single supernova with precomputed quantities.

Notes
-----
All data classes are registered as JAX pytree nodes for use with
jax.jit, jax.vmap, and automatic differentiation.
"""

from saltshaker.util.readutils import (
    SALTtrainingSN,
    SALTtraininglightcurve,
    SALTtrainingspectrum,
)


from scipy.special import factorial
from scipy.interpolate import bisplev, interp1d
from scipy import sparse as scisparse
import numpy as np

from jax import numpy as jnp
import jax
from jax.experimental import sparse
from jax.scipy import linalg as jaxlinalg
from jax.tree_util import register_pytree_node_class


from functools import partial
from saltshaker.util.jaxoptions import jaxoptions, sparsejaxoptions

import extinction
import warnings
import logging
import abc

log = logging.getLogger(__name__)

warnings.simplefilter("ignore", category=FutureWarning)

from sncosmo.constants import HC_ERG_AA, MODEL_BANDFLUX_SPACING

recalmax = 20

_SCALE_FACTOR = 1e-12


def __anyinnonzeroareaforsplinebasis__(
    phase, wave, phaseknotloc, waveknotloc, bsorder, i
):
    """
    Check if any data points fall within the non-zero region of a B-spline basis function.

    B-spline basis functions are only non-zero within a limited range defined by
    the knot locations and spline order. This function determines if any of the
    given phase/wavelength points overlap with the support of basis function i.

    Parameters
    ----------
    phase : ndarray
        Phase values of the data points.
    wave : ndarray
        Wavelength values of the data points.
    phaseknotloc : ndarray
        Knot locations along the phase axis.
    waveknotloc : ndarray
        Knot locations along the wavelength axis.
    bsorder : int
        B-spline order (degree).
    i : int
        Linear index of the basis function (flattened from 2D phase/wave grid).

    Returns
    -------
    bool
        True if any data points fall within the non-zero region of basis i.
    """
    # Convert linear index to 2D indices for phase and wavelength
    phaseindex, waveindex = i // (waveknotloc.size - bsorder - 1), i % (
        waveknotloc.size - bsorder - 1
    )
    # Check if any phase values fall within the knot span for this basis function
    return (
        (phase >= phaseknotloc[phaseindex])
        & (phase <= phaseknotloc[phaseindex + bsorder + 1])
    ).any() and (
        (wave >= waveknotloc[waveindex])
        & (wave <= waveknotloc[waveindex + bsorder + 1])
    ).any()


def mvn_likelihood_simplified_cov(x, mu, D_diag, u):
    """
    Compute the likelihood of a multivariate normal distribution with
    covariance matrix Sigma = diag(D_diag) + u*u^T

    Parameters:
    -----------
    x : jnp.ndarray
        Observed vector (or batch of vectors) of shape (..., d)
    mu : jnp.ndarray
        Mean vector of shape (d,)
    D_diag : jnp.ndarray
        Diagonal elements of the diagonal matrix D, shape (d,)
    u : jnp.ndarray
        Vector for the outer product, shape (d,)

    Returns:
    --------
    likelihood : jnp.ndarray
        The likelihood value(s)
    """
    # Get dimensions
    d = mu.shape[0]

    # Center the data
    x_centered = x - mu

    # Compute D^(-1)
    D_inv_diag = 1.0 / D_diag

    # Compute u^T D^(-1) u
    u_D_inv_u = jnp.sum(u * D_inv_diag * u)

    # Compute log determinant using the matrix determinant lemma
    # |D + uu^T| = |D| * (1 + u^T D^(-1) u)
    log_det = jnp.sum(jnp.log(D_diag)) + jnp.log(1.0 + u_D_inv_u)

    # Compute D^(-1) x
    D_inv_x = x_centered * D_inv_diag

    # Compute u^T D^(-1) x
    u_D_inv_x = jnp.sum(u * D_inv_x, axis=-1)

    # Compute the quadratic form (x - μ)^T Σ^(-1) (x - μ) using Sherman-Morrison formula
    # (D + uu^T)^(-1) = D^(-1) - (D^(-1)u u^T D^(-1))/(1 + u^T D^(-1) u)
    quad_form_1 = jnp.sum(x_centered * D_inv_x, axis=-1)
    quad_form_2 = (u_D_inv_x**2) / (1.0 + u_D_inv_u)
    quad_form = quad_form_1 - quad_form_2

    # Compute log likelihood
    log_likelihood = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + quad_form)

    # Return likelihood
    return log_likelihood


def toidentifier(input):
    """
    Convert an input value to a unique string identifier.

    Creates a deterministic identifier by hashing the input and prefixing
    with 'x' to ensure it's a valid Python identifier.

    Parameters
    ----------
    input : any
        Any hashable value to convert to an identifier.

    Returns
    -------
    str
        A unique string identifier of the form 'x<hash>'.
    """
    return "x" + str(abs(hash(input)))


@register_pytree_node_class
class SALTparameters:
    """
    Container for SALT model parameters extracted from parameter vector.

    Provides convenient access to different parameter types (SN parameters,
    model components, error surfaces) from the flat parameter array.

    Parameters
    ----------
    data : dict or object
        Object containing parameter index arrays (ix0, ic, etc.).
    parsarray : ndarray
        Full parameter vector.

    Attributes
    ----------
    x0 : ndarray
        Amplitude parameters for each SN.
    coordinates : ndarray
        SN coordinates (x1, xhost, etc.).
    components : ndarray
        Model component spline coefficients (M0, M1, etc.).
    c : ndarray
        Color parameters for each SN.
    CL : ndarray
        Color law coefficients.
    modelerrs : ndarray
        Model error surface coefficients.
    modelcorrs : ndarray
        Model correlation surface coefficients.
    """

    __slots__ = [
        "x0",
        "coordinates",
        "components",
        "c",
        "CL",
        "modelcorrs",
        "modelerrs",
        "spcrcl",
        "clscat",
        "surverrfloor",
    ]
    __ismapped__ = {"x0", "c", "coordinates", "spcrcl", "surverrfloor"}

    def __init__(self, data, parsarray):
        """
        Initialize SALTparameters by extracting parameter values from an array.

        For each parameter type (x0, c, components, etc.), looks up the
        corresponding index array in the data object and extracts the
        relevant slice from the full parameter array.

        Parameters
        ----------
        data : dict or object
            Object containing index arrays (ix0, ic, icomponents, etc.)
            that specify which elements of parsarray correspond to each
            parameter type. Can be a dict or an object with __indexattributes__.
        parsarray : ndarray
            Full parameter vector containing all model parameters.
        """
        for var in self.__slots__:
            # Determine appropriate indices by looking for 'i<varname>' in data
            indexvar = f"i{var}"
            if isinstance(data, dict) and indexvar in data:
                idxs = data[indexvar]
            elif (
                "__indexattributes__" in dir(data)
                and indexvar in data.__indexattributes__
            ):
                idxs = getattr(data, indexvar)
            else:
                idxs = np.array([])

            # Extract parameter values at the specified indices
            if idxs.size > 0:
                vals = parsarray[idxs]
            else:
                vals = np.array([])
            setattr(self, var, vals)

    def tree_flatten(self):
        """
        Flatten this object for JAX pytree registration.

        Returns all parameter arrays as children (traced by JAX) with no
        auxiliary data (compile-time constants).

        Returns
        -------
        tuple
            (children, aux_data) where children is a tuple of all parameter
            arrays and aux_data is an empty tuple.
        """
        children = tuple(getattr(self, x) for x in self.__slots__)
        aux_data = tuple()
        return (children, aux_data)

    @property
    def mappingaxes(self):
        """
        Determine which axes should be mapped over in vmap operations.

        For each parameter attribute, returns 0 if it should be mapped
        (varies across batch), None if it should be broadcast (same for all).
        Empty arrays always return None.

        Returns
        -------
        list
            List of axis specifications (0 or None) for each attribute in __slots__.
        """
        return [((0 if x in self.__ismapped__ else None) if getattr(self, x).size > 0 else None)
                for x in self.__slots__]

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a SALTparameters object from flattened pytree data.

        Parameters
        ----------
        aux_data : tuple
            Auxiliary data (unused, empty tuple).
        children : tuple
            Tuple of parameter arrays in __slots__ order.

        Returns
        -------
        SALTparameters
            Reconstructed parameter container.
        """
        self = cls.__new__(cls)
        for attr, val in zip(self.__slots__, children):
            setattr(self, attr, val)
        return self


class modeledtrainingdata(metaclass=abc.ABCMeta):
    """
    Abstract base class for modeled training data.

    Defines the interface for data objects that can compute model
    residuals and likelihoods. Subclasses must implement methods for
    flux prediction, variance computation, and residual calculation.

    Notes
    -----
    Subclasses are registered as JAX pytree nodes to enable JIT compilation.
    Static attributes are treated as compile-time constants; dynamic
    attributes can change between calls.
    """

    __slots__ = []

    @property
    @abc.abstractmethod
    def __staticattributes__(self):
        """List of which attributes should be considered static for jax compilation."""
        pass

    @property
    @abc.abstractmethod
    def __dynamicattributes__(self):
        """List of which attributes should be considered dynamic for jax compilation"""
        pass

    @abc.abstractmethod
    def modelresidual(
        self, x, cachedresults=None, fixuncertainties=False, fixfluxes=False
    ):
        """Calculate residuals and log-normalization term for this data given the model"""
        pass

    @abc.abstractmethod
    def modelfluxvariance(self, pars):
        """Calculate the predicted model variance given the data and model"""
        pass

    @abc.abstractmethod
    def modelflux(self, pars):
        """Calculate the predicted flux given the data and model"""
        pass

    def modelloglikelihood(
        self, x, cachedresults=None, fixuncertainties=False, fixfluxes=False
    ):
        """
        Compute the log-likelihood of the data given model parameters.

        Parameters
        ----------
        x : ndarray or SALTparameters
            Model parameters.
        cachedresults : optional
            Precomputed flux or variance values.
        fixuncertainties : bool
            If True, use cached uncertainties instead of recomputing.
        fixfluxes : bool
            If True, use cached fluxes instead of recomputing.

        Returns
        -------
        float
            Log-likelihood value.
        """
        resids = self.modelresidual(x, cachedresults, fixuncertainties, fixfluxes)

        return resids["lognorm"] - ((resids["residuals"] ** 2).sum() / 2.0)

    def tree_flatten(self):
        """
        Flatten this object for JAX pytree registration.

        Dynamic attributes become children (traced by JAX), static attributes
        become auxiliary data (compile-time constants).

        Returns
        -------
        tuple
            (children, aux_data) tuple for JAX pytree.
        """
        children = tuple(getattr(self, x) for x in self.__dynamicattributes__)
        aux_data = tuple(getattr(self, x) for x in self.__staticattributes__)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct object from flattened pytree data.

        Parameters
        ----------
        aux_data : tuple
            Static attribute values.
        children : tuple
            Dynamic attribute values.

        Returns
        -------
        modeledtrainingdata
            Reconstructed object.
        """
        self = cls.__new__(cls)
        for attr, val in zip(self.__dynamicattributes__, children):
            setattr(self, attr, val)
        for attr, val in zip(self.__staticattributes__, aux_data):
            setattr(self, attr, val)
        return self

    def unpack(self):
        """
        Extract all attributes as a tuple for batching.

        Returns
        -------
        tuple
            All attribute values in __slots__ order.
        """
        return tuple(getattr(self, x) for x in self.__slots__)

    @classmethod
    def repack(cls, data):
        """
        Reconstruct object from a tuple of attribute values.

        Parameters
        ----------
        data : tuple or list
            Attribute values in __slots__ order.

        Returns
        -------
        modeledtrainingdata
            Reconstructed object.
        """
        self = cls.__new__(cls)
        for attr, val in zip(self.__slots__, data):
            setattr(self, attr, val)
        return self

    @abc.abstractmethod
    def __len__(self):
        """Return the number of data points."""
        pass

    def determineneededparameters(self, modelobj):
        return []


@register_pytree_node_class
class modeledtraininglightcurve(modeledtrainingdata):
    __indexattributes__ = [
        "iCL",
        "ix0",
        "ic",
        "icoordinates",
        "icomponents",
        "imodelcorrs",
        "imodelerrs",
        "iclscat",
        "ipad",
        "isurverrfloor",
    ]

    __dynamicattributes__ = [
        "phase",
        "fluxcal",
        "fluxcalerr",
        "lambdaeff",
        "lambdaeffrest",
        "errordesignmat",
        "pcderivsparse",
        "varianceprefactor",
        "clscatderivs",
        "wavebasis",
        "padding",
    ] + __indexattributes__
    __staticattributes__ = [
        "preintegratebasis",
        "imodelcorrs_coordinds",
        "bsplinecoeffshape",
        "errorgridshape",
        "uniqueid",
        "colorlawfunction",
    ]

    __slots__ = __dynamicattributes__ + __staticattributes__

    __ismapped__ = {
        "ix0",
        "ic",
        "icoordinates",
        "isurverrfloor",
        "ipad",
        "phase",
        "fluxcal",
        "fluxcalerr",
        "lambdaeff",
        "lambdaeffrest",
        "errordesignmat",
        "pcderivsparse",
        "varianceprefactor",
        "clscatderivs",
        "padding",
        "uniqueid",
    }

    def __init__(self, sn, lc, residsobj, kcordict, padding=0):
        """
        Initialize a modeled light curve for SALT training.

        Precomputes B-spline basis function convolutions with the filter
        passband, design matrices for error surfaces, and other quantities
        needed for efficient flux and variance evaluation.

        Parameters
        ----------
        sn : SALTfitcacheSN
            Parent supernova object with MW extinction and redshift info.
        lc : SALTtraininglightcurve
            Raw light curve data (phases, fluxes, errors).
        residsobj : SALTResids
            Residuals object containing model configuration (knot locations,
            spline order, parameter indices, etc.).
        kcordict : dict
            K-correction dictionary with filter transmission curves.
        padding : int
            Number of zero-padding elements to add for batching alignment.
        """
        # Copy basic attributes from raw light curve data
        for attr in lc.__slots__:
            if attr in self.__slots__:
                setattr(self, attr, getattr(lc, attr))

        #######################################################################
        # SECTION 1: Basic setup and padding configuration
        #######################################################################
        z = sn.zHelio
        padding = max(0, padding)
        self.padding = padding
        # Boolean mask: True for padded elements, False for real data
        self.ipad = np.arange(len(lc) + padding) >= len(lc)
        self.uniqueid = f"{sn.snid}_{lc.filt}"

        #######################################################################
        # SECTION 2: Filter passband setup for synthetic photometry
        # Goal: Create a normalized passband response function (pbspl) that
        # converts spectral flux density to observed photon counts
        #######################################################################
        filtwave = kcordict[sn.survey][lc.filt]["filtwave"]
        filttrans = kcordict[sn.survey][lc.filt]["filttrans"]

        # Find wavelength range where filter and model overlap
        waveidxs = (sn.obswave >= kcordict[sn.survey][lc.filt]["minlam"]) & (
            sn.obswave <= kcordict[sn.survey][lc.filt]["maxlam"]
        )
        # Interpolate filter transmission onto model wavelength grid
        pbspl = np.interp(sn.obswave[waveidxs], filtwave, filttrans)
        # Multiply by wavelength (photon counting) and normalize
        # Division by HC_ERG_AA converts from energy to photon flux
        pbspl *= sn.obswave[waveidxs]
        denom = np.trapz(pbspl, sn.obswave[waveidxs])
        pbspl /= denom * HC_ERG_AA

        # Store filter effective wavelength (observer and rest frame)
        self.lambdaeff = kcordict[sn.survey][lc.filt]["lambdaeff"]
        self.lambdaeffrest = self.lambdaeff / (1 + z)

        # Shape of the 2D B-spline coefficient grid (phase x wavelength)
        self.bsplinecoeffshape = (
            residsobj.phaseBins[0].size,
            residsobj.waveBins[0].size,
        )

        self.preintegratebasis = residsobj.preintegrate_photometric_passband

        #######################################################################
        # SECTION 3: Parameter index setup
        # Store indices into the full parameter vector for this light curve
        #######################################################################
        self.icoordinates = sn.icoordinates  # x1, xhost indices for this SN
        self.icomponents = residsobj.icomponents  # M0, M1 component indices

        self.iCL = residsobj.iCL  # Color law coefficient indices
        self.ix0 = sn.ix0  # Amplitude parameter index
        self.ic = sn.ic  # Color parameter index
        self.iclscat = residsobj.iclscat  # Color scatter parameter indices

        # Model correlation indices: map coordinate pairs to parameter indices
        self.imodelcorrs = np.array([np.arange(x, y + 1)
                                     for x, y in zip(residsobj.corrmin, residsobj.corrmax)])
        # Convert 'host' labels to -1 index for array indexing
        self.imodelcorrs_coordinds = np.array([
            ((-1, comb[1]) if "host" == comb[0]
             else ((comb[0], -1) if "host" == comb[1] else comb))
            for comb in residsobj.corrcombinations])

        # Model error surface indices
        self.imodelerrs = np.array([np.arange(x, y + 1)
                                    for x, y in zip(residsobj.errmin, residsobj.errmax)])
        # Survey-specific error floor (if defined for this filter)
        self.isurverrfloor = np.where(residsobj.parlist == f"surverrfloor_{lc.filt}")[0]

        self.wavebasis = residsobj.wavebasis
        self.colorlawfunction = residsobj.colorlawfunction

        # Clip phases to model grid range for basis function evaluation
        clippedphase = np.clip(self.phase, residsobj.phase.min(), residsobj.phase.max())

        #######################################################################
        # SECTION 4: B-spline basis function evaluation and passband convolution
        # This is the core computation: precompute how each B-spline basis
        # function contributes to the observed flux through this filter.
        # Result: pcderivsparse matrix maps spline coefficients -> observed flux
        #######################################################################
        dwave = sn.dwave  # Wavelength step size
        fluxfactor = residsobj.fluxfactor[sn.survey][lc.filt]  # Zeropoint correction

        wave = residsobj.wave[waveidxs]

        # Determine which basis functions have non-zero support at our phases/wavelengths
        # This optimization avoids evaluating basis functions that contribute nothing
        inds = np.array(range(residsobj.im0.size))
        # Convert linear index to 2D (phase, wave) indices
        phaseind, waveind = inds // (
            residsobj.waveknotloc.size - residsobj.bsorder - 1
        ), inds % (residsobj.waveknotloc.size - residsobj.bsorder - 1)

        # Check which basis functions overlap with our observation phases
        inphase = ((clippedphase[:, np.newaxis] >= residsobj.phaseknotloc[np.newaxis, phaseind])
                   & (clippedphase[:, np.newaxis]
                      <= residsobj.phaseknotloc[np.newaxis, phaseind + residsobj.bsorder + 1])).any(axis=0)
        # Check which basis functions overlap with filter wavelength range
        inwave = (wave.max() >= residsobj.waveknotloc[waveind]) & (
            wave.min() <= residsobj.waveknotloc[waveind + residsobj.bsorder + 1]
        )

        # Only evaluate basis functions that overlap in both phase AND wavelength
        isrelevant = inphase & inwave

        # Evaluate each relevant B-spline basis function on the (phase, wavelength) grid
        # derivInterp[phase_idx, wave_idx, basis_idx] = basis function value
        derivInterp = np.zeros((clippedphase.size, waveidxs.sum(), residsobj.im0.size))
        for i in np.where(isrelevant)[0]:
            derivInterp[:, :, i] = bisplev(
                clippedphase, wave,
                (residsobj.phaseknotloc, residsobj.waveknotloc,
                 np.arange(residsobj.im0.size) == i,  # Unit vector for basis i
                 residsobj.bsorder, residsobj.bsorder))

        # Combine passband with MW extinction and flux scaling factors
        # This is the "effective passband" that converts flux to counts
        reddenedpassband = (
            sn.mwextcurve[waveidxs]  # MW extinction correction
            * pbspl  # Normalized passband response
            * dwave  # Wavelength integration element
            * fluxfactor  # Zeropoint/calibration factor
            * _SCALE_FACTOR  # Global scaling to avoid numerical issues
            / (1 + z)  # K-correction: (1+z) for flux density transformation
        )

        # Convolve each basis function with the passband for each observation
        splinebasisconvolutions = []
        for pdx in range(len(lc)):
            # Handle late-time photometry beyond model phase range
            # Extrapolate assuming linear magnitude decline
            if self.phase[pdx] > sn.obsphase.max():
                decayFactor = 10 ** (-0.4 * residsobj.extrapolateDecline
                                     * (self.phase[pdx] - sn.obsphase.max()))
            else:
                decayFactor = 1

            if self.preintegratebasis:
                # Preintegrate: dot product integrates over wavelength
                # Result is vector of length n_basis
                splinebasisconvolutions += [
                    decayFactor * (np.dot(derivInterp[pdx, :, :].T, reddenedpassband))
                ]
            else:
                # Keep wavelength dimension for later color law application
                # Result is matrix of shape (n_wavelength, n_basis)
                splinebasisconvolutions += [
                    decayFactor
                    * (derivInterp[pdx, :, :] * reddenedpassband[:, np.newaxis])
                ]

        # Store as sparse matrix for efficient JAX operations
        # Add zero rows for padding elements
        self.pcderivsparse = sparse.BCOO.fromdense(
            np.stack(
                splinebasisconvolutions
                + [np.zeros(splinebasisconvolutions[-1].shape)] * padding
            )
        )

        #######################################################################
        # SECTION 5: Error model setup
        # Build design matrix mapping error surface parameters to observation variances
        #######################################################################

        # Variance prefactor: scales error surface values to flux variance units
        self.varianceprefactor = (
            fluxfactor
            * (pbspl.sum())
            * dwave
            * _SCALE_FACTOR
            * sn.mwextcurveint(self.lambdaeff)
            / (1 + z)
        )

        # Shape of the error surface grid
        self.errorgridshape = (
            residsobj.errphaseknotloc.size - 1,
            residsobj.errwaveknotloc.size - 1,
        )

        if residsobj.errbsorder == 0:
            # Piecewise constant error model: each observation maps to one bin
            # Find which error bin each observation falls into
            errorwaveind = (
                np.searchsorted(residsobj.errwaveknotloc, self.lambdaeffrest) - 1
            )
            errorphaseind = np.searchsorted(residsobj.errphaseknotloc, clippedphase) - 1
            waveindtemp = np.array([errorwaveind for x in errorphaseind])
            # Convert 2D bin indices to linear index
            ierrorbin = np.ravel_multi_index(
                (errorphaseind, waveindtemp), self.errorgridshape
            )

            # Build sparse design matrix: row i has a 1 in column ierrorbin[i]
            errordesignmat = scisparse.lil_matrix(
                (len(lc) + padding, residsobj.imodelerr0.size)
            )
            errordesignmat[np.arange(0, len(lc)), ierrorbin] = 1
            self.errordesignmat = sparse.BCOO.from_scipy_sparse(errordesignmat)
        else:
            # B-spline error model: smooth interpolation of error surface
            inds = np.array(range(residsobj.imodelerr0.size))
            derivInterp = np.zeros((clippedphase.size, residsobj.imodelerr0.size))
            phaseind, waveind = inds // (
                residsobj.errwaveknotloc.size - residsobj.errbsorder - 1
            ), inds % (residsobj.errwaveknotloc.size - residsobj.errbsorder - 1)
            # Find relevant error basis functions (same logic as flux basis)
            inphase = ((clippedphase[:, np.newaxis] >= residsobj.phaseknotloc[np.newaxis, phaseind])
                       & (clippedphase[:, np.newaxis]
                          <= residsobj.phaseknotloc[np.newaxis, phaseind + residsobj.bsorder + 1])).any(axis=0)
            inwave = (self.lambdaeffrest >= residsobj.waveknotloc[waveind]) & (
                self.lambdaeffrest
                <= residsobj.waveknotloc[waveind + residsobj.bsorder + 1]
            )
            isrelevant = inphase & inwave
            for i in np.where(isrelevant)[0]:
                derivInterp[:, i] = bisplev(
                    clippedphase, self.lambdaeffrest,
                    (residsobj.errphaseknotloc, residsobj.errwaveknotloc,
                     np.arange(residsobj.imodelerr0.size) == i,
                     residsobj.errbsorder, residsobj.errbsorder))
            self.errordesignmat = sparse.BCOO.fromdense(
                np.concatenate((derivInterp, np.zeros((padding, residsobj.imodelerr0.size)))))

        #######################################################################
        # SECTION 6: Color scatter model setup
        # Polynomial basis for wavelength-dependent color scatter
        #######################################################################
        # Powers for Taylor expansion centered at 5500 Angstrom
        pow = self.iclscat.size - 1 - np.arange(self.iclscat.size)
        # Normalized wavelength coordinate (in units of 1000 Angstrom from 5500)
        colorscateval = (self.lambdaeffrest - 5500) / 1000

        # Derivatives of polynomial terms (divided by factorial for Taylor series)
        self.clscatderivs = ((colorscateval) ** (pow)) / factorial(pow)
        
        #######################################################################
        # SECTION 7: Color scatter model setup
        # Zeropad everything
        #######################################################################

        for attr in lc.__slots__:
            if attr in lc.__listdatakeys__ and attr in self.__slots__:
                setattr(
                    self, attr, np.concatenate((getattr(self, attr), np.zeros(padding)))
                )
        self.fluxcalerr[self.ipad] = 1

    def __len__(self):
        """Return the number of photometric observations (including padding)."""
        return self.fluxcal.size

    def modelflux(self, pars):
        """
        Compute predicted model flux for this light curve.

        Evaluates the SALT model flux by combining model components weighted
        by SN coordinates, applying the color law, and convolving with the
        precomputed passband response.

        Parameters
        ----------
        pars : ndarray or SALTparameters
            Model parameters (will be converted to SALTparameters if needed).

        Returns
        -------
        jnp.ndarray
            Predicted flux at each observation phase, clipped to non-negative.
        """
        if not isinstance(pars, SALTparameters):
            pars = SALTparameters(self, pars)
        # Evaluate the coefficients of the spline bases
        # coordinates = [1, x1, x2, ..., xhost] for combining model components
        coordinates = jnp.concatenate((jnp.ones(1), pars.coordinates))

        fluxcoeffs = jnp.dot(coordinates, pars.components) * pars.x0
        # Evaluate color law at the wavelength basis centers
        colorlaw = sum(
            [
                fun(c, cl, self.wavebasis)
                for fun, c, cl in zip(self.colorlawfunction, pars.c, pars.CL)
            ]
        )
        colorexp = 10.0 ** (-0.4 * colorlaw)

        if self.preintegratebasis:
            # Redden flux coefficients before passband integration
            fluxcoeffsreddened = (
                colorexp[np.newaxis, :] * fluxcoeffs.reshape(self.bsplinecoeffshape)
            ).flatten()
            # Multiply spline bases by flux coefficients
            return jnp.clip(self.pcderivsparse @ fluxcoeffsreddened, 0, None)
        else:
            # Integrate basis functions over wavelength and sum over flux coefficients
            return jnp.clip((self.pcderivsparse @ fluxcoeffs) @ colorexp, 0, None)

    def modelfluxvariance(self, pars):
        """
        Compute predicted model flux variance for this light curve.

        Evaluates the model uncertainty by combining error surfaces weighted
        by SN coordinates, including correlation terms, and scaling by the
        color law and amplitude.

        Parameters
        ----------
        pars : ndarray or SALTparameters
            Model parameters.

        Returns
        -------
        jnp.ndarray
            Predicted model flux variance at each observation phase.
        """
        if not isinstance(pars, SALTparameters):
            pars = SALTparameters(self, pars)

        # Evaluate color law at filter effective wavelength
        colorlaw = sum(
            [
                fun(c, cl, self.lambdaeffrest)
                for fun, c, cl in zip(self.colorlawfunction, pars.c, pars.CL)
            ]
        )
        colorexp = 10.0 ** (-0.4 * colorlaw)

        # Evaluate model uncertainty from error surfaces
        # Combine error surfaces weighted by coordinates squared
        coordinates = jnp.concatenate((jnp.ones(1), pars.coordinates))

        errorsurfaces = (
            (coordinates[: len(pars.modelerrs), np.newaxis] * pars.modelerrs) ** 2
        ).sum(axis=0)
        # Add correlation terms between different error surfaces
        for (i, j), correlation in zip(self.imodelcorrs_coordinds, pars.modelcorrs):
            errorsurfaces = (errorsurfaces + 2 * correlation * coordinates[i]
                             * coordinates[j] * pars.modelerrs[i] * pars.modelerrs[j])
        # Apply design matrix to get error at each observation
        errorsurfaces = self.errordesignmat @ errorsurfaces

        modelfluxvar = jnp.clip(
            colorexp**2 * self.varianceprefactor**2 * pars.x0**2 * errorsurfaces,
            0,
            None,
        )
        # Add survey-specific error floor if present
        if self.isurverrfloor.size > 0:
            return jnp.hypot(modelfluxvar, pars.surverrfloor * self.modelflux(pars))
        else:
            return modelfluxvar

    def colorscatter(self, pars):
        """
        Compute wavelength-dependent color scatter term.

        Evaluates a polynomial color scatter model at the filter effective
        wavelength to account for intrinsic SN color variation.

        Parameters
        ----------
        pars : ndarray or SALTparameters
            Model parameters.

        Returns
        -------
        jnp.ndarray
            Color scatter scaling factor.
        """
        if not isinstance(pars, SALTparameters):
            pars = SALTparameters(self, pars)
        return jnp.exp(self.clscatderivs @ pars.clscat)

    def modelloglikelihood(
        self, x, cachedresults=None, fixuncertainties=False, fixfluxes=False
    ):
        """
        Compute log-likelihood for this light curve given model parameters.

        Uses a multivariate normal likelihood with covariance that includes
        both diagonal variance terms and a rank-1 color scatter term.

        Parameters
        ----------
        x : ndarray or SALTparameters
            Model parameters.
        cachedresults : optional
            Precomputed flux or (variance, clscat) tuple.
        fixuncertainties : bool
            If True, use cached uncertainties.
        fixfluxes : bool
            If True, use cached fluxes.

        Returns
        -------
        float
            Log-likelihood value.
        """

        if fixfluxes:
            modelflux = cachedresults
        else:
            modelflux = self.modelflux(x)

        if fixuncertainties:
            if isinstance(cachedresults, tuple):
                modelvariance, clscat = cachedresults
            else:
                modelvariance, clscat = cachedresults, 0
        else:
            modelvariance = self.modelfluxvariance(x)
            clscat = self.colorscatter(x)

        variance = self.fluxcalerr**2 + modelvariance
        zeropoint = jax.scipy.stats.norm.logpdf(
            self.fluxcalerr * (~self.ipad), 0, self.fluxcalerr
        ).sum()
        return (
            mvn_likelihood_simplified_cov(
                self.fluxcal, modelflux, D_diag=variance, u=clscat * modelflux
            )
            - zeropoint
        )

    def modelresidual(
        self, x, cachedresults=None, fixuncertainties=False, fixfluxes=False
    ):
        """
        Compute normalized residuals and log-normalization term.

        When color scatter is present, uses Cholesky decomposition to
        compute whitened residuals accounting for off-diagonal covariance.

        Parameters
        ----------
        x : ndarray or SALTparameters
            Model parameters.
        cachedresults : optional
            Precomputed flux or (variance, clscat) tuple.
        fixuncertainties : bool
            If True, use cached uncertainties.
        fixfluxes : bool
            If True, use cached fluxes.

        Returns
        -------
        dict
            Dictionary with 'residuals' (whitened) and 'lognorm' (log
            normalization factor for the likelihood).
        """
        if fixfluxes:
            modelflux = cachedresults
        else:
            modelflux = self.modelflux(x)

        if fixuncertainties:
            if isinstance(cachedresults, tuple):
                modelvariance, clscat = cachedresults
            else:
                modelvariance, clscat = cachedresults, 0
        else:
            modelvariance = self.modelfluxvariance(x)
            clscat = self.colorscatter(x)

        variance = self.fluxcalerr**2 + modelvariance
        numresids = (~self.ipad).sum()
        zeropoint = -jnp.log(self.fluxcalerr).sum() - numresids / 2

        # Use Cholesky decomposition to handle correlated covariance from color scatter
        def choleskyresidsandnorm(variance, clscat, modelflux):
            cholesky = jaxlinalg.cholesky(
                jnp.diag(variance) + clscat**2 * jnp.outer(modelflux, modelflux),
                lower=True,
            )
            return {
                "residuals": jnp.nan_to_num(
                    jaxlinalg.solve_triangular(
                        cholesky, modelflux - self.fluxcal, lower=True
                    ),
                    nan=0,
                ),
                "lognorm": -jnp.log(jnp.diag(cholesky)).sum() - zeropoint,
            }

        return choleskyresidsandnorm(variance, clscat, modelflux)

    def dumptostring(self, pars):
        """
        Generate formatted string output of model vs data for debugging.

        Yields one line per observation with SN ID, filter, phase, wavelength,
        observed flux, error, model flux, model error, and residual.

        Parameters
        ----------
        pars : ndarray or SALTparameters
            Model parameters.

        Yields
        ------
        str
            Formatted string for each non-padded observation.
        """
        if not isinstance(pars, SALTparameters):
            pars = SALTparameters(self, pars)
        fluxes = self.modelflux(pars)
        variances = self.modelfluxvariance(pars)
        res = self.modelresidual(pars)
        for i in np.where(~self.ipad)[0]:
            yield (
                f"{self.uniqueid.split('_')[0]: >20} {self.uniqueid.split('_')[1]: >20} {self.phase[i]: >11.1f} {self.lambdaeffrest: >11.0f} {self.fluxcal[i]: >11.4e} {self.fluxcalerr[i]: >11.4e} {fluxes[i]: >11.4e} {np.sqrt(variances[i]): >12.4e} {res['residuals'][i]: >11.2e}"
            )


@register_pytree_node_class
class modeledtrainingspectrum(modeledtrainingdata):
    __indexattributes__ = [
        "ix0",
        "ispcrcl",
        "icomponents",
        "icoordinates",
        "ipad",
        "iCL",
        "ic",
        "imodelerrs",
        "imodelcorrs",
        "imodelcorrs_coordinds",
    ]
    __dynamicattributes__ = [
        "flux",
        "wavelength",
        "phase",
        "fluxerr",
        "restwavelength",
        "recaltermderivs",
        "varianceprefactor",
        "errordesignmat",
        "pcderivsparse",
        "spectralsuppression",
    ]
    __staticattributes__ = [
        "padding",
        "bsplinecoeffshape",
        "errorgridshape",
        "uniqueid",
        "colorlawfunction",
        "n_specrecal",
    ] + __indexattributes__
    __slots__ = __dynamicattributes__ + __staticattributes__

    __ismapped__ = {
        "ix0",
        "ic",
        "ispcrcl",
        "icoordinates",
        "ipad",
        "phase",
        "flux",
        "fluxerr",
        "restwavelength",
        "recaltermderivs",
        "pcderivsparse",
        "errordesignmat",
        "uniqueid",
        "n_specrecal",
    }

    def __init__(self, sn, spectrum, k, residsobj, padding=0):
        """
        Initialize a modeled spectrum for SALT training.

        Precomputes B-spline basis function evaluations at observed wavelengths,
        spectral recalibration polynomial terms, and design matrices for error
        surfaces.

        Parameters
        ----------
        sn : SALTfitcacheSN
            Parent supernova object with MW extinction and redshift info.
        spectrum : SALTtrainingspectrum
            Raw spectrum data (wavelengths, fluxes, errors).
        k : int or str
            Spectrum identifier (index or key) within the SN.
        residsobj : SALTResids
            Residuals object containing model configuration.
        padding : int
            Number of zero-padding elements to add for batching alignment.
        """
        # Copy basic attributes from raw spectrum data
        for attr in spectrum.__slots__:
            if attr in self.__slots__:
                setattr(self, attr, getattr(spectrum, attr))

        z = sn.zHelio
        self.n_specrecal = spectrum.n_specrecal

        #######################################################################
        # SECTION 1: Basic setup and parameter indices
        #######################################################################
        padding = max(0, padding)

        # Spectrum-specific amplitude parameter (allows independent flux scaling)
        self.ix0 = np.where(residsobj.parlist == f"specx0_{sn.snid}_{k}")[0][0]
        # Spectral recalibration polynomial coefficients for this spectrum
        self.ispcrcl = np.where(residsobj.parlist == f"specrecal_{sn.snid}_{k}")[0]

        # Shape of the 2D B-spline coefficient grid
        self.bsplinecoeffshape = (
            residsobj.phaseBins[0].size,
            residsobj.waveBins[0].size,
        )
        self.padding = padding
        # Boolean mask: True for padded elements, False for real data
        self.ipad = np.arange(len(spectrum) + padding) >= len(spectrum)
        self.uniqueid = f"{sn.snid}_{k}"

        # Spectral suppression factor: down-weights spectra relative to photometry
        # to balance their contributions to the total chi-squared
        # Factor is sqrt(N_phot/N_spec) * user scaling, capped at 1
        self.spectralsuppression = min(
            np.sqrt(residsobj.num_phot / residsobj.num_spec)
            * residsobj.spec_chi2_scaling,
            1,
        )

        # Store parameter indices for color law and SN-specific parameters
        self.iCL = residsobj.iCL
        self.ic = sn.ic
        self.colorlawfunction = residsobj.colorlawfunction
        self.icomponents = residsobj.icomponents
        self.icoordinates = sn.icoordinates

        # Evaluate MW extinction at observed wavelengths
        mwextcurve = sn.mwextcurveint(spectrum.wavelength)

        #######################################################################
        # SECTION 2: B-spline basis function evaluation
        # Unlike photometry (which integrates over filter), spectra sample the
        # model directly at each wavelength. Precompute basis function values.
        #######################################################################
        wave = self.restwavelength

        # Determine which basis functions have non-zero support at this phase/wavelength
        inds = np.array(range(residsobj.im0.size))
        # Convert linear index to 2D (phase, wave) indices
        phaseind, waveind = inds // (
            residsobj.waveknotloc.size - residsobj.bsorder - 1
        ), inds % (residsobj.waveknotloc.size - residsobj.bsorder - 1)

        # Check overlap with spectrum phase (single value for each spectrum)
        inphase = (self.phase >= residsobj.phaseknotloc[phaseind]) & (
            self.phase <= residsobj.phaseknotloc[phaseind + residsobj.bsorder + 1]
        )
        # Check overlap with spectrum wavelength range
        inwave = (wave.max() >= residsobj.waveknotloc[waveind]) & (
            wave.min() <= residsobj.waveknotloc[waveind + residsobj.bsorder + 1]
        )

        isrelevant = inphase & inwave

        # Evaluate each relevant basis function at all wavelengths
        # derivInterp[wave_idx, basis_idx] = basis function value
        derivInterp = np.zeros((spectrum.wavelength.size, residsobj.im0.size))
        for i in np.where(isrelevant)[0]:
            derivInterp[:, i] = bisplev(
                spectrum.phase, self.restwavelength,
                (residsobj.phaseknotloc, residsobj.waveknotloc,
                 np.arange(residsobj.im0.size) == i,  # Unit vector for basis i
                 residsobj.bsorder, residsobj.bsorder))

        # Apply MW extinction and flux scaling
        # Scale factor and (1+z) handle unit conversions
        derivInterp = (
            derivInterp * (_SCALE_FACTOR / (1 + z) * mwextcurve)[:, np.newaxis]
        )

        # Store as sparse matrix, add zero rows for padding
        self.pcderivsparse = sparse.BCOO.fromdense(
            np.concatenate((derivInterp, np.zeros((padding, residsobj.im0.size))))
        )

        #######################################################################
        # SECTION 3: Spectral recalibration polynomial setup
        # Recalibration absorbs flux calibration errors in observed spectra
        # using a polynomial in wavelength: exp(sum_i c_i * ((λ-λ_mean)/scale)^i / i!)
        #######################################################################
        # Powers for polynomial expansion (highest power first for stability)
        pow = self.ispcrcl.size - np.arange(self.ispcrcl.size)
        # Normalized wavelength coordinate centered on spectrum mean
        recalCoord = (
            self.wavelength - np.mean(self.wavelength)
        ) / residsobj.specrange_wavescale_specrecal

        # Build polynomial basis matrix: row i is basis evaluated at wavelength i
        # Division by factorial gives Taylor series coefficients
        self.recaltermderivs = (
            (recalCoord)[:, np.newaxis] ** (pow)[np.newaxis, :]
        ) / factorial(pow)[np.newaxis, :]
        # Add zero rows for padding
        self.recaltermderivs = np.concatenate((self.recaltermderivs, np.zeros((padding, pow.size))))

        #######################################################################
        # SECTION 4: Error model setup
        # Design matrix maps error surface parameters to observation variances
        #######################################################################
        # Variance prefactor: scales error surface to flux variance units
        self.varianceprefactor = (
            _SCALE_FACTOR * sn.mwextcurveint(self.wavelength) / (1 + z)
        )
        self.errorgridshape = (
            residsobj.errphaseknotloc.size - residsobj.errbsorder - 1,
            residsobj.errwaveknotloc.size - residsobj.errbsorder - 1,
        )
        # Add zeros for padding
        self.varianceprefactor = np.concatenate((self.varianceprefactor, np.zeros(padding)))

        if residsobj.errbsorder == 0:
            # Piecewise constant error model
            # Each wavelength maps to one bin in the error grid
            errorwaveind = (
                np.searchsorted(residsobj.errwaveknotloc, self.restwavelength) - 1
            )
            errorphaseind = np.searchsorted(residsobj.errphaseknotloc, self.phase) - 1
            # Same phase bin for all wavelengths (spectrum is at single phase)
            phaseindtemp = np.tile(errorphaseind, errorwaveind.size)
            ierrorbin = np.ravel_multi_index(
                (phaseindtemp, errorwaveind), self.errorgridshape
            )
            # Sparse design matrix: row i has 1 in column ierrorbin[i]
            errordesignmat = scisparse.lil_matrix(
                (len(spectrum) + padding, residsobj.imodelerr0.size)
            )
            errordesignmat[np.arange(0, len(spectrum)), ierrorbin] = 1
            self.errordesignmat = sparse.BCOO.from_scipy_sparse(errordesignmat)
        else:
            # B-spline error model: smooth interpolation
            inds = np.array(range(residsobj.imodelerr0.size))
            derivInterp = np.zeros(
                (spectrum.wavelength.size, residsobj.imodelerr0.size)
            )
            phaseind, waveind = inds // (
                residsobj.errwaveknotloc.size - residsobj.errbsorder - 1
            ), inds % (residsobj.errwaveknotloc.size - residsobj.errbsorder - 1)
            # Find relevant error basis functions
            inphase = (
                (self.phase >= residsobj.phaseknotloc[np.newaxis, phaseind])
                & (
                    self.phase
                    <= residsobj.phaseknotloc[
                        np.newaxis, phaseind + residsobj.errbsorder + 1
                    ]
                )
            ).any(axis=0)
            inwave = (self.restwavelength.max() >= residsobj.waveknotloc[waveind]) & (
                self.restwavelength.min()
                <= residsobj.waveknotloc[waveind + residsobj.errbsorder + 1]
            )
            isrelevant = inphase & inwave
            for i in np.where(isrelevant)[0]:
                derivInterp[:, i] = bisplev(
                    spectrum.phase, self.restwavelength,
                    (residsobj.errphaseknotloc, residsobj.errwaveknotloc,
                     np.arange(residsobj.imodelerr0.size) == i,
                     residsobj.errbsorder, residsobj.errbsorder))
            self.errordesignmat = sparse.BCOO.fromdense(
                np.concatenate((derivInterp, np.zeros((padding, residsobj.imodelerr0.size)))))

        #######################################################################
        # SECTION 5: Spectral error surface parameter indices
        # Note: spectra use separate error surfaces from photometry
        #######################################################################
        # Indices for spectral error surface parameters
        self.imodelerrs = np.array([np.where(residsobj.parlist == f"specmodelerr_{i}")[0]
                                    for i in range(residsobj.n_errorsurfaces)])
        # Indices for error correlation parameters
        self.imodelcorrs = np.array([np.where(residsobj.parlist == f"specmodelcorr_{i}{j}")[0]
                                     for i, j in residsobj.corrcombinations])
        # Convert 'host' labels to -1 for array indexing
        self.imodelcorrs_coordinds = np.array([
            ((-1, comb[1]) if "host" == comb[0]
             else ((comb[0], -1) if "host" == comb[1] else comb))
            for comb in residsobj.corrcombinations])

        #######################################################################
        # SECTION 6: Finalize data arrays with padding
        #######################################################################
        # Extend list-type data attributes with zero padding
        for attr in spectrum.__slots__:
            if attr in spectrum.__listdatakeys__ and attr in self.__slots__:
                setattr(
                    self, attr, np.concatenate((getattr(self, attr), np.zeros(padding)))
                )
        # Set padded error values to 1 to avoid division by zero
        self.fluxerr[self.ipad] = 1

    def __len__(self):
        """Return the number of spectral wavelength bins (including padding)."""
        return self.flux.size

    def modelflux(self, pars):
        """
        Compute predicted model flux for this spectrum.

        Evaluates the SALT spectral model by combining components weighted
        by SN coordinates, applying either color law or spectral recalibration
        depending on whether recalibration parameters are available.

        Parameters
        ----------
        pars : ndarray or SALTparameters
            Model parameters.

        Returns
        -------
        jnp.ndarray
            Predicted flux at each wavelength bin.
        """
        if not isinstance(pars, SALTparameters):
            pars = SALTparameters(self, pars)
        x0 = pars.x0
        # Define recalibration factor
        coeffs = pars.spcrcl
        coordinates = jnp.concatenate((jnp.ones(1), pars.coordinates))
        components = pars.components

        recalterm = jnp.dot(self.recaltermderivs, coeffs)
        recalterm = jnp.clip(recalterm, -recalmax, recalmax)
        recalexp = jnp.exp(recalterm)

        colorlaw = sum(
            [
                fun(c, cl, self.restwavelength)
                for fun, c, cl in zip(self.colorlawfunction, pars.c, pars.CL)
            ]
        )
        colorexp = 10.0 ** (-0.4 * colorlaw)

        fluxcoeffs = jnp.dot(coordinates, components) * x0

        return jax.lax.cond(
            self.n_specrecal == 0, lambda: colorexp, lambda: recalexp
        ) * (self.pcderivsparse @ (fluxcoeffs))

    def modelfluxvariance(self, pars):
        """
        Compute predicted model flux variance for this spectrum.

        Evaluates the spectral model uncertainty by combining error surfaces
        weighted by SN coordinates, including correlation terms.

        Parameters
        ----------
        pars : ndarray or SALTparameters
            Model parameters.

        Returns
        -------
        jnp.ndarray
            Predicted model flux variance at each wavelength bin.
        """
        if not isinstance(pars, SALTparameters):
            pars = SALTparameters(self, pars)
        x0 = pars.x0
        # Define recalibration factor
        coeffs = pars.spcrcl
        coordinates = jnp.concatenate((jnp.ones(1), pars.coordinates))
        errs = pars.modelerrs

        recalterm = jnp.dot(self.recaltermderivs, coeffs)
        recalterm = jnp.clip(recalterm, -recalmax, recalmax)
        recalexp = jnp.exp(recalterm)

        # Evaluate model uncertainty from error surfaces
        errorsurfaces = ((coordinates[: len(errs), np.newaxis] * errs) ** 2).sum(axis=0)
        # Add correlation terms between different error surfaces
        for (i, j), correlation in zip(self.imodelcorrs_coordinds, pars.modelcorrs):
            errorsurfaces = (errorsurfaces + 2 * correlation * coordinates[i]
                             * coordinates[j] * errs[i] * errs[j])
        errorsurfaces = self.errordesignmat @ errorsurfaces
        modelfluxvar = recalexp**2 * self.varianceprefactor**2 * x0**2 * errorsurfaces
        return jnp.clip(modelfluxvar, 0, None)

    def modelresidual(
        self, x, cachedresults=None, fixuncertainties=False, fixfluxes=False
    ):
        """
        Compute normalized residuals and log-normalization term.

        Spectral residuals are scaled by the spectral suppression factor to
        balance photometric and spectroscopic contributions to the likelihood.

        Parameters
        ----------
        x : ndarray or SALTparameters
            Model parameters.
        cachedresults : optional
            Precomputed flux or variance values.
        fixuncertainties : bool
            If True, use cached uncertainties.
        fixfluxes : bool
            If True, use cached fluxes.

        Returns
        -------
        dict
            Dictionary with 'residuals' (normalized, suppressed) and 'lognorm'
            (log normalization factor for the likelihood).
        """
        if fixfluxes:
            modelflux = cachedresults
        else:
            modelflux = self.modelflux(x)

        if fixuncertainties:
            modelvariance = cachedresults
        else:
            modelvariance = self.modelfluxvariance(x)

        variance = self.fluxerr**2 + modelvariance

        uncertainty = jnp.sqrt(variance)

        numresids = (~self.ipad).sum()
        zeropoint = -jnp.log(self.fluxerr).sum() - numresids / 2

        return {
            "residuals": jnp.nan_to_num(
                self.spectralsuppression * (modelflux - self.flux) / uncertainty, nan=0
            ),
            "lognorm": (self.spectralsuppression**2)
            * (-jnp.log(uncertainty).sum() - zeropoint),
        }

    def determineneededparameters(self, modelobj):
        return []


class SALTfitcacheSN(SALTtrainingSN):
    """
    Cached supernova data container for efficient SALT model fitting.

    Extends SALTtrainingSN with precomputed quantities (MW extinction curves,
    parameter indices) and converts raw light curve and spectrum data to
    modeledtraininglightcurve and modeledtrainingspectrum objects.

    Attributes
    ----------
    ix0, ix1, ic, ixhost : int or ndarray
        Indices into the parameter vector for SN-specific parameters.
    icoordinates : ndarray
        Indices for all coordinate parameters (x1, xhost, etc.).
    mwextcurve : ndarray
        Milky Way extinction curve evaluated on the model wavelength grid.
    mwextcurveint : callable
        Interpolator for MW extinction at arbitrary wavelengths.
    photdata : dict
        Dictionary of modeledtraininglightcurve objects keyed by filter.
    specdata : dict
        Dictionary of modeledtrainingspectrum objects keyed by spectrum ID.
    """

    __slots__ = [
        "ix0",
        "ix1",
        "ic",
        "ixhost",
        "icoordinates",
        "mwextcurve",
        "mwextcurveint",
        "dwave",
        "obswave",
        "obsphase",
        "photdata",
        "specdata",
        "zHelio",
        "snid",
    ]

    def __init__(
        self,
        sndata,
        residsobj,
        kcordict,
        lcpaddingsizes=None,
        specpaddingsizes=None,
        n_specrecal=None,
    ):
        """
        Initialize cached SN data from raw training data.

        Parameters
        ----------
        sndata : SALTtrainingSN
            Raw supernova training data.
        residsobj : SALTResids
            Residuals object with model configuration and parameter list.
        kcordict : dict
            K-correction dictionary with filter transmission curves.
        lcpaddingsizes : list of int, optional
            Available padding sizes for light curves (for batching).
        specpaddingsizes : list of int, optional
            Available padding sizes for spectra (for batching).
        n_specrecal : int, optional
            Number of spectral recalibration parameters (unused, for compat).
        """
        # Copy scalar attributes from raw SN data (skip photdata/specdata for now)
        for attr in sndata.__slots__:
            if attr == "photdata" or attr == "specdata":
                pass
            else:
                setattr(self, attr, getattr(sndata, attr))

        # Normalize padding size inputs to lists
        if isinstance(lcpaddingsizes, int):
            lcpaddingsizes = [lcpaddingsizes]
        if isinstance(specpaddingsizes, int):
            specpaddingsizes = [specpaddingsizes]

        #######################################################################
        # SECTION 1: Observer-frame wavelength and phase grids
        # Transform rest-frame model grid to observer frame for this SN
        #######################################################################
        self.obswave = residsobj.wave * (1 + self.zHelio)
        self.obsphase = residsobj.phase * (1 + self.zHelio)
        # Wavelength step size in observer frame
        self.dwave = residsobj.wave[1] * (1 + self.zHelio) - residsobj.wave[0] * (
            1 + self.zHelio
        )

        #######################################################################
        # SECTION 2: Milky Way extinction correction
        # Precompute extinction curve using Fitzpatrick99 law with R_V=3.1
        #######################################################################
        self.mwextcurve = 10 ** (
            -0.4 * extinction.fitzpatrick99(self.obswave, sndata.MWEBV * 3.1)
        )
        # Interpolator for extinction at arbitrary wavelengths (e.g., filter centers)
        self.mwextcurveint = interp1d(
            self.obswave,
            self.mwextcurve,
            kind=residsobj.interpMethod,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )

        #######################################################################
        # SECTION 3: Parameter index lookup
        # Find indices into the full parameter vector for this SN's parameters
        #######################################################################
        # Amplitude parameter (x0)
        self.ix0 = np.where(residsobj.parlist == f"x0_{self.snid}")[0][0]
        # Light curve shape parameter (x1)
        self.ix1 = np.where(residsobj.parlist == f"x1_{self.snid}")[0][0]
        # Host mass parameter (optional)
        self.ixhost = np.where(residsobj.parlist == f"xhost_{self.snid}")[0]
        if len(self.ixhost):
            self.ixhost = self.ixhost[0]
        # Color parameters (may be multiple for multi-component color laws)
        self.ic = np.array(
            [
                np.where(residsobj.parlist == f"c{i}_{self.snid}")[0][0]
                for i in range(residsobj.ncl)
            ]
        )

        # Combined coordinate indices: [x1, x2, ..., xhost] for model component weighting
        self.icoordinates = np.array(
            [
                np.where(residsobj.parlist == f"x{i}_{self.snid}")[0][0]
                for i in range(1, residsobj.n_components)
            ]
            + ([self.ixhost] if residsobj.host_component else [])
        )

        #######################################################################
        # SECTION 4: Convert raw data to modeled data objects
        # Each light curve and spectrum gets wrapped in a class that precomputes
        # quantities for efficient model evaluation
        #######################################################################

        def choosesmallestpadsize(padsizes, datasize):
            """Select minimum padding size that accommodates the data.

            For batching with vmap, all data in a batch must have the same size.
            This function finds the smallest available padded size >= datasize.
            """
            if padsizes is None:
                return 0
            else:
                padneeded = np.array(padsizes) - datasize
                padneeded = padneeded[padneeded >= 0]
                try:
                    return np.min(padneeded)
                except ValueError:
                    raise ValueError(
                        f"Data of length {datasize} is longer than requested zero-padded length of {max(padsizes)}"
                    )

        # Create modeledtraininglightcurve for each filter's light curve
        self.photdata = {
            flt: modeledtraininglightcurve(
                self,
                sndata.photdata[flt],
                residsobj,
                kcordict,
                choosesmallestpadsize(lcpaddingsizes, len(sndata.photdata[flt])),
            )
            for flt in sndata.photdata
        }

        # Create modeledtrainingspectrum for each spectrum
        self.specdata = {
            k: modeledtrainingspectrum(
                self,
                sndata.specdata[k],
                k,
                residsobj,
                choosesmallestpadsize(specpaddingsizes, len(sndata.specdata[k])),
            )
            for k in sndata.specdata
        }

    def determineneededparameters(self, modelobj):
        paramsneeded = [f"x{i}_{self.snid}" for i in range(modelobj.n_components)] + [
            f"c_{self.snid}"
        ]
        for k in self.specdata.keys():
            paramsneeded += self.specdata[k].determineneededparameters(self, modelobj)
        return paramsneeded

    @partial(
        jaxoptions,
        static_argnums=[3, 4],
        static_argnames=["fixuncertainties", "fixfluxes"],
        diff_argnum=1,
    )
    def modelloglikelihood(self, *args, **kwargs):
        """
        Compute total log-likelihood for this SN across all data.

        Sums log-likelihoods from all light curves and spectra.

        Parameters
        ----------
        *args, **kwargs
            Passed to individual light curve and spectrum modelloglikelihood methods.

        Returns
        -------
        float
            Total log-likelihood for this supernova.
        """
        return sum(
            [lc.modelloglikelihood(*args, **kwargs) for lc in self.photdata.values()]
        ) + sum(
            [
                spec.modelloglikelihood(*args, **kwargs)
                for spec in self.specdata.values()
            ]
        )

    def dumptostring(self, pars):
        """
        Generate formatted string output of model vs data for all photometry.

        Parameters
        ----------
        pars : ndarray or SALTparameters
            Model parameters.

        Returns
        -------
        str
            Formatted multi-line string with all photometric observations.
        """
        result = ""
        for data in self.photdata.values():
            result += "\n".join(list(data.dumptostring(pars))) + "\n"
        result = result[:-1]
        return result
