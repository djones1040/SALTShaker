"""Validation plots for the host-mass correction to the SALT3 components."""

import os

import matplotlib.pyplot as plt
import numpy as np
import sncosmo
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm

from saltshaker.training.masscorrection import readmasscorrection, surfacesatmass

BANDS = ("bessellux", "bessellb", "bessellv", "bessellr", "besselli")
BANDLABELS = ("U", "B", "V", "R", "I")
BANDCOLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SPECTRUMPHASES = (-10, 0, 10, 30)
# Knot masses are a magnitude, so they share one hue from light to dark
KNOTCOLORS = LinearSegmentedColormap.from_list(
    "knots", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"])
INK, MUTED = "#0b0b0b", "#52514e"


def bandfluxes(surface, wave, bands=BANDS):
    """Rest-frame photon fluxes of a flux surface in each band, up to a factor per band, shape (len(bands), n_phase)."""
    fluxes = []
    for band in bands:
        bandpass = sncosmo.get_bandpass(band)
        trans = np.interp(wave, bandpass.wave, bandpass.trans, left=0, right=0)
        fluxes.append(np.trapz(surface * trans * wave, wave, axis=-1))
    return np.array(fluxes)


def deltamagnitudes(surface, reference, wave, minfraction=0.1):
    """
    Magnitudes of a flux surface relative to a reference surface in each band, shape (len(BANDS), n_phase).

    NaN where the reference's band flux is below minfraction of its maximum over phase,
    since small changes there give large magnitude differences.
    """
    flux, ref = bandfluxes(surface, wave), bandfluxes(reference, wave)
    bright = ref > minfraction * ref.max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bright & (flux > 0), -2.5 * np.log10(flux / ref), np.nan)


def readtemplates(outputdir, n_components, phase, wave):
    """Trained component surfaces, which are the sample-average model, shape (n_components, n_phase, n_wave)."""
    return np.array([
        np.loadtxt(os.path.join(outputdir, f"salt3_template_{i}.dat"), usecols=[2]).reshape(phase.size, wave.size)
        for i in range(n_components)])


def plotmasscorrection(outfile, outputdir, logmasses=None, xlimits=None):
    """
    Plot the host-mass correction saved in outputdir.

    The top row shows the training sample's host masses with the knots and the fitted
    population, and the change in rest-frame UBVRI magnitudes of M0 (x1 = 0, c = 0)
    with host mass at peak and with phase at the end knots, where the average band flux
    is at least a tenth of its peak. The lower rows show the change in M0 and M1 at each
    knot, relative to the peak of the average M0 at each phase. Changes are relative to
    the sample-average model, salt3_template_*.dat.

    Parameters
    ----------
    outfile : str
        Output figure path.
    outputdir : str
        Training output directory with salt3_masscorrection.npz.
    logmasses : array_like, optional
        Host log masses of the training sample; NaN for no mass.
    xlimits : sequence of two floats, optional
        Wavelength range of the spectral panels.
    """
    result, phase, wave = readmasscorrection(outputdir)
    knots = np.asarray(result.knots, dtype=float)
    average = readtemplates(outputdir, result.surfaces.shape[1], phase, wave)
    knotcolors = KNOTCOLORS(np.linspace(0, 1, knots.size))

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(3, len(SPECTRUMPHASES), hspace=0.35, wspace=0.35)
    axes = np.empty((3, len(SPECTRUMPHASES)), dtype=object)
    for row in range(3):
        for col in range(len(SPECTRUMPHASES)):
            # The two Delta m against phase panels share a scale
            axes[row, col] = fig.add_subplot(grid[row, col], sharey=axes[0, 2] if (row, col) == (0, 3) else None)
    for ax in axes.flat:
        ax.tick_params(colors=MUTED, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_color(MUTED)

    ax = axes[0, 0]
    popmean, popspread = result.population
    lo, hi = min(knots[0], popmean - 3 * popspread), max(knots[-1], popmean + 3 * popspread)
    if logmasses is not None:
        logmasses = np.asarray(logmasses, dtype=float)
        measured = logmasses[np.isfinite(logmasses)]
        lo, hi = min(lo, measured.min()), max(hi, measured.max())
    massgrid = np.linspace(lo - 0.5, hi + 0.5, 200)
    if logmasses is not None:
        bins = np.arange(massgrid[0], massgrid[-1] + 0.2, 0.2)
        ax.hist(measured, bins=bins, color="#9ec5f4", edgecolor="#fcfcfb", linewidth=0.5,
                label=f"{measured.size} hosts with masses")
        scale = measured.size * (bins[1] - bins[0])
        title = f"{logmasses.size - measured.size} of {logmasses.size} SNe without a mass"
    else:
        scale, title = 1.0, ""
    ax.plot(massgrid, scale * norm.pdf(massgrid, popmean, popspread), color=INK, linewidth=2,
            label=f"population {popmean:.2f} $\\pm$ {popspread:.2f}")
    for knot, color in zip(knots, knotcolors):
        ax.axvline(knot, color=color, linewidth=2, linestyle=":")
    ax.set_xlabel(r"host $\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel("SNe" if logmasses is not None else "density")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(top=1.4 * ax.get_ylim()[1])
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    peak = np.argmin(np.abs(phase))
    curvemasses = np.linspace(knots[0] - 0.5, knots[-1] + 0.5, 41)
    deltapeak = 1000 * np.array([deltamagnitudes(surfacesatmass(result, mass)[0], average[0], wave)[:, peak]
                                 for mass in curvemasses])
    ax = axes[0, 1]
    for i, (label, color) in enumerate(zip(BANDLABELS, BANDCOLORS)):
        ax.plot(curvemasses, deltapeak[:, i], color=color, linewidth=2, label=label)
    for knot in knots:
        ax.axvline(knot, color=MUTED, linewidth=0.5, linestyle=":")
    ax.axhline(0, color=MUTED, linewidth=0.5)
    ax.set_xlabel(r"host $\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel(r"$\Delta m$ at peak (mmag)")
    ax.legend(fontsize=8, frameon=False, ncol=5, columnspacing=0.8, handlelength=1.2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))

    for ax, k in zip(axes[0, 2:], (0, knots.size - 1)):
        delta = 1000 * deltamagnitudes(result.surfaces[k, 0], average[0], wave)
        for i, (label, color) in enumerate(zip(BANDLABELS, BANDCOLORS)):
            ax.plot(phase, delta[i], color=color, linewidth=2, label=label)
        ax.axhline(0, color=MUTED, linewidth=0.5)
        ax.set_xlabel("phase (days)")
        ax.set_title(f"$\\Delta m$ at log mass {knots[k]:g}", fontsize=10)
    axes[0, 2].set_ylabel(r"$\Delta m$ (mmag)")
    axes[0, 3].tick_params(labelleft=False)

    for row, component in zip(axes[1:], ("M0", "M1")):
        c = 0 if component == "M0" else 1
        for ax, spectrumphase in zip(row, SPECTRUMPHASES):
            p = np.argmin(np.abs(phase - spectrumphase))
            normalization = np.max(np.abs(average[0, p]))
            for k, color in enumerate(knotcolors):
                ax.plot(wave, (result.surfaces[k, c, p] - average[c, p]) / normalization, color=color,
                        linewidth=1.5, label=f"{knots[k]:g}")
            ax.axhline(0, color=MUTED, linewidth=0.5)
            if xlimits is not None:
                ax.set_xlim(xlimits)
            ax.set_title(f"{component}, phase {phase[p]:g} d", fontsize=10)
            ax.set_xlabel(r"wavelength ($\AA$)")
        row[0].set_ylabel(rf"$\Delta${component} / max M0")
        row[-1].legend(title="log mass", fontsize=8, title_fontsize=8, frameon=False)

    fig.suptitle("Host-mass correction relative to the sample-average model", color=INK)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
