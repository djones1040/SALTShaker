Tensor Train Surface Parameterization
======================================

SALTShaker supports an alternative Tensor Train (TT) parameterization for the
M0/M1 spectral surfaces. The TT decomposition represents surfaces as low-rank
factorizations, reducing the number of free parameters while optionally adding
a host-galaxy stellar mass dimension.

Quick Start
-----------

To use TT surfaces, add these options to the ``[modelparams]`` section of your
model config file::

    # Use TT instead of B-spline surfaces
    surface_type = tt
    tt_rank = 10

    # Optional: add host-mass dimension (set to 0 to disable)
    tt_mass_bins = 10
    tt_mass_range = 7.0,12.0

Then run training as usual::

    trainsalt -c your_training.conf

Everything else (data format, optimizer, constraints, priors) works the same as
with B-spline surfaces.

Configuration Options
---------------------

``surface_type``
    Surface parameterization type. ``bspline`` (default) or ``tt``.

``tt_rank``
    TT decomposition rank. Controls the expressiveness of the surface
    approximation. Rank 5-10 is a good starting point. Higher rank = more
    parameters but closer to the full B-spline representation. Default: 5.

``tt_mass_bins``
    Number of host-mass grid points for the mass dimension. Set to 0
    (default) to disable the mass dimension.

``tt_mass_range``
    Range of log10(M★/M☉) for the mass grid. Default: ``7.0,12.0``.

How It Works
------------

**2D (no mass dimension)**:

The M0 surface is decomposed as:

.. math::

    M_0(p, \lambda) \approx \sum_{k=1}^{r} U_k(p) \cdot V_k(\lambda)

where *r* is the TT rank, and U, V are the TT "cores" (phase and wavelength
factors). During training, the cores are optimized directly. For photometric
model evaluation, the cores are expanded to full B-spline coefficients
(``core_phase @ core_wave``) so the existing ``pcderivsparse`` infrastructure
works unchanged.

**3D (with mass dimension)**:

When ``tt_mass_bins > 0``, a mass core W is added:

.. math::

    M_0(p, \lambda, M_\star) \approx \sum_{j,k} U_j(p) \cdot W_{jk}(M_\star) \cdot V_k(\lambda)

The mass core is initialized as identity (no mass dependence) and the model
learns mass-dependent spectral variations during training. Host masses are read
from the ``HOST_LOGMASS`` field in SNANA FITS headers.

**Initialization**:

TT cores are initialized by performing a truncated SVD on the B-spline
initialization surface (e.g., Hsiao template). This means TT training starts
from a reasonable approximation of the standard model.

Example: Training with Host Mass
---------------------------------

Here is a minimal model config for TT training with a host-mass dimension::

    [modelparams]
    waverange = 2000,11000
    colorwaverange = 2800,8000
    interpfunc = bspline
    interporder = 3
    wavesplineres = 69.3
    waveinterpres = 10
    waveoutres = 10
    phaserange = -20,50
    phasesplineres = 3.0
    phaseinterpres = 0.2
    phaseoutres = 1
    n_colorpars = 5
    n_colorscatpars = 5
    n_components = 2
    host_component =
    error_snake_phase_binsize = 6
    error_snake_wave_binsize = 1200
    use_snpca_knots = False
    colorlaw_function = colorlaw_default
    constraints = centeranddecorrelatedcolorsandcoords, fixbbandfluxes
    secondary_constraints =

    # TT surface with host-mass dimension
    surface_type = tt
    tt_rank = 10
    tt_mass_bins = 10
    tt_mass_range = 7.0,12.0

The training data must include host galaxy masses in the SNANA FITS headers
(``HOST_LOGMASS`` field). SNe without host mass information will use a default
value of 10.0.

Output
------

TT training produces the same output files as standard B-spline training:

- ``salt3_template_0.dat``, ``salt3_template_1.dat`` — M0/M1 surfaces
- ``salt3_color_correction.dat`` — Color law
- ``salt3train_snparams.txt`` — Fitted SN parameters
- ``SALTmodelcomp.png`` — Model comparison plot
- ``parametercovariance.npy`` — Parameter covariance (if Hessian enabled)

The output templates are standard SALT3 format and can be used directly with
light curve fitters like ``sncosmo``.
