.. _configuration:

=============
Configuration
=============

SALTShaker uses a two-level configuration system: a main configuration file
that specifies paths and key training options, and three secondary configuration files
file containing logging options, model structure, and optimizer hyperparameters. Command-line arguments
can override any configuration option.

Configuration files use INI format with sections denoted by ``[section_name]``.

Usage
=====

Basic usage with a configuration file::

    trainsalt -c myconfig.conf

Override specific options from the command line::

    trainsalt -c myconfig.conf --binspec True --maxsn 50

Configuration Sections
======================

.. contents:: Sections
   :local:
   :depth: 1


[iodata] - Input/Output Options
-------------------------------

These options control input data files, output directories, and data selection.

Input Files
^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``snlists``
     - path(s)
     - **Required.** Path to ASCII file(s) listing supernova data files.
       Multiple files can be comma-separated. Each file contains paths to
       SNANA-format photometry and spectroscopy files. See :ref:`data-format`.
   * - ``tmaxlist``
     - path
     - File containing time of maximum light for each SN. Space-delimited
       with columns: SNID, tmax, tmaxerr. See example files in
       ``examples/SALT3TRAIN_K21_PUBLIC/``.
   * - ``snparlist``
     - path
     - Initial SN parameters from a SALT fit. Columns: SNID,
       zHelio, x0, x1, c, FITPROB. The FITPROB column is used for quality cuts.
   * - ``specrecallist``
     - path
     - *Deprecated.* Initial spectral recalibration parameters. No longer recommended.
   * - ``calibrationshiftfile``
     - path
     - File specifying adjustments to filter zeropoints and central wavelengths,
       used for systematic uncertainty studies.
   * - ``calibrationcovariance``
     - path
     - File containing calibration covariance matrix for filters.
   * - ``loggingconfig``
     - path
     - YAML file configuring logging output. Default: ``logging.yaml``.
   * - ``trainingconfig``
     - path
     - Path to secondary configuration file with training hyperparameters.
       Default: ``training.conf`` (searches package directory if not found locally).
   * - ``modelconfig``
     - path
     - Path to secondary configuration file describing model construction.

Output Files
^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``outputdir``
     - path
     - **Required.** Directory for trained model outputs. Will contain M0, M1,
       color law, error model, and validation plots.
   * - ``yamloutputfile``
     - path
     - File for YAML summary of the training process for use by SNANA. Default: ``/dev/null``.
   * - ``trainingcachefile``
     - path
     - Cache file for pre-processed training data. If exists, loads cached data;
       otherwise writes cache after processing. Speeds up subsequent runs.

Data Selection
^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``dospec``
     - bool
     - If True, include spectroscopy in training. Default: True.
   * - ``maxsn``
     - int/None
     - Limit training to this many SNe. Useful for debugging. Default: None (all SNe).
   * - ``keeponlyspec``
     - bool
     - If True, only train on SNe with spectroscopic data. Default: False.
   * - ``filter_mass_tolerance``
     - float
     - Fraction of filter transmission allowed outside model wavelength range.
       Filters exceeding this are excluded. Default: 0.01.
   * - ``spectra_cut``
     - float
     - Minimum median S/N for including spectra. Default: 0 (no cut), but this is recommended.
   * - ``filtercen_obs_waverange``
     - float float
     - Observed-frame wavelength range (Angstroms) for filter central wavelengths.
       Filters outside this range are excluded.

Model Initialization
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``initsalt2model``
     - bool
     - If True, initialize M0/M1 from the built-in SALT2 model. Default: True.
   * - ``initsalt2var``
     - bool
     - If True, initialize error model from SALT2. Not recommended as SALT3
       uses a different error prescription. Default: False.
   * - ``initm0modelfile``
     - path
     - Custom initial M0 model (ASCII: phase, wavelength, flux columns).
       Default: Hsiao07.dat.
   * - ``initm1modelfile``
     - path
     - Custom initial M1 model (ASCII: phase, wavelength, flux columns).
       If not provided, M1 is derived from a time-dilated M0.
   * - ``initbfilt``
     - path
     - B-filter definition for normalization. Default: Bessell90_B.dat.
   * - ``resume_from_outputdir``
     - path
     - Resume training from a previous output directory. Uses saved parameters
       as initial values.
   * - ``resume_from_gnhistory``
     - path
     - Resume from a ``gaussnewtonhistory.pickle`` file. Useful for recovering
       from crashes.
   * - ``error_dir``
     - path
     - Directory with previous error files, for use with ``use_previous_errors``.
   * - ``fix_salt2components_initdir``
     - path
     - Initialize component parameters from this directory without fitting them.

Validation Options
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``validate_modelonly``
     - bool
     - If True, only generate model validation plots (skip SN-by-SN plots).
       Faster and avoids occasional crashes. Default: False.
   * - ``use_previous_errors``
     - bool
     - Use error model from previous run instead of recomputing. Default: False.
   * - ``filters_use_lastchar_only``
     - bool
     - Use only final character of filter names. Workaround for some SNANA files.
       Default: False.
   * - ``calib_survey_ignore``
     - bool
     - If True, ignore survey names when applying calibration shifts. Default: False.


[survey_<NAME>] - Survey Definitions
------------------------------------

Each survey in your data requires a ``[survey_<NAME>]`` section where ``<NAME>``
matches the SURVEY keyword in your SNANA files.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``kcorfile``
     - path
     - **Required.** K-correction file defining filters, zeropoints, and primary
       standards for this survey.
   * - ``subsurveylist``
     - str
     - Comma-separated list of subsurveys. E.g., for survey
       ``PS1_LOWZ_COMBINED(CFA4)``, set ``subsurveylist = CFA4``.
   * - ``ignore_filters``
     - str
     - Comma-separated list of filter names to exclude from training.

Example::

    [survey_CFA3]
    kcorfile = kcor/kcor_CFA3.fits
    subsurveylist =
    ignore_filters = U

    [survey_PS1_LOWZ_COMBINED]
    kcorfile = kcor/kcor_PS1.fits
    subsurveylist = CFA3S,CFA3K,CFA4p1,CFA4p2


[trainparams] - Training Parameters
-----------------------------------

Core parameters controlling the optimization process.

Optimizer Settings
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``optimizer``
     - str
     - Optimizer algorithm to use. Default: Gauss-Newton.
   * - ``gaussnewton_maxiter``
     - int
     - Maximum Gauss-Newton iterations before stopping (if convergence not reached).
       Default: 30.
   * - ``n_repeat``
     - int
     - *Deprecated.* Leave at 1.
   * - ``regularize``
     - bool
     - Enable regularization terms in the loss function. Default: True.
   * - ``fitprobmin``
     - float
     - Minimum SALT2 FITPROB for including SNe. SNe with lower fit probability
       are excluded. Default: varies by config.
   * - ``fitsalt2``
     - bool
     - Fit SN parameters with SALT2 model during validation as a cross-check.
       Default: False.
   * - ``fixedparams``
     - str
     - Comma-separated list of parameter names to hold fixed during training.
   * - ``preintegrate_photometric_passband``
     - bool
     - If True, pre-integrate color law over passbands for speed. Approximation
       that may reduce accuracy slightly. Default: False.

Error Estimation
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``fit_model_err``
     - bool
     - If True, fit model uncertainties during training. Default: True.
   * - ``fit_cdisp_only``
     - bool
     - If True and ``fit_model_err`` is True, only fit color scatter (not full
       error model). Default: False.
   * - ``steps_between_errorfit``
     - int
     - Estimate model errors every N iterations. Error estimation is slow
       (~4.5 hours), so increasing this speeds up training. Default: 5.
   * - ``model_err_max_chisq``
     - float
     - Only begin error estimation when reduced chi-squared drops below this.
       Default: 4.
   * - ``errors_from_hessianapprox``
     - bool
     - Get model surface errors from approximate Hessian matrix. Default: False.
   * - ``errors_from_bootstrap``
     - bool
     - Get model surface errors from bootstrap resampling. Default: False.
   * - ``n_bootstrap``
     - int
     - Number of bootstrap resamples. Default: varies.
   * - ``maxiter_bootstrap``
     - int
     - Maximum Gauss-Newton iterations per bootstrap resample. Default: varies.
   * - ``bootstrap_batch_mode``
     - bool
     - Run bootstrap in batch mode (for cluster computing). Default: False.
   * - ``bootstrap_sbatch_template``
     - str
     - SLURM batch template for bootstrap jobs.
   * - ``get_bootstrap_output_only``
     - bool
     - Collect bootstrap output without running new jobs. Default: False.

Memory/Performance
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``photometric_zeropadding_batches``
     - int
     - Number of batches for photometric zero-padding. Increase to reduce memory
       at cost of speed. Default: 1.
   * - ``spectroscopic_zeropadding_batches``
     - int
     - Number of batches for spectroscopic zero-padding. Increase to reduce memory
       at cost of speed. Default: 1.
   * - ``usesurverrfloors``
     - bool
     - Fit error floors for each survey/filter combination. Default: False.

Gauss-Newton Optimizer
^^^^^^^^^^^^^^^^^^^^^^

These options control the Gauss-Newton optimizer (default). Set ``optimizer = gaussnewton``
in ``[trainparams]`` to use this optimizer.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``gaussnewton_maxiter``
     - int
     - Maximum number of Gauss-Newton iterations before stopping. Training
       will end early if convergence is achieved. Default: 30.
   * - ``fitting_sequence``
     - str
     - Order in which parameter groups are fit within each iteration.
       Options: ``all``, ``pcaparams``, ``color``, ``colorlaw``,
       ``spectralrecalibration``, ``sn``. Use comma-separated list for
       custom sequence, or ``default`` for standard approach.
       Default: default.
   * - ``dampingscalerate``
     - float
     - Controls how quickly the Levenberg-Marquardt damping parameter is
       adjusted during optimization. Higher values allow faster adaptation
       but may cause instability.
   * - ``lsmrmaxiter``
     - int
     - Maximum iterations allowed for the LSMR linear solver within each
       Gauss-Newton step. LSMR solves the linearized least-squares problem.
   * - ``preconditioningmaxiter``
     - int
     - Number of operations used to evaluate preconditioning for the linear
       system. Preconditioning improves convergence of the iterative solver.
   * - ``preconditioningchunksize``
     - int
     - Batch size for evaluating preconditioning scales. Increasing may
       improve memory performance at cost of speed.
   * - ``fit_tpkoff``
     - bool
     - *Deprecated.* Previously allowed fitting time-of-maximum offset as a
       free parameter. This feature is no longer supported. Default: False.
   * - ``no_transformed_err_check``
     - bool
     - For host-mass SALTShaker: ignore x1/xhost de-correlation error issues.
       Bootstrap errors are required if enabled. Default: False.


RProp Optimizer (Gradient Descent)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options control the RProp with backtracking optimizer, an alternative
gradient-based method. Set ``optimizer = rpropwithbacktracking`` in
``[trainparams]`` to use this optimizer. Options are specified in a
``[rpropconfig]`` section in the training config file.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``gradientmaxiter``
     - int
     - Maximum number of gradient descent iterations allowed before
       termination.
   * - ``burninmaxiter``
     - int
     - Maximum iterations for the burn-in phase, which fits the flux model
       before enabling full parameter optimization. Default: 100.
   * - ``learningratesinitscale``
     - float
     - Global scale factor applied to initial learning rates. Higher values
       mean larger initial steps.
   * - ``searchsize``
     - float
     - Step size for backtracking line search, expressed as a fraction.
       Must be between 0 and 1.
   * - ``searchtolerance``
     - float
     - Armijo criterion tolerance for line search. Smaller values impose
       looser constraints on step acceptance. Must be between 0 and 1.
   * - ``etaminus``
     - float
     - Factor by which to decrease learning rates when the gradient changes
       sign (indicating overshoot). Must be between 0 and 1.
   * - ``etaplus``
     - float
     - Factor by which to increase learning rates when the gradient maintains
       direction (indicating efficient descent). Must be greater than 1.
   * - ``convergencetolerance``
     - float
     - Convergence threshold. Optimization terminates when the change in loss
       is consistently below this value. Must be greater than 0.
   * - ``memorydebug``
     - bool
     - Enable JAX memory profiling. Writes memory profiles to the output
       directory for debugging memory issues. Default: False.


[trainingparams] - Training Hyperparameters
-------------------------------------------

*Located in training.conf.* Low-level hyperparameters that rarely need modification.

Time of Maximum
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``estimate_tpk``
     - bool
     - If True and no ``tmaxlist`` is provided, estimate time of maximum for
       each SN by fitting a Bazin function to the B-band (or g-band) light
       curve. Useful when peak times are unknown. The estimated values are
       held fixed during training. Default: False.

Spectral Recalibration
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``specrecal``
     - bool
     - Enable spectral recalibration. Default: True.
   * - ``n_min_specrecal``
     - int
     - Minimum polynomial order for spectral recalibration. Default: 4.
   * - ``n_max_specrecal``
     - int
     - Maximum polynomial order for spectral recalibration. Default: 4.
   * - ``specrange_wavescale_specrecal``
     - float
     - Wavelength scale for recalibration normalization. Default: 2500.
   * - ``n_specrecal_per_lightcurve``
     - float
     - Add one recalibration parameter per this many photometric bands. Default: 0.5.
   * - ``recalprior``
     - float
     - Prior width constraining recalibration parameters. Default: 50.

Regularization
^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``regulargradientphase``
     - float
     - Amplitude of gradient regularization penalty in phase direction.
       *Semi-arbitrary scaling.* Default: 1e4.
   * - ``regulargradientwave``
     - float
     - Amplitude of gradient regularization penalty in wavelength direction.
       *Semi-arbitrary scaling.* Default: 1e5.
   * - ``regulardyad``
     - float
     - Amplitude of dyadic regularization penalty. *Semi-arbitrary.* Default: 1e4.
   * - ``m1regularization``
     - float
     - Multiplier for M1 regularization amplitude. Default: 100.
   * - ``mhostregularization``
     - float
     - Multiplier for host-mass component regularization. Default: 100.
   * - ``regularizationScaleMethod``
     - str
     - Method for adjusting regularization scale. Options in ``saltresids.py``.
       Default: fixed.
   * - ``wavesmoothingneff``
     - float
     - Gaussian smoothing scale for N_eff in wavelength. Default: 1.
   * - ``phasesmoothingneff``
     - float
     - Gaussian smoothing scale for N_eff in phase. Default: 3.
   * - ``nefffloor``
     - float
     - Below this N_eff, regularization stops increasing. Default: 1e-4.
   * - ``neffmax``
     - float
     - Above this N_eff, regularization is turned off. Default: 0.1.

Spectral Processing
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``binspec``
     - bool
     - Bin spectra to reduce data volume (~10x fewer points). Speeds up training
       significantly with minimal impact on results. Default: True.
   * - ``binspecres``
     - int
     - Resolution (number of bins) for spectral binning. Default: 29.
   * - ``spec_chi2_scaling``
     - float
     - Scale factor so spectra and photometry contribute equally to chi-squared.
       Default: 0.5.


[modelparams] - Model Structure
-------------------------------

*Located in training.conf.* Defines the structure of the SALT3 model.

Wavelength Grid
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``waverange``
     - float,float
     - Rest-frame wavelength range (Angstroms) for the model. Default: 2000,11000.
   * - ``colorwaverange``
     - float,float
     - Wavelength range for fitting the color law polynomial. Default: 2800,8000.
   * - ``wavesplineres``
     - float
     - Spacing (Angstroms) between wavelength B-spline control points. Default: 69.3.
   * - ``waveinterpres``
     - float
     - Wavelength resolution used during training. Default: 10.
   * - ``waveoutres``
     - float
     - Wavelength resolution of output model files. Default: 10.

Phase Grid
^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``phaserange``
     - float,float
     - Rest-frame phase range (days relative to B-max) for the model.
       Default: -20,50.
   * - ``phasesplineres``
     - float
     - Spacing (days) between phase B-spline control points. Default: 3.0.
   * - ``phaseinterpres``
     - float
     - Phase resolution used during training. Default: 0.2.
   * - ``phaseoutres``
     - float
     - Phase resolution of output model files. Default: 1.

Interpolation
^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``interpfunc``
     - str
     - Interpolation function type. Default: bspline.
   * - ``interporder``
     - int
     - B-spline order for model interpolation. Default: 3.
   * - ``errinterporder``
     - int
     - B-spline order for error model interpolation. Default: 0.
   * - ``use_snpca_knots``
     - bool
     - Use knot locations from SALT2 training. Default: False.

Model Components
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``n_components``
     - int
     - Number of SED components (M0, M1). Currently must be 2. Default: 2.
   * - ``host_component``
     - str
     - Host-mass component configuration. Leave empty for standard SALT3.
   * - ``n_colorpars``
     - int
     - Number of color law polynomial parameters. Default: 5.
   * - ``n_colorscatpars``
     - int
     - Number of color scatter polynomial parameters. Default: 5.
   * - ``colorlaw_function``
     - str
     - Color law functional form. Default: colorlaw_default.
   * - ``error_snake_phase_binsize``
     - float
     - Phase spacing (days) for error model B-spline. Default: 6.
   * - ``error_snake_wave_binsize``
     - float
     - Wavelength spacing (Angstroms) for error model B-spline. Default: 1200.


[priors] - Prior Constraints
----------------------------

*Located in training.conf.* Each key is a decorator name from ``training/priors.py``;
the value sets the prior width.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Prior
     - Default
     - Description
   * - ``x1mean``
     - 0.1
     - Prior enforcing mean(x1) = 0.
   * - ``x1std``
     - 0.1
     - Prior enforcing std(x1) = 1.
   * - ``m0endalllam``
     - 1e-2
     - Prior forcing M0 flux to zero at the earliest phase in the model
       (typically -20 days). Penalizes non-zero flux before explosion.
   * - ``m1endalllam``
     - 1e-2
     - Prior forcing M1 flux to zero at the earliest phase in the model
       (typically -20 days). Penalizes non-zero flux before explosion.
   * - ``colorstretchcorr``
     - 1e-4
     - Prior enforcing zero correlation between color and stretch.
   * - ``colormean``
     - 1e-3
     - Prior enforcing mean(c) = 0.
   * - ``m0positiveprior``
     - 1e-2
     - Prior preventing negative M0 flux.
   * - ``recalprior``
     - 50
     - Prior constraining spectral recalibration magnitudes.


[bounds] - Parameter Bounds
---------------------------

*Located in training.conf.* Constrain parameter values during optimization.
Format: ``parameter = lower, upper, prior_width``.

Parameter options include:

- **Nominal:** x0, x1, c, m0, m1, tpk
- **Spectral recalibration:** spcrcl, spcrcl_norm, spcrcl_poly
- **Uncertainties:** modelerr, modelcorr, clscat, clscat_0, clscat_poly

Example::

    [bounds]
    x1 = -5,5,0.01
    x0 = 0,inf,1e-5


[init_offsets] - Initial Parameter Offsets
------------------------------------------

*Located in training.conf.* Apply global shifts to parameter initial values
before optimization begins. This is useful for systematically offsetting
starting points during testing or when resuming from a previous run with
known biases.

Format: ``parameter = shift_value``

Example::

    [init_offsets]
    x1 = 0.1
    c = -0.02


Command-Line Options
====================

These options are only available from the command line, not configuration files.

.. list-table::
   :widths: 20 60
   :header-rows: 1

   * - Option
     - Description
   * - ``-c``, ``--configfile``
     - Path to configuration file.
   * - ``-v``, ``--verbose``
     - Increase verbosity. Can be repeated (``-vv``).
   * - ``--debug``
     - Enable debug mode with additional output and diagnostic files.
   * - ``--clobber``
     - Overwrite existing output directory.
   * - ``-s``, ``--stage``
     - Run specific stage: ``train``, ``validate``, or ``all`` (default).
   * - ``--skip_validation``
     - Skip validation plot generation.
   * - ``--fast``
     - Fast mode for debugging (reduced iterations).
   * - ``--bootstrap_single``
     - Run single bootstrap iteration and save to outputdir.
   * - ``-g``, ``--get-example-data``
     - Download example training data.


Example Configuration
=====================

Minimal configuration file::

    [iodata]
    snlists = snlist.txt
    tmaxlist = tmax.list
    snparlist = snparams.list
    outputdir = output/

    [survey_CFA3]
    kcorfile = kcor/kcor_CFA3.fits

    [survey_CSP]
    kcorfile = kcor/kcor_CSP.fits

Full example with common options::

    [iodata]
    snlists = data/snlist_training.txt
    tmaxlist = data/SALT3_PKMJD_INIT.LIST
    snparlist = data/SALT3_PARS_INIT.LIST
    outputdir = output_salt3/
    dospec = True
    initsalt2model = True
    trainingconfig = training.conf
    filter_mass_tolerance = 0.01

    [survey_CFA3]
    kcorfile = kcor/kcor_CFA3.fits
    subsurveylist =
    ignore_filters = U

    [survey_Foundation]
    kcorfile = kcor/kcor_Foundation.fits
    subsurveylist =

    [trainparams]
    regularize = True
    gaussnewton_maxiter = 30
    steps_between_errorfit = 5
    binspec = True


See Also
========

- :ref:`training` for a training tutorial
- :ref:`data-format` for input data format specifications
- Example configurations in ``examples/SALT3TRAIN_K21_PUBLIC/``
