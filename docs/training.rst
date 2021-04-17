************************
Training the SALT3 Model
************************

usage::

  trainsalt -c <configfile> <options>
  
Although there are a number of training configuration files
in the :code:`examples/` directory, the simplest way to train the
SALT3.K21 model with all data and spectra and with the latest calibrations
is to use the configuration files and data in the
:code:`examples/SALT3TRAIN_K21_PUBLIC` directory.

To train the SALT3.K21 model, run::

  trainsalt -c Train_SALT3_public.conf

This directory contains all the lightcurves, spectra, and
filter definition files needed to train the model, with outputs
in the :code:`output` directory.

The training is slow given the large data volume and takes approximately
1 to 1.5 days, but can be sped up with a couple reasonable choices.
The first is changing the :code:`steps_between_errorfit` argument
to estimate model uncertainties less frequently, as uncertainty estimation
(~4.5 hours) is the slowest component of the code::

  trainsalt -c Train_SALT3_public.conf --steps_between_errorfit 15

Another option is to bin the spectra, which will reduce the amount of
spectroscopic data points by an order of magnitude::

  trainsalt -c Train_SALT3_public.conf --binspec True

This should not result in any noticeable difference to the model surfaces
but hasn't yet been tested fully.  Additional speed and memory usage
improvements are currently in progress.

SALT3 Training Configuration Options
====================================

See the :code:`examples/SALT3TRAIN_K21_PUBLIC/Train_SALT3_public.conf` file
and the :code:`examples/SALT3TRAIN_K21_PUBLIC/training.conf` files for
the full list of training options.  Two configuration files are used with the
goal that users should rarely have to modify the default :code:`training.conf` options.
Descriptions of each option are given below.

=============================  ================  ====================================================================================================================================================
Name                           Default           Description                                                                             
=============================  ================  ====================================================================================================================================================
**main config file**
[iodata]
snlists                                          ASCII file or comma-separated list of files.  Each file contains a list of SN files (see :ref:`data-format` for input file format)
tmaxlist                                         Time of maximum light for each SN in training.  See :code:`examples/SALT3TRAIN_K21_PUBLIC/SALT3_PKMJD_INIT.LIST`.
snparlist                                        initial list x0,x1,c and FITPROB (prob. that the data matches model, from SALT2). See :code:`examples/SALT3TRAIN_K21_PUBLIC/SALT3_PARS_INIT.LIST`
specrecallist                                    Option to provide an initial set of spectral recalibration parameters.  No longer recommended.
dospec                         True              If set, use spectra in training
maxsn                          None              Debug option to limit the training to a given number of SNe
outputdir                                        Directory for trained model outputs
keeponlyspec                   False             Debug option - keep only those SNe with spectroscopic data
initm0modelfile                Hsiao07.dat       Initial SN SED model.  Initial parameter guesses are derived from this file.  Default is the Hsiao model.
initm1modelfile                                  Initial SN SED model.  Will guess M1 from a time-dilated Hsiao model if no file is given.
initsalt2model                 True              If True, use SALT2 as the initial guess.  Otherwise use initm0modelfile.
initsalt2var                   False             If set, initialize model uncertainties using SALT2 values.  No longer recommended as SALT3 error prescription is different.
initbfilt                      Bessell90_B.dat   Nominal *B*-filter for putting priors on the normalization
resume_from_outputdir                            Resume the training from an existing output directory
resume_from_gnhistory                            If resume_from_outputdir is set, set to same directory name to resume training from a gnhistory.pickle file.  This is useful if training crashes.
loggingconfig                  logging.yaml      Gives configuration options for the training logs
trainingconfig                 training.conf     Additional configuration file.  Will look in the package directory if it's not found in the current directory
calibrationshiftfile                             A file that can adjust the calibration of the input files, e.g. for estimating systematics
filter_mass_tolerance          0.01              Amount of filter "mass" allowed to be outside the SALT wavelength range
fix_salt2modelpars             False             Debug option - if True, does not fit for M0 and M1.
validate_modelonly             False             If True, only produces model validation plots but not plots spectra or lightcurves (slow, and occasionally crashes).

[survey_<:code:`SURVEY`>]                        The parameters file requires a category for **every** :code:`SURVEY` key in SN data files
kcorfile                                         Kcorfile (includes filter ZPT offsets and filter definitions) for each :code:`SURVEY` key in SN data files
subsurveylist                                    Comma-separated list of sub-surveys for every survey, e.g. :code:`CFA4` is the subsurvey for survey name :code:`PS1_LOWZ_COMBINED(CFA4)`

[trainparams]
gaussnewton_maxiter            30                Maximum number of Gauss-Newton iterations allowed if convergence (delta chi^2 < 1) is not reached
regularize                     True              Include regularization if True
fitsalt2                       False             Try to fit SN parameters with SALT2 model in the validation stage if True
n_repeat                       1                 **deprecated, leave alone**
fit_model_err                  True              If True, fits model errors every :code:`steps_between_errorfit` iterations
fit_cdisp_only                 False             If True and :code:`fit_model_err` is True, fits for the color scatter but no other model errors
steps_between_errorfit         5                 Estimate model errors every x iterations
model_err_max_chisq            4                 Begin estimating model errors when the reduced chi^2 of the training is below this
condition_number               1e-80             Conditioning matrices for the Gauss-Newton process.  Leave this alone.
fit_tpkoff                     False             if true, fit for time of maximum light along with other parameters (not well tested yet)
fitting_sequence               all               optionally, can fit for different model components in sequence.  Can make it hard for training to converge


**training.conf file**                           **In most cases, leave these alone**
[trainingparams]
specrecal                      1                 if 1 (or True), do the spectral recalibration
n_processes                    1                 **deprecated**
estimate_tpk                   False             **not recommended** estimate time of maximum light for each SN before beginning the training.  Not robust.
fix_t0                         False             **deprecated**
n_min_specrecal                3                 minimum number of parameters for the spectral recalibration polynomial
n_max_specrecal                10                maximum number of parameters for the spectral recalibration polynomial
regulargradientphase           1e4               amplitude of gradient regularization chi^2 penalty for phase (**semi-arbitrary**)
regulargradientwave            1e5               amplitude of gradient regularization chi^2 penalty for wavelength (**semi-arbitrary**)
regulardyad                    1e4               amplitude of dyadic regularization chi^2 penalty (**semi-arbitrary**)
m1regularization               100               multiply regularization amplitude for the M1 component by this amount (**semi-arbitrary**)
specrange_wavescale_specrecal  2500              normalizes the spectra for recalibration
n_specrecal_per_lightcurve     0.5               add one spectral recal parameter for every two photometric bands in a given SN
regularizationScaleMethod      fixed             options for adjusting regularization scale in :code:`training/saltresids.py`
wavesmoothingneff              1                 Gaussian smoothing scale for the amount of training data at each wavelength for smoothly varying Neff
phasesmoothingneff             3                 Gaussian smoothing scale for the amount of training data at each phase for smoothly varying Neff
nefffloor                      1e-4              below nefffloor, regularization does not continue to increase in strength
neffmax                        0.01              above neffmax, regularization is turned off
binspec                        False             use spectral binning if True
binspecres                     29                resolution of the spectral binning
spec_chi2_scaling              0.5               tuned so that spectra and photometry contribute ~equally to total chi^2 in training

[modelparams]
waverange                      2000,11000        wavelength range over which the model is defined
colorwaverange                 2800,8000         wavelength range over which the color law polynomial is fit
interpfunc                     bspline           function for interpolating the model between control points (b-spline is default)
errinterporder                 0                 order of the spline interpolation for the errors
interporder                    3                 order of the spline interpolation for the model
wavesplineres                  69.3              number of Angstroms between wavelength control points
waveinterpres                  10                wavelength resolution of the model used during training (Angstroms)
waveoutres                     10                wavelength resolution of the trained model written to output directory (Angstroms)
phaserange                     -20,50            phase range over which the model is defined (rest-frame days)
phasesplineres                 3.0               phase resolution of the trained output model (days)
phaseinterpres                 0.2               phase resolution of the model used during training (days)
phaseoutres                    1                 phase resolution of the trained model written to output directory (days)
n_colorpars                    5                 number of parameters used to define the color law polynomial
n_colorscatpars                5                 number of parameters used to define the color scatter
n_components                   2                 number of model components (M0, M1) - additional components not yet allowed
error_snake_phase_binsize      6                 spacing in days for the SALT error model B-spline interpolation
error_snake_wave_binsize       1200              spacing in Angstroms for the SALT error model B-spline interpolation
use_snpca_knots                False             if true, use the knot locations from the SALT2 training

[priors]                                         key is the name of a decorator in :code:`training/priors.py`; value determines the (semi-arbitrary) width of each prior
x1mean                         0.1               mean x1 = 0
x1std                          0.1               standard deviation of x1 values = 1
m0endalllam                    1e-5              at -20 days, M0 must go to zero flux
m1endalllam                    1e-4              at -20 days, M1 must go to zero flux
colorstretchcorr               1e-4              color and stretch should not be correlated
colormean                      1e-3              mean sample color is zero
m0positiveprior                1e-2              M0 is not allowed to be negative
recalprior                     50                don't allow spectral recalibration to go crazy

[bounds]
x1                             -5,5,0.01         min,max,prior width on x1

=============================  ================  ====================================================================================================================================================
