************************
Training the SALT3 Model
************************

usage::

  TrainSALT.py -c <configfile> <options>
  
To run an example, change to the "examples" directory
in the main package and run::

  TrainSALT.py -c SALT.conf

This will use the lightcurves and spectra in the :code:`examples/exampledata/`
directory and the kcor files in the :code:`examples/kcor/` directory to
run the SALT3 training, writing outputs to the :code:`examples/output/`
directory.

Training on SNANA Simulations
=============================

There are several "SNANA" simulations included in the examples directory
for training.  One of these is already set in the :code:`SALT_SNANA.conf`
parameters file.  The relevant data are in :code:`exampledata/snana/photdata`
and :code:`exampledata/snana/photdata_nox1` for a simple model where
the training only attempts to find the first principal component.
For this simplified model, run::

  TrainSALT.py -c SALT_SNANA.conf --n_components 1

Additional examples will be added as the code matures.


SALT3 Training Configuration Options
====================================

See the SALT.conf file in the :code:`examples/` directory
for a list of training options.
Descriptions of each option are given below.

=============================  ================  ======================================================================================================================================
Name                           Default           Description                                                                             
=============================  ================  ======================================================================================================================================
[iodata]
snlist                                           ASCII file with list of SN files (see :ref:`data-format` for input file format)
speclist                                         List of spectra (not sure if this is getting used currently)
outputdir                                        Directory for trained model outputs
initmodelfile                  Hsiao07.dat       Initial SN SED model.  Initial parameter guesses are derived from this file.  Default is the Hsiao model
initbfilt                      Bessell90_B.dat   Nominal *B*-filter for putting priors on the normalization

[survey_<:code:`SURVEY`>]                        The parameters file requires a category for **every** :code:`SURVEY` key in SN data files
kcorfile                                         Kcorfile for each :code:`SURVEY` key in SN data files
subsurveylist                                    Comma-separated list of sub-surveys for every survey, e.g. :code:`CFA4` and :code:`CFA3` are part of survey :code:`PS1_LOWZ_COMBINED`

[mcmcparams]                                     **In most cases, leave these alone**
n_steps_mcmc                   10000             Number of MCMC steps
n_burnin_mcmc                  8000              Number of MCMC steps before saving the chain
stepsize_magscale_M0           0.01              Initial step size for M0 (mag units)
stepsize_magadd_M0             0.001             Initial step size for M0 (additive)
stepsize_magscale_M1           0.05              Initial step size for M0 (mag units)
stepsize_magadd_M1             0.01              Initial step size for M1 (additive)
stepsize_magscale_err          0.0001            Initial step size for phase-dependent error model (mag units)
stepsize_cl                    0.1               Initial step size for color law
stepsize_magscale_clscat       0.01              Initial step size for color scatter model
stepsize_specrecal             1.0               Initial step size in spectral recalibration
stepsize_x0                    0.005             Initial step size in x0 (mag units)
stepsize_x1                    0.05              Initial step size in x1
stepsize_c                     0.01              Initial step size in color
stepsize_tpk                   0.05              Initial step size in time of max
nsteps_before_adaptive         2000              Number of steps before starting the adaptive metropolis-hastings algorithm
nsteps_adaptive_memory         200               Number of recent MH steps to use when adjusting the MH step sizes
adaptive_sigma_opt_scale       3                 Scale factor to multiply the step sizes suggested by adaptive MCMC

						
[trainparams]                                    **In most cases, leave these alone**
waverange                      2000,9200         Wavelength range over which the model is defined                                        
colorwaverange                 2800,7000         Wavelength range over which the color law polynomial is fit                             
interpfunc                     bspline           Function for interpolating the model between control points (b-spline is default)   
interporder                    3                 Order of the spline interpolation
wavesplineres                  72                Number of Angstroms between wavelength control points
waveoutres                     2                 Wavelength resolution of the trained output model (Angstroms)
phaserange                     -14,50            Phase range over which the model is defined (days)
phasesplineres                 3.2               Wavelength resolution of the trained output model (days)
phaseoutres                    2                 Phase resolution of the trained output model (days)
n_colorpars                    4                 Number of parameters used to define the color law polynomial
n_colorscatpars                0                 Number of parameters used to define the color scatter law
n_components                   2                 Number of principal components (1 or 2)
n_specrecal                    4                 Number of spectral recalibration parameters
n_processes                    1                 Number of parallel processes for the training
estimate_tpk                                     
regulargradientphase           0                 Regularization parameters (gradient)
regulargradientwave            0                 Regularization parameters (gradient)
regulardyad                    0                 Regularization parameters (dyadic)
n_min_specrecal                1                
specrange_wavescale_specrecal  2500             
n_specrecal_per_lightcurve     0.34             
filter_mass_tolerance          0.01              Amount of filter "mass" allowed to be outside the SALT wavelength range
error_snake_phase_binsize      5                 Spacing in days for the SALT error model B-spline interpolation
error_snake_wave_binsize       600               Spacing in Angstroms for the SALT error model B-spline interpolation
n_components                   2                 Number of principal components
n_processes                    1                 Number of worker processes to spawn to calculate the chi2

=============================  ================  ======================================================================================================================================
