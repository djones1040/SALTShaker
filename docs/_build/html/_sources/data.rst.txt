*******************************************
Spectroscopic and Photometric Training Data
*******************************************

SALTShaker input files use `SNANA <http://http://snana.uchicago.edu/>`_ format,
which allows easy synergy between model training and SN simulations, light-curve
fitting, and systematic uncertainty estimation.  The SNANA-formatted data
necessary for training includes photometry, spectroscopy, and filter
functions/photometric system information.

For photometry and spectroscopy, a number of light curves and spectra are provided
in the :code:`examples/SALT3TRAIN_K21_PUBLIC/` directory
for training.  Light curves and spectra are
combined into a single file.  The training data themselves are described
in Kenworthy et al., 2021.

For the photometric information, so-called "kcor" files - which confusingly contain no *k*-corrections - are given in the
:code:`examples/SALT3TRAIN_K21_PUBLIC/kcor` directory.  These FITS-formatted files define
the photometric system associated with each survey that comprises the training sample.
The SNANA function :code:`kcor.exe` will create these files from the :code:`.input` files in the
same directory if anything needs to be adjusted.  "kcor" files contain filter transmission
functions, AB, BD17, or Vega spectra depending on the photometric system of the data, zeropoint offsets,
and optional shifts to the central wavelength of each filter.

.. _data-format:

==================================
Photometry and Spectroscopy Format
==================================

`SNANA <http://http://snana.uchicago.edu/>`_ file format
consists of a number of header keys giving information
about each SN, followed by photometry and spectroscopy.

An example of the minimum required header is below::

  SURVEY: FOUNDATION
  SNID: ASASSN-15bc
  RA: 61.5609874
  DEC: -8.8856098
  MWEBV: 0.037 # Schlafly & Finkbeiner MW E(B-V)
    
Below the header, the photometry is included in the following
format::

  NOBS: 64
  NVAR:   7
  VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR    MAG     MAGERR
  OBS: 57422.54 g VOID  21576.285 214.793 16.665 0.011
  OBS: 57428.47 g VOID  30454.989 229.733 16.291 0.008
  OBS: 57436.55 g VOID  26053.054 253.839 16.460 0.011
  OBS: 57449.46 g VOID  11357.888 158.107 17.362 0.015
  ...
  END_PHOTOMETRY:
  
The SALT3 training code only reads the MJD, FLT (filter),
FLUXCAL, and FLUXCALERR values.  FLUXCAL and FLUXCALERR use a
zeropoint of 27.5 mag.  

The beginning of the spectroscopy section is identified by the following
header lines::
  
  NVAR_SPEC: 3
  VARNAMES_SPEC: LAMAVG  FLAM  FLAMERR

Where the columns are wavelength (angstrom), flux (erg/cm^2/s/A), and flux
uncertainty (not currently used).  Each spectrum has
the following format::
  
  SPECTRUM_ID:       1
  SPECTRUM_MJD:      54998.378  # Tobs =  -13.832
  SPECTRUM_TEXPOSE:  100000.000  # seconds
  SPECTRUM_NLAM:     352 (of 352)  # Number of valid wavelength bins
  SPEC:  4200.00  4209.35  -2.068e-10   5.701e-10
  SPEC:  4209.35  4218.76  -2.704e-10   6.359e-10    2.557e-10  23.25
  SPEC:  4218.76  4228.23  -2.725e-10   6.312e-10    2.543e-10  23.26
  SPEC:  4228.23  4237.76  -4.588e-11   6.232e-10    2.538e-10  23.25
  SPEC:  4237.76  4247.35  -8.320e-10   6.152e-10    2.541e-10  23.25
  ...
  END_SPECTRUM:

