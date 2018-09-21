*******************************************
Spectroscopic and Photometric Training Data
*******************************************

A number of light curves and spectra are provided
in the :code:`examples/exampledata/` directory
for training.  Light curves and spectra are
combined into a single `SNANA-formatted <http://http://snana.uchicago.edu/>`_
file.

.. _data-format:

===================
Data Format
===================

`SNANA <http://http://snana.uchicago.edu/>`_ file format
consists of a number of header keys giving information
about each SN, followed by photometry and spectroscopy.
Example SNANA-formatted data with both photometry and
spectra are provided in the :code:`examples/exampledata/phot+specdata/`
directory.

An example of the minimum required header is below::

  SURVEY:  PS1MD # matches SN to the filter functions given by each kcor file
  SNID:  ASASSN-16bc # SN identifier
  REDSHIFT_HELIO:  0.05 +- 0.01 # needed so that SALT model can be redshifted to match the data

Below the header, the photometry is included in the following
format::

  NOBS: 64
  NVAR:   7
  VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR    MAG     MAGERR
  OBS: 57422.54 g NULL  21576.285 214.793 16.665 0.011
  OBS: 57428.47 g NULL  30454.989 229.733 16.291 0.008
  OBS: 57436.55 g NULL  26053.054 253.839 16.460 0.011
  OBS: 57449.46 g NULL  11357.888 158.107 17.362 0.015
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

the :code:`salt3/util/` directory will soon include utilities for
adding ASCII spectra to a pre-existing light curve file.
