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
directory


SALT3 Training Configuration Options
====================================

See the SALT.conf file in the :code:`examples/` directory
for a list of training options.
Descriptions of each option are given below.

==================   =============   =========================================================================================================
Name                 Default         Description                                                                             
==================   =============   =========================================================================================================  
[iodata]
snlist                               ASCII file with list of SN files (see :ref:`data-format` for input file format)
outputdir                            Directory for trained model outputs
initmodelfile        Hsiao07.dat     Initial SN SED model.  Initial parameter guesses are derived from this file.  Default is the Hsiao model
kcor_path                            Comma-separated surveyname,kcorfile for each :code:`SURVEY` key in SN data files
kcor_path+=                          Same as :code:`kcor_path`, but appends a second entry to the kcor_path variable

[trainparams]
waverange            2000,9200       Wavelength range over which the model is defined                                        
colorwaverange       2800,7000       Wavelength range over which the color law polynomial is fit                             
interpfunc           bspline         Function for interpolating the model between control points (b-spline is default)   
interporder          3               Order of the spline interpolation
wavesplineres        72              Number of Angstroms between wavelength control points
waveoutres           2               Wavelength resolution of the trained output model (Angstroms)
phaserange           -14,50          Phase range over which the model is defined (days)
phasesplineres       3.2             Wavelength resolution of the trained output model (days)
phaseoutres          2               Phase resolution of the trained output model (days)
minmethod            trf             Numerical algorithm used to minimize the chi2.  Default is Trust Region Reflective.
n_colorpars          4               Number of parameters used to define the color law polynomial
n_components         2               Number of principal components
==================   =============   =========================================================================================================
