************************
Training the SALT3 Model
************************

usage::

  TrainSALT.py -c <configfile> <options>
  
To run an example, change to the "examples" directory
in the main package and run::

  TrainSALT.py -c SALT.conf

This will use the lightcurves and spectra in the "exampledata"
directory and the kcor files in the "kcor" directory to
run the SALT3 training, writing outputs to the "output"
directory

************************************
SALT3 Training Configuration Options
************************************

See the SALT.conf file for a list of training options.
Descriptions of each option are given below.
