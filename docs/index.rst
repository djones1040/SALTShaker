.. raw:: html
    <style media="screen" type="text/css">
      h1 { display:none; }
      th { display:none; }
    </style>

********************
SALTShaker and SALT3
********************

Overview
==========================================

SALT is a model of Type Ia supernovae (SNe Ia)
that accounts for spectral variations as
a function of shape and color (Guy et al., 
2007; Guy et al., 2010; Betoule et al., 2014).
With SALTShaker we have developed an open-source
model training framework and created the "SALT3" model.
We more than doubled the amount of photometric and spectroscopic data used for model training and
have extended the SALT framework to 11,000 Angstroms.  SALT3 will make use
of *iz* data from PS1, the Vera Rubin Observatory,
and the *Nancy Grace Roman Space Telescope* and can be re-trained easily in
the coming years as more SN Ia data become available.

Please report bugs, issues and requests via the SALTShaker GitHub page.

SALT3 Model and Training Data
==========================================

The first version of the SALT3 model has been released in:
    
Kenworthy et al., 2021, ApJ, submitted

The SALT3 model files are linked `here <_static/salt3-k21.tar.gz>`_.
SALT3 light curve fits can be performed using `sncosmo <https://sncosmo.readthedocs.io/en/latest/>`_ 
(currently the `latest version <https://github.com/sncosmo/sncosmo>`_ on GitHub is required)
or `SNANA <https://snana.uchicago.edu/>`_ with the SALT3.K21
model, with a brief sncosmo example given below.

The SALT3 training data is also fully public and included `here <_static/SALT3TRAIN_K21_PUBLIC.tgz>`_.  This release includes all photometry and spectra
along with everything required to run the code.  Once SALTShaker has been installed via the instructions in :ref:`install`, the SALT3 model can be
(re)trained with the following command::

  trainsalt -c Train_SALT3_public.conf


  
Example SALT3 Fit
=================

Fitting SN Ia data with SALT3 can be done through the sncosmo or
SNANA software packages.  With sncosmo, the fitting can be performed
in nearly the exact same way as SALT2.  Here is the example from the sncosmo
documentation, altered to use the SALT3 model.  First, install the latest
development version of sncosmo from source (SALT3 will be included in an official
release soon)::

  git clone git://github.com/sncosmo/sncosmo.git
  cd sncosmo
  ./setup.py install

Then, in a python terminal::

  import sncosmo
  data = sncosmo.load_example_data()
  model = sncosmo.Model(source='salt3')
  res, fitted_model = sncosmo.fit_lc(data, model,
                                    ['z', 't0', 'x0', 'x1', 'c'],
                                    bounds={'z':(0.3, 0.7)})
  sncosmo.plot_lc(data, model=fitted_model, errors=res.errors)


Pipeline
========

We are developing a pipeline to fully test and validate the
SALT3 model in the context of cosmological measurements.  Defails
are given in :ref:`pipeline`.

    .. image:: _static/schematic.png

    
.. toctree::
   :maxdepth: 1
   :titlesonly:

   install
   data
   training
   simulation
   pipeline
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
