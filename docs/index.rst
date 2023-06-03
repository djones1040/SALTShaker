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
a function of shape and color
(`Guy et al., 2007 <https://ui.adsabs.harvard.edu/abs/2007A%26A...466...11G/abstract>`_;
`Guy et al., 2010 <https://ui.adsabs.harvard.edu/abs/2010A%26A...523A...7G/abstract>`_;
`Betoule et al., 2014 <https://ui.adsabs.harvard.edu/abs/2014A%26A...568A..22B/abstract>`_).
With SALTShaker we have developed an open-source
model training framework and created the "SALT3" model.
We more than doubled the amount of photometric and
spectroscopic data used for model training and
have extended the SALT framework to 11,000 Angstroms.  In the coming years, SALT3 will make use
of data from the Vera Rubin Observatory,
and the *Nancy Grace Roman Space Telescope* and can be re-trained easily in
the coming years as more SN Ia data become available.

Please report bugs, issues and requests via the `SALTShaker GitHub page <https://github.com/djones1040/SALTShaker/issues>`_.

SALT3 Model and Training Data
==========================================

The latest version of the SALT3 model has been released in:
    
`Taylor et al., 2023, MNRAS, 520, 5209T <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.5209T/abstract>`_

This model includes full re-calibration of the SALT3 training data (`Brout et al., 2021 <https://arxiv.org/abs/2112.03864>`_) to match SALT training sets used in the `Pantheon+ analysis <https://ui.adsabs.harvard.edu/abs/2022ApJ...938..113S/abstract>`_.  Other recent SALT3 publications include:

`Pierel et al., 2021, ApJ, 911, 96P <https://ui.adsabs.harvard.edu/abs/2021ApJ...911...96P/abstract>`_: model-independent simulation framework for SALT3 validation
`Kenworthy et al., 2021, ApJ, 923, 265K <https://ui.adsabs.harvard.edu/abs/2021ApJ...923..265K/abstract>`_: SALT3 model and SALTShaker framework.  The first SALT3 version.
`Pierel et al., 2022, ApJ, 939, 11P <https://ui.adsabs.harvard.edu/abs/2022ApJ...939...11P/abstract>`_: A near-infrared extension to the SALT3 model
`Dai et al., 2023, ApJ, in press <https://ui.adsabs.harvard.edu/abs/2022arXiv221206879D/abstract>`_: SALT3 model validation with extensive simulations and presentation
`Jones et al., 2023, ApJ, in press <https://ui.adsabs.harvard.edu/abs/2022arXiv220905584J/abstract>`_: a host-galaxy mass-dependent SALT3 model

The latest SALT3 model files are linked `here <_static/salt3-f22.tar.gz>`_.
SALT3 light curve fits can be performed using `sncosmo <https://sncosmo.readthedocs.io/en/latest/>`_ 
(currently the `latest version <https://github.com/sncosmo/sncosmo>`_ on GitHub is required)
or `SNANA <https://snana.uchicago.edu/>`_ with the SALT3.K21
model, with a brief sncosmo example given below.

The latest SALT3 training data is also fully public and included `here <_static/SALT3TRAIN_K21-Frag.tgz>`_.  This release includes all photometry and spectra
along with everything required to run the code.  Once SALTShaker has been installed via the instructions in :ref:`install`, the SALT3 model can be
(re)trained with the following command using the files in this directory::

  trainsalt -c traingradient.conf

  
Example SALT3 Fit
=================

Fitting SN Ia data with SALT3 can be done through the sncosmo or
SNANA software packages.  With sncosmo, the fitting can be performed
in nearly the exact same way as SALT2.  Here is the example from the sncosmo
documentation, altered to use the SALT3 model.  First, install the latest
version of sncosmo; SALT3 is included beginning in version 2.5.0::

  conda install -c conda-forge sncosmo

or::
  
  pip install sncosmo

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

In `Dai et al., 2023 <https://ui.adsabs.harvard.edu/abs/2022arXiv221206879D/abstract>`_ we present a pipeline to fully test and validate the
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
