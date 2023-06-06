.. _install:

************
Installation
************

Install from PyPI
=================

SALTShaker can be installed with PyPI, but jax might need
to be installed separately via conda first::

  conda create -n saltshaker python=3.10
  conda activate saltshaker
  conda install -c conda-forge jax
  pip install saltshaker-sn

Check out :ref:`gettingstarted` to start using SALTShaker.

Install from GitHub
=================================

Currently, the software can only be installed via GitHub::

  git clone https://github.com/djones1040/SALTShaker.git

If you wish, create an isolated conda environment for
the code::

  conda create -n saltshaker python=3.10
  conda activate saltshaker
  conda install numpy
  conda install cython

Finally, install the code with::
  
  cd SALTShaker
  pip install .
