.. _gettingstarted:

***********************
Getting Started Quickly
***********************

To make it easy to run the latest SALTShaker training, the `trainsalt -g`
command downloads and unpacks the training data/config files into a local directory called
`saltshaker-latest-training`.  You can run the full training (takes a few hours) with the
following commands::

  conda activate saltshaker
  trainsalt -g
  cd saltshaker-latest-training
  trainsalt -c traingradient.conf

Training results will be in the `output/` directory.
