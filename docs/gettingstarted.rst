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

To train with the host-mass-dependent correction to M0 and M1 (see ``masscorrection_rank``
in :ref:`configuration`), run::

  trainsalt -c traingradient_masscorrection.conf

This reads host masses from `SALT3_PARS_INIT_PP_MASS.LIST`, which copies the masses in the
light-curve headers except the placeholders 2.0 +- 2.0 (18 DES SNe), 6.0 +- 0.1 (9 Foundation
SNe) and 6.0 +- 1.0 (4 PS1 SNe), which it lists as -9, or missing. SNe without a mass follow the
sample's mass distribution. The training writes to `output_masscorrection/` and makes only the
model-summary validation plots, including `SALTmasscomp.pdf`.
