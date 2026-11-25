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

This reads host masses from `SALT3_PARS_INIT_PP_MASS.LIST`, which takes them from the
Pantheon+ distance table (`Pantheon+SH0ES.dat` in the Pantheon+ data release) and, for SNe not
in that table, from the light-curve headers; 1001 of its 1145 SNe have a mass. Values that mark
a host without a measurement are listed as -9, or missing: masses of 7.0 or less (the Pantheon+
floor of 7.0 and the header placeholders 2.0 and 6.0), uncertainties of 5 dex or more, and 10.0
without an uncertainty. Pantheon+ caps masses at 11.5, the highest default knot, and gives no
uncertainty for 65 of the masses, which the training then treats as exact. SNe without a mass
follow the sample's mass distribution. The training writes to `output_masscorrection/` and
makes only the model-summary validation plots, including `SALTmasscomp.pdf`.
