:orphan:

========
Examples
========

This section will contain example scripts for SALTShaker training and validation.

Basic Usage
===========

To train a SALT3 model, use the ``trainsalt`` command with a configuration file::

    trainsalt -c myconfig.conf

For pipeline-based training and validation::

    from saltshaker.pipeline.pipeline import SALT3pipe

    pipe = SALT3pipe(finput='pipeline_input.txt')
    pipe.configure()
    pipe.run()

See :ref:`gettingstarted` for a complete tutorial.
