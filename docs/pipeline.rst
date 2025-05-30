.. _pipeline:

********************
Running the Pipeline
********************

Pipeline Discription
====================

The SALT3 Training pipeline consists of several procedures that will be run in series. The pipeline modifies a base input file to create a customized one and calls the external program with the customized input. Details are described below and the pipeline will be described and published in Dai et al. (in prep.).

Param File
==========

General Structure
-----------------

Each section in the param file defines one procedure in the pipeline. The gerenal structure is as follows:

::

    [Procedure Name] ([byosed], [simulation], [training], [lcfitting], [cosmology], ...)

    # external program to call
    pro =

    # arguments for the external program
    proargs = 

    # base input location
    baseinput =  

    # define the section (optional), key and value to be added or changed from the base input

    set_key= [NCOL] # 2 if no section or 3 if section exists in the config file
        [SECTION1] [KEY] [VALUE]
        [SECTION2] [KEY2] [VALUE2]
        [SECTION2] [KEY3] [VALUE3]

Batch mode
----------

The pipeline supports batch submission for certain stages (e.g. simulation, lcfitting, ...)

set `batch=True` under that stage.



Running the Pipeline
====================

The pipeline class
------------------

The pipeline can be run using the SALT3pipe class.  In
the :code:`examples/pipelinetest` directory,
you can run the pipeline with the following commands:

::

    from salt3.pipeline.pipeline import *
    pipe = SALT3pipe(finput='sampleinput.txt')
    pipe.build()
    pipe.configure()
    pipe.run()


Building the pipeline with selected stages
------------------------------------------

The :code:`build` method need to be called before :code:`configure` and :code:`run` 
The default pipeline includes all the stages. Currently they are :code:`['byosed','sim','train','lcfit','getmu','cosmofit']` or :code:`['data','train','lcfit','getmu','cosmofit']`, depending on the value of the :code:`data` option.
This can be set simply by

::

    pipe.build()
    
The option :code:`data` can be turned on/off to use data/sims, for example:

::

    pipe.build(data=False)
    
The default value is :code:`data=True`

To specify or skip certain stages, set the option :code:`mode='customize'`, and specify/skip stages using :code:`onlyrun`/:code:`skip`. Note that the only one of the options can be set.

::

    pipe.build(data=False,mode='customize',onlyrun=['lcfit','getmu','cosmofit'])

Once the :code:`build` method is called, the :code:`configure` method need to be called following it so that the input files are properly configured.


Connecting the input/output of different stages using the 'glue' method
-----------------------------------------------------------------------

The :code:`glue` method can be called so that the input and output of the gluing stages are properly connected. This will overwrite the config (input) files of the stages and should be called after :code:`configure`.

::
    
    pipe.glue(['sim','train'])
    
For some stages that are connected with multiple stages, the :code:`on` option specify what input/output files to glue on:

::
    
    pipe.glue(['train','lcfit'],on='model')

::
    
    pipe.glue(['sim','lcfit'],on='phot')


Running the pipeline
--------------------

After calling :code:`build` and :code:`glue`, call the :code:`run` method the execute the pipeline:

::
    
    pipe.run()
    
Note the :code:`build`, :code:`configure`, :code:`glue` and :code:`run` methods can be called multiple times to build a customized pipeline. Keep in mind each time :code:`configure` is called, it modifies the config (input) file of certain stages in specified in :code:`build`; and each time :code:`glue` is called, it overwrites the existing config (input) file. So these methods should be called logically given how the pipeline is run. 

The following example will run the Simulation and Training stages first with their input/output properly connected, then run the LCfitting, Getmu, and Cosmofit stages. Since to glue Training and Lcfitting (lcfitting using the trained model), the training code need to be run first so that the trained model files exist.

::
    
    def test_pipeline():
        pipe = SALT3pipe(finput='sampleinput.txt')
        pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','train'])
        pipe.configure()
        pipe.glue(['sim','train'])
        pipe.run()
        pipe.build(data=False,mode='customize',onlyrun=['lcfit','getmu','cosmofit'])
        pipe.configure()
        pipe.glue(['train','lcfit'],on='model')
        pipe.glue(['sim','lcfit'],on='phot')
        pipe.glue(['lcfit','getmu'])
        pipe.glue(['getmu','cosmofit'])
        pipe.run()

Running the Pipeline using the `runpipe.py` utility [batch submission supported]
================================================================================

Currently the `runpipe.py` utility is under `salt3/pipeline/`. We plan to pre-install it in the future. 

Using `runpipe.py`
------------------

To use the utility, first define the environmental variable `MY_SALT3_DIR`:

::

    export MY_SALT3_DIR='THE_SALT3_DIRECTORY'

Then in the terminal call:

:: 

    python $MY_SALT3_DIR/SALT3/salt3/pipeline/runpipe.py -[OPTIONS] [OPTVALUES]

To see the currently available options, use

::

    python $MY_SALT3_DIR/SALT3/salt3/pipeline/runpipe.py --help

::

    usage: runpipe.py [-h] [-c PIPEINPUT] [--mypipe MYPIPE]
                      [--batch_mode BATCH_MODE] [--batch_script BATCH_SCRIPT]
                      [--randseed RANDSEED] [--fseeds FSEEDS] [--num NUM]
                      [--norun]

    Run SALT3 Pipe.

    optional arguments:
      -h, --help            show this help message and exit
      -c PIPEINPUT          pipeline input file
      --mypipe MYPIPE       define your own pipe in yourownfilename.py
      --batch_mode BATCH_MODE
                            >0 to specify how many batch jobs to submit
      --batch_script BATCH_SCRIPT
                            base batch submission script
      --randseed RANDSEED   [internal use] specify randseed for single simulation
      --fseeds FSEEDS       provide a list of randseeds for multiple batch jobs
      --num NUM             [internal use] suffix for multiple batch jobs
      --norun               set to only check configurations without launch jobs


Define your own pipeline
------------------------

Define your own pipeline is supported by `runpipe.py`. 

Simply write your own pipeline in a `MYPIPE.py` (name can be arbitrary) file and use the `--mypipe MYPIPE` flag when calling the program. Make sure to drop the `pipe.run()` line, the pipeline will be called and run in the program. Example `MYPIPE.py` file:

::
    
    def MyPipe(finput,**kwargs):
        from pipeline import SALT3pipe
        # write your own pipeline here        
        pipe = SALT3pipe(finput)
        pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','train','lcfit'])
        pipe.configure()
        pipe.glue(['sim','train'])
        pipe.glue(['sim','lcfit'])
        return pipe
