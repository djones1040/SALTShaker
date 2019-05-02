********************
Running the Pipeline
********************

Pipeline Discription
====================

The SALT3 Training pipeline consists of several procedures that will be run in series. The pipeline modifies a base input file to create a customized one and calls the external program with the customized input. Details are described below.


Param File
==========

General Sturcture
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

set_key=[SECTION1] [KEY] [VALUE]
    [SECTION2] [KEY2] [VALUE2]
    [SECTION2] [KEY3] [VALUE3]


Running the Pipeline
====================

::
pipe = SALT3pipe(finput='sampleinput.txt')
pipe.configure()
pipe.run()


















