***********************
Creating Simulated Data
***********************

BYO SED
=======

The BYOSED framework allows any spectrophotometric model to be used
as the underlying template to generate simulated Type Ia light curve data
with SNANA. By default, this model is the Hsiao+07 model (initfiles/Hsiao07.dat).
This can be replaced by any model


Param File
==========

The first file to setup is the BYOSED.params file. This contains the general aspects
of the simulated SN you want to create using BYOSED. This file is separated into 2
sections:

[main]
------

This section contains **sed_file** (name of SED file), as well as **magsmear** (magnitude 
smearing) and **magoff** (magnitude offsets) definitions to be applied to the base SED defined by
sed_file.

[effects]
---------

This section contains all of the various wavelength/phase dependent effects that you want
to apply to your SED. You can turn effects on and off with True/false flags, and you can
name them whatever you want **with the exception of a "color law" effect, which must be
named "color"**. An example of the BYOSED.params file is as follows:

::

	[main]
	sed_file = Hsiao07.dat
	magsmear = 0.1
	magoff = 0.0

	[effects]
	color = True
	stretch = True
	host_mass = False

In this case, a magnitude smearing of 0.1 would be applied to the Hsiao model at all wavelengths,
and some color and stretch effects are applied as well based on functions you will 
define in the next section. 

Warping Effects
===============

You must set up a second file that defines the warping functions you've set up as effects in the
BYOSED.params file above. This file is called **sed_warps.dat** and similarly has two sections:

[functions]
-----------

The names used for the various warping functions must match those used in the BYOSED.params file
above. Then, each function can either be defined as an array of phase,wave,value:

::

	#p w v
	dust = [[-100, 1000, 1],
	    	[-100,10000,1],
		[-100,15000,1],
		[-100,25000,1],
		[0,1000,1],
		[0,10000,1],
		[0,15000,1],
		[0,25000,1],
		[100,1000,1],
		[100,10000,1],
		[100,15000,1],
		[100,25000,1],
		[500,1000,1],
		[500,10000,1],
		[500,15000,1],
		[500,25000,1]]

Or else as a file name to be read, which contains a similar definition:

::

	#p w v
	-20 1000 25.75805
	-20 1010 25.64852
	-20 1020 25.53899
	-20 1030 25.42946
	-20 1040 25.31993
	-20 1050 25.2104
	     ...


[distributions]
---------------

The second section in the sed_warps file defines the distributions that will be
used to randomly generate a scale parameter for each effect, for each SN being
simulated. They are defined as (a)symmetric gaussian functions as follows:

::

	color = {"mu":0.0,"sigma_left":0.07, "sigma_right":0.1,"lower_lim":-0.3, "upper_lim":0.3}

All together, an **sed_warps.dat** file might look something like this:

::

	[functions]

	stretch = 'stretch_function.dat'
	color = 'color_function.dat'

	[distributions]

	stretch = {"mu":0.5, "sigma_left":1.0, "sigma_right":0.7,"lower_lim":-2.5, "upper_lim":2.5}
	color = {"mu":0.0,"sigma_left":0.07, "sigma_right":0.1,"lower_lim":-0.3, "upper_lim":0.3}


Final Notes
===========

Now you can replace the Hsiao template with your own template SED, and start adding in warping
effects. This warping process is designed so that as many effects as you would like can be
included. Anything but a color effect (which should affect the final SED as a function of
wavelength and possibly phase) is applied additively, while the color effect is applied
multiplicatively. This is similar to the existing SALT2 framework. For the example file 
above, the final flux would look like this 

.. math::

	F(\lambda,\phi)=A\Big[H(\lambda,\phi)+S(\lambda,\phi)s\Big]\times10^{-0.4C(\lambda,\phi)c}

Where here F is the final flux, H is the Hsiao template, S is the defined stretch function,
C is the defined color function, s is the scale parameter pulled from the distribution defined
for the stretch function, and c is the scale parameter pulled from the distribution defined 
for the color function. In principle this could look like the following if you had N such effects:

.. math::

	F(\lambda,\phi)=A\Big[H(\lambda,\phi)+X_1(\lambda,\phi)x_1+X_2(\lambda,\phi)x_2+...+X_N(\lambda,\phi)x_N\Big]\times10^{-0.4C(\lambda,\phi)c}




Example Files
=============

Thse are example files that can be used for your :download:`sed_file <./example_files/Hsiao07.dat>`, :download:`BYOSED.params <./example_files/BYOSED.params>` and :download:`sed_warps.dat <./example_files/sed_warps.dat>`.
The color and stretch functions are defined by accompanying :download:`color <./example_files/color_func.dat>` and :download:`stretch <./example_files/stretch_func.dat>` files.






















