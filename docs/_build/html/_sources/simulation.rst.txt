***********************
Creating Simulated Data
***********************

BYO SED
=======

The BYOSED framework allows any spectrophotometric model to be used
as the underlying template to generate simulated Type Ia light curve data
with SNANA. By default, this model is the Hsiao+07 model (initfiles/Hsiao07.dat).
This can be replaced by any model


Param File Basics
=================

The only file to set up is the BYOSED.params file. This contains the general aspects
of the simulated SN you want to create using BYOSED, and any warping effects you
want to add in. This file is separated into the following required and optional sections:

[MAIN]
------
**(Required)**

This section contains **SED_FILE** (name of SED file), as well as **MAGSMEAR** (magnitude 
smearing) and **MAGOFF** (magnitude offsets) definitions to be applied to the base SED defined by
sed_file. You may also define **CLOBBER** and **VERBOSE** flags here as well. This section may look
like the following:

::
	
	[MAIN]

	SED_FILE: Hsiao07.dat
	MAGSMEAR: 0.0
	MAGOFF: 0.0


[FLAGS]
-------
**(Optional)**

This section allows you to simply turn warping effects defined in the next section(s) on and off. If
this section exists, then it supersedes later sections and defines the warping effects to be used. 
If it does not exist, all defined warping effects are used. Adding this onto the **[MAIN]** section,
the params file might now look like the following:

::

	[MAIN]

	SED_FILE: Hsiao07.dat
	MAGSMEAR: 0.0
	MAGOFF: 0.0

	[FLAGS]

	COLOR: True
	STRETCH: True
	HOST_MASS = False


In this case, a magnitude smearing of 0.1 would be applied to the Hsiao model at all wavelengths,
and some color and stretch effects are applied as well based on functions you will 
define in the next sections. 

Warping Effects
===============

The following sections contain all of the various wavelength/phase dependent effects that you want
to apply to your SED. In this case, based on the **[FLAGS]** section, you must have a "COLOR" section
and a "STRETCH" section. You can name effects whatever you want **with the exception of a "color law" 
effect, which must be named **"COLOR"**, as long as the name of your section and the corresponding
name in the **[FLAGS]** section are identical. Creating a warping effect section requires the following
variables in no particular order:

1. DIST_PEAK

  * The PEAK of an (a)symmetric Gaussian that will define the distribution for the scale parameter

2. DIST_SIGMA

  * The "low" and "high" standard deviations of the same distribution

3. DIST_LIMITS

  * The lower and upper cutoff you would like for the same distribution 

6. DIST_FUNCTION

  * A file name to be read that contains a list of phase, wave, value like the following:

::

	#p w v
	-20 1000 25.75805
	-20 1010 25.64852
	-20 1020 25.53899
	-20 1030 25.42946
	-20 1040 25.31993
	-20 1050 25.2104
	     ...

You must now define a section for each warping effect, with these variables. For our current example,
where I have defined color and stretch effects in my **[FLAGS]** section, I must define these two
sections. If I do not define a **[FLAGS]** section, then whatever sections that exist apart from
the **[MAIN]** section are assumed to be warping effects. One such section might look like the
following:


::

	[COLOR]

	WARP_FUNCTION: salt2_colorlaw.dat
	DIST_PEAK: 0.0
	DIST_SIGMA: 0.07 0.1
	DIST_LIMITS: -0.3 0.3

All together, after adding in the stretch section as well, a **BYOSED.params** file might look something like this:

::

	[MAIN]

	SED_FILE: Hsiao07.dat
	MAGSMEAR: 0.0
	MAGOFF: 0.0

	[FLAGS]

	COLOR: True
	STRETCH: True
	HOST_MASS = False

	[COLOR]

	WARP_FUNCTION: salt2_colorlaw.dat
	DIST_PEAK: 0.0
	DIST_SIGMA: 0.07 0.1
	DIST_LIMITS: -0.3 0.3

	[STRETCH]

	WARP_FUNCTION: salt2_m1.dat
	DIST_PEAK: 0.5
	DIST_SIGMA: 1.0 0.7
	DIST_LIMITS: -2.5 2.5

Or, if you do not define a flags section, color and stretch will automatically be used as 
warping effects with the following **BYOSED.params** file:

::

	[MAIN]

	SED_FILE: Hsiao07.dat
	MAGSMEAR: 0.0
	MAGOFF: 0.0

	[COLOR]

	WARP_FUNCTION: salt2_colorlaw.dat
	DIST_PEAK: 0.0
	DIST_SIGMA: 0.07 0.1
	DIST_LIMITS: -0.3 0.3

	[STRETCH]

	WARP_FUNCTION: salt2_m1.dat
	DIST_PEAK: 0.5
	DIST_SIGMA: 1.0 0.7
	DIST_LIMITS: -2.5 2.5

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

These are example files that can be used for your :download:`sed_file <./example_files/Hsiao07.dat>` and :download:`BYOSED.params <./example_files/BYOSED.params>`.
The color and stretch functions are defined by accompanying :download:`color <./example_files/color_func.dat>` and :download:`stretch <./example_files/stretch_func.dat>` files.






















