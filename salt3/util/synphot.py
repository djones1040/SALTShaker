#!/usr/bin/env python
import pysynphot

def synphot(wave,flux,filtfile,zpt,plot=False,oplot=False,allowneg=False):
	import numpy as np
	x1=filtfile.split('/')
	nx=len(x1)
	xfin=x1[nx-1]
	fp='/'
	for i in range(nx-1): fp=fp+x1[i]+'/'


	#import pysynphot.spectrum as S
	#sp = S.Vega
	mag = zpt - 2.5 * np.log10( synflux(wave,flux,filtfile,plot=plot,oplot=oplot,
										 allowneg=allowneg))
	#vegamag = zpt - 2.5 * np.log10( synflux(wave,sp(wave),filtfile,
	#									   plot=plot,oplot=oplot,
	#									   allowneg=allowneg))

	return(mag)

def synflux(x,spc,pb,plot=False,oplot=False,allowneg=False):
	import numpy as np

	nx = len(x)
	pbphot = 1
	pbx,pby = np.loadtxt(pb,unpack=True)

	npbx = len(pbx)
	if (len(pby) != npbx):
		print(' pbs.wavelength and pbs.response have different sizes')

	if nx == 1 or npbx == 1:
		print('warning! 1-element array passed, returning 0')
		return(spc[0]-spc[0])

	diffx = x[1:nx]-x[0:nx-1]
	diffp = pbx[1:npbx]-pbx[0:npbx-1]

	if (np.min(diffx) <= 0) or (np.min(diffp) <= 0):
		print('passed non-increasing wavelength array')

	if x[0] > pbx[0]:
		print("spectrum doesn''t go blue enough for passband!")

	if x[nx-1] < pbx[npbx-1]:
		print("spectrum doesn''t go red enough for passband!")

	g = np.where((x >= pbx[0]) & (x <= pbx[npbx-1]))  # overlap range

	pbspl = np.interp(x[g],pbx,pby)#,kind='cubic')

	if not allowneg:
		pbspl = pbspl
		col = np.where(pbspl < 0)[0]
		if len(col): pbspl[col] = 0

	if (pbphot): pbspl *= x[g]


	res = np.trapz(pbspl*spc[g],x[g])/np.trapz(pbspl,x[g])

	return(res)
