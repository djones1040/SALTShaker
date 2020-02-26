#!/usr/bin/env python
#import pysynphot
import numpy as np
from sncosmo.constants import HC_ERG_AA
import logging
log=logging.getLogger(__name__)
def synphot(wave,flux,zpoff=0,filtfile=None,primarywave=[],primaryflux=[],
			filtwave=[],filttp=[],
			plot=False,oplot=False,allowneg=False):

	if filtfile:
		mag = zpoff - 2.5 * np.log10( synflux(wave,flux,pb=filtfile,plot=plot,oplot=oplot,
									   allowneg=allowneg))
		#if len(primarywave) and len(primaryflux):
		#	stdmag = - 2.5 * np.log10( synflux(primarywave,primaryflux,pb=filtfile,
		#									   plot=plot,oplot=oplot,
		#									   allowneg=allowneg))
	elif len(filtwave) and len(filttp):
		mag = zpoff - 2.5 * np.log10( synflux(wave,flux,pbx=filtwave,pby=filttp,plot=plot,oplot=oplot,
											  allowneg=allowneg))
		#if len(primarywave) and len(primaryflux):
		#	stdmag = -2.5 * np.log10( synflux(primarywave,primaryflux,pbx=filtwave,pby=filttp,
		#									  plot=plot,oplot=oplot,
		#									  allowneg=allowneg))
	else:
		raise RuntimeError("filter file or throughput must be defined")

	return(mag)

def synflux(x,spc,pb=None,plot=False,oplot=False,allowneg=False,pbx=[],pby=[]):
	import numpy as np

	nx = len(x)
	pbphot = 1
	if pb:
		pbx,pby = np.loadtxt(pb,unpack=True)
	elif not len(pbx) or not len(pby):
		raise RuntimeError("filter file or throughput must be defined")
		
	npbx = len(pbx)
	if (len(pby) != npbx):
		log.warning(' pbs.wavelength and pbs.response have different sizes')

	if nx == 1 or npbx == 1:
		log.warning('warning! 1-element array passed, returning 0')
		return(spc[0]-spc[0])

	diffx = x[1:nx]-x[0:nx-1]
	diffp = pbx[1:npbx]-pbx[0:npbx-1]

	if (np.min(diffx) <= 0) or (np.min(diffp) <= 0):
		log.warning('passed non-increasing wavelength array')

	#if x[0] > pbx[0]:
	#	print("spectrum doesn''t go blue enough for passband!")

	#if x[nx-1] < pbx[npbx-1]:
	#	print("spectrum doesn''t go red enough for passband!")

	g = np.where((x >= pbx[0]) & (x <= pbx[npbx-1]))  # overlap range

	pbspl = np.interp(x[g],pbx,pby)#,kind='cubic')

	if not allowneg:
		pbspl = pbspl
		col = np.where(pbspl < 0)[0]
		if len(col): pbspl[col] = 0

	if (pbphot): pbspl *= x[g]


	res = np.trapz(pbspl*spc[g]/HC_ERG_AA,x[g])/np.trapz(pbspl,x[g])

	return(res)
