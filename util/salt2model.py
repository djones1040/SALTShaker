#!/usr/bin/env python
from make_2d import make_2d
import numpy as np

def initmodel(waverange=[2000,9200],colorrange=[2800,9200],
			  phaserange=[-14,50],
			  waveres=72,phaseres=3.2):

	phasedelta=phaserange[1]-phaserange[0]
	phasearr = np.linspace(phaserange[0],phaserange[1],int(phasedelta/phaseres))

	wavedelta = waverange[1]-waverange[0]
	wavearr = np.linspace(waverange[0],waverange[1],int(wavedelta/waveres))
	
	salt2init = make_2d(phasearr,wavearr)
	salt2flux = np.ones(salt2init.shape)

def splinemodel(waverange=[2000,9200],colorrange=[2800,9200],
				phaserange=[-14,50],
				waveres=72,phaseres=3.2):
	pass
