#!/usr/bin/env python
# D. Jones - 12/28/21

import pylab as plt
plt.ion()
import numpy as np
import extinction
from scipy.optimize import minimize, least_squares
import emcee
import corner

def main():
    hphase,hwave,hflux = np.loadtxt('output/salt3_template_host.dat',unpack=True)
    hwave = np.unique(hwave)
    peakflux = hflux[hphase == 0]

    ebv = 0.09
    def extfit(x):
        RV1 = x[0]
        RV2 = x[0]+x[1]
        scale = x[2]

        a_filt_rv1 = extinction.fitzpatrick99(hwave,ebv*RV1)
        a_filt_rv2 = extinction.fitzpatrick99(hwave,ebv*RV2)
        deltaext_base = extinction.fitzpatrick99(np.array([11000]),ebv*RV2)[0] - extinction.fitzpatrick99(np.array([11000]),ebv*RV1)[0]
        deltaext = extinction.fitzpatrick99(hwave,ebv*RV2) - extinction.fitzpatrick99(hwave,ebv*RV1)
        return peakflux-(deltaext-deltaext_base)*scale

    md = least_squares(extfit,(1.5,1.5,0.2),bounds=((0.5,-np.inf,-np.inf),(5.0,np.inf,np.inf)))

    ndim, nwalkers, nsteps = 4, 32, 200

    # some parameter bounds
    rvmin,rvmax = 1,5
    delrvmin,delrvmax = 0.0,2.0
    scalemin,scalemax = 0.0,1.0
    ebvmin,ebvmax = 0.05,0.15
    
    rv_rand = np.random.randn(nwalkers)*(rvmax-rvmin) - rvmin
    delrv_rand = np.random.randn(nwalkers)*(delrvmax-delrvmin) - delrvmin
    scale_rand = np.random.randn(nwalkers)*(scalemax-scalemin) - scalemin
    ebv_rand = np.random.randn(nwalkers)*(ebvmax-ebvmin) - ebvmin

    def lnprob(x):
        RV1 = x[0]
        RV2 = x[0]+x[1]
        scale = x[2]
        ebv = x[3]
        
        a_filt_rv1 = extinction.fitzpatrick99(hwave,ebv*RV1)
        a_filt_rv2 = extinction.fitzpatrick99(hwave,ebv*RV2)
        deltaext_base = extinction.fitzpatrick99(np.array([11000]),ebv*RV2)[0] - \
            extinction.fitzpatrick99(np.array([11000]),ebv*RV1)[0]
        deltaext = extinction.fitzpatrick99(hwave,ebv*RV2) - \
            extinction.fitzpatrick99(hwave,ebv*RV1)
        if RV1 > rvmin and RV1 < rvmax and \
           x[1] > delrvmin and x[1] < delrvmax and \
           scale > scalemin and scale < scalemax and \
           ebv > ebvmin and ebv < ebvmax:
            return -np.sum((peakflux-(deltaext-deltaext_base)*scale)**2.)
        else:
            return -np.inf
    
    #p0 = (rv_rand,delrv_rand,scale_rand,ebv_rand)
    p0 = np.zeros([nwalkers,ndim])
    p0[:,0] = rv_rand
    p0[:,1] = delrv_rand
    p0[:,2] = scale_rand
    p0[:,3] = ebv_rand
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    #import pdb; pdb.set_trace()
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 20000)
    sampler.reset()

    print("Running production...")
    p, blah1, blah2 = sampler.run_mcmc(p0, 20000)
    samples = sampler.flatchain

    isamples = np.where((samples[:,0] > rvmin) & (samples[:,0] < rvmax) &
                        (samples[:,1] > delrvmin) & (samples[:,1] < delrvmax) &
                        (samples[:,2] > scalemin) & (samples[:,2] < scalemax) &
                        (samples[:,3] > ebvmin) & (samples[:,3] < ebvmax))
    samples = samples[isamples,:][0]
    corner.corner(samples,labels=('$R_V$','$\Delta R_V$','scale','$E(B-V)$'),
                  quantiles = (0.16,0.5,0.85),show_titles=True)
    import pdb; pdb.set_trace()
    
    plt.plot(hwave,peakflux,label='SALT3 $M_{host}$')
    #RV1=3.1;RV2=1.5
    RV1=md.x[0];RV2=md.x[0]+md.x[1]
    scale = md.x[2]

    deltaext = extinction.fitzpatrick99(np.array([11000]),ebv*RV2)[0] - extinction.fitzpatrick99(np.array([11000]),ebv*RV1)[0]
    plt.plot(hwave,((extinction.fitzpatrick99(hwave,ebv*RV2)-extinction.fitzpatrick99(hwave,ebv*RV1))-deltaext)*scale,label=f'F99, low-mass $R_V$ = {RV1:.1f}, high-mass $R_V$ = {RV2:.1f}')
    plt.legend()
    
    return md
    
if __name__ == "__main__":
    main()
