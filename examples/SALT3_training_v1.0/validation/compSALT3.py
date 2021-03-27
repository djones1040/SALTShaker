import numpy as np
from txtobj import txtobj
import pylab as plt
import cosmo
from scipy.stats import binned_statistic
plt.ion()

_alpha2 = 0.15859
_beta2 = 2.92668
_alpha3 = 0.15716
_beta3 = 2.74441

def main():
    fr2 = txtobj('FITOPT000_MUOPT000.FITRES',fitresheader=True)
    fr3  =txtobj('SALT3.FITRES',fitresheader=True)

    zall,muall2,muall3 = np.array([]),np.array([]),np.array([])
    mubias2,mubias3 = np.array([]),np.array([])
    muresraw2,muresraw3 = np.array([]),np.array([])
    survey = np.array([])
    
    for j,i in enumerate(fr2.CID):
        if fr2.zHEL[j] < 0.01: continue
        if i in fr3.CID:
            survey = np.append(survey,fr2.IDSURVEY[j])
            zall = np.append(zall,fr2.zHEL[j])
            muall2 = np.append(muall2,fr2.MU[j]) #fr2.MURES[j]+fr2.M0DIF[j]-fr2.MUMODEL[j])
            muall3 = np.append(muall3,fr3.MU[fr3.CID == i][0]) #fr3.MURES[fr3.CID == i][0]+fr3.M0DIF[fr3.CID == i][0]-fr3.MUMODEL[fr3.CID == i][0])
            mubias2 = np.append(mubias2,fr2.biasCor_mu[j])
            mubias3 = np.append(mubias3,fr3.biasCor_mu[fr3.CID == i][0])
            muresraw2 = np.append(muresraw2,fr2.mB[j] + _alpha2*fr2.x1[j] - _beta2*fr2.c[j] + 19.36 - cosmo.mu(fr2.zCMB[j]))
            muresraw3 = np.append(muresraw3,fr3.mB[fr3.CID == i][0] + _alpha3*fr3.x1[fr3.CID == i][0] - _beta3*fr3.c[fr3.CID == i][0] + 19.36 - cosmo.mu(fr3.zCMB[fr3.CID == i][0]))

    plt.plot(zall,muall3-muall2,'.',color='0.8')
    #plt.plot(zall,muall3+mubias3-(muall2+mubias2),'.',color='C1')
    
    zbins = np.linspace(0,1.5,20)
    dmubins = binned_statistic(zall,muall3-muall2,bins=zbins,statistic='median').statistic
    dmubins_raw = binned_statistic(zall,muresraw3-muresraw2-np.median(muresraw3-muresraw2),bins=zbins,statistic='median').statistic
    plt.plot((zbins[1:]+zbins[:-1])/2.,dmubins_raw,'o-',color='b',label='raw')
    plt.plot((zbins[1:]+zbins[:-1])/2.,dmubins,'o-',color='r',label='bias-corrected')
    plt.axhline(0,color='k',lw=2)
    plt.legend()
    plt.xlabel('$z$',fontsize=15)
    plt.ylabel('$\mu_{SALT3}-\mu_{SALT2}$',fontsize=15)

    delmu = muall3-muall2
    delmuraw = muresraw3-muresraw2-np.median(muresraw3-muresraw2)
    
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()
