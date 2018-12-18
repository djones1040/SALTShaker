import numpy as np
import pylab as plt
import sncosmo
import argparse
from salt3.util import snana
from astropy.table import Table
from os import path
parser = argparse.ArgumentParser(description='Plot lightcurves from SALT3 model against data, Hsiao model, and SALT2 model.')
parser.add_argument('lcfile',type=str,help='Supernova lightcurve data in SNANA format')
parser.add_argument('SALT3',type=str,help='Directory with SALT3 parameters')
parser=parser.parse_args()

sn = snana.SuperNova(parser.lcfile)
sn.FLT = sn.FLT.astype('U20')
for i in range(len(sn.FLT)):
    sn.FLT[i] = 'sdss%s'%sn.FLT[i]
zpsys='AB'
data = Table([sn.MJD,sn.FLT,sn.FLUXCAL,sn.FLUXCALERR,
              np.array([27.5]*len(sn.MJD)),np.array([zpsys]*len(sn.MJD))],
             names=['mjd','band','flux','fluxerr','zp','zpsys'],
             meta={'t0':sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]})
flux = sn.FLUXCAL
salt2model = sncosmo.Model(source='salt2')
hsiaomodel = sncosmo.Model(source='hsiao')
salt3 = sncosmo.SALT2Source(modeldir=parser.SALT3,m0file='salt3_template_0.dat',m1file='salt3_template_1.dat')
salt3model =  sncosmo.Model(salt3)
fitparams_salt2=['t0', 'x0', 'x1', 'c']
salt2model.set(z=sn.REDSHIFT_FINAL[0:5])
result_salt2, fitted_salt2_model = sncosmo.fit_lc(data, salt2model, fitparams_salt2)
fitparams_hsiao = ['t0','amplitude']
hsiaomodel.set(z=sn.REDSHIFT_FINAL[0:5])
result_hsiao, fitted_hsiao_model = sncosmo.fit_lc(data, hsiaomodel, fitparams_hsiao)
fitparams_salt3=['t0', 'x0', 'x1', 'c']
hsiaomodel.set(z=sn.REDSHIFT_FINAL[0:5])
result_salt3, fitted_salt3_model = sncosmo.fit_lc(data, salt3model, fitparams_salt3)
plotmjd = np.linspace(sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]-20,
                      sn.MJD[sn.FLUXCAL == np.max(sn.FLUXCAL)]+40,100)
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
for flt,i,ax in zip(['sdssg','sdssr','sdssi'],range(3),[ax1,ax2,ax3]):
    hsiaoflux = fitted_hsiao_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
    salt2flux = fitted_salt2_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
    salt3flux = fitted_salt3_model.bandflux(flt, plotmjd,zp=27.5,zpsys='AB')
    ax.plot(plotmjd,hsiaoflux,color='C0',
            label='Hsiao, $\chi^2_{red} = %.1f$'%(result_hsiao['chisq']/result_hsiao['ndof']))
    ax.plot(plotmjd,salt2flux,color='C1',
            label='SALT2, $\chi^2_{red} = %.1f$'%(result_salt2['chisq']/result_salt2['ndof']))
    ax.plot(plotmjd,salt3flux,color='C2',
            label='SALT3, $\chi^2_{red} = %.1f$'%(result_salt3['chisq']/result_salt3['ndof']))
    ax.errorbar(sn.MJD[sn.FLT == flt],sn.FLUXCAL[sn.FLT == flt],
                yerr=sn.FLUXCALERR[sn.FLT == flt],
                fmt='o',label=sn.SNID,color='k')
    ax.set_title(flt)
ax1.legend()
plt.show()