#!/usr/bin/env python
# D. Jones - 8/28/20

import numpy as np
import sncosmo

def main(salt2dir_noerr='/usr/local/SNDATA_ROOT/models/SALT2/SALT2.JLA-B14-NOERR',salt2dir_cerr='/usr/local/SNDATA_ROOT/models/SALT2/SALT2.JLA-B14-cERR'):

    salt2ne = sncosmo.SALT2Source(modeldir=salt2dir_noerr)
    modelne=sncosmo.Model(source=salt2ne)
    salt2ce = sncosmo.SALT2Source(modeldir=salt2dir_cerr)
    modelce = sncosmo.Model(source=salt2ce)
    
    band = 'sdssz'
    plotmjd = np.array([55000,55001])
    lameffrest = 8944.1240
    for z in np.arange(0.25,1.1,0.05):
        modelne.set(z=z,t0=55000)
        modelce.set(z=z,t0=55000)
        lameff = lameffrest/(1+z)
        
        snrne = modelne.bandfluxcov(band, plotmjd, zp=23.9,zpsys='ab')[0][0]/np.sqrt(modelne.bandfluxcov(band, plotmjd, zp=23.9,zpsys='ab')[1][0,0])
        snrce = modelce.bandfluxcov(band, plotmjd, zp=23.9,zpsys='ab')[0][0]/np.sqrt(modelce.bandfluxcov(band, plotmjd, zp=23.9,zpsys='ab')[1][0,0])
        print(f"{lameff:.1f} cdisp SNR: {snrce:.1f}")
        
if __name__ == "__main__":
    main()
