#!/usr/bin/env python

import numpy as np
import fileutils
import glob
import snana
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import astropy.table as at
sdss = at.Table.read('/Users/David/Dropbox/research/sngalex/sdsspos.txt',format='ascii')

_c=299792458/1000.0
_beta = 0.43
_beta_err = 0.021*5.0  # 5 sigma is industry standard
_v_helio = 371.
_l_h = 263.85 # galactic longitude
_b_h = 48.25 # galactic latitude

_v_LG = 318.
_l_LG = 105.7467
_b_LG = -5.9451
_l_0 = np.radians(264.14)
_b_0 = np.radians(48.26)

_vpecerr = 250.0  # only for printout

def sin_d(x):
    """ sine in degrees for convenience """
    return np.sin(x*np.pi/180.)

def cos_d(x):
    """ cosine in degrees, for convenience """
    return np.cos(x*np.pi/180.)

def dmdz(z):
    """ converts uncertainty in redshift to uncertainty in magnitude, for clarity"""
    return 5./(np.log(10)*z)

def correct_redshift(z_h, vpec, l, b):
    """ convert helio redshift to cosmological redshift (zbar; in CMB frame)
        input needs to be vector in galactic cartesian coordinates, w.r.t. CMB frame
        components are: heliocentric motion, peculiar velocity (in radial direction)
        """
    helio_corr = _v_helio/_c*((sin_d(b)*sin_d(_b_h)
                                       + cos_d(b)*cos_d(_b_h)*cos_d(l-_l_h)))
    #pec_corr = vpec.dot(np.array([cos_d(l)*cos_d(b),
    #                              sin_d(l)*cos_d(b), sin_d(b)]))/_c
    corr_term = 1 - helio_corr #+ pec_corr
    return (1+z_h)/corr_term - 1

def vnew(ra, dec, z):
    c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame = 'icrs')
    c_icrs = c_icrs.galactic
    b = c_icrs.b.degree
    l = c_icrs.l.degree

    b = np.radians(b)
    l = np.radians(l)
    l_0 = np.radians(264.14)
    b_0 = np.radians(48.26)


    v = (float(z)*3*10**5 + 371 * (np.sin(b) * np.sin(b_0) + np.cos(b) * np.cos(b_0) * np.cos(l-l_0)))/(3*10**5)

    return v

def main():
    import os
    files = np.concatenate((glob.glob(os.path.expandvars('$SNDATA_ROOT/lcmerge/Hamuy1996_LC/*dat')),
                            glob.glob(os.path.expandvars('$SNDATA_ROOT/lcmerge/Hicken2009_LC/*dat')),
                            glob.glob(os.path.expandvars('$SNDATA_ROOT/lcmerge/Riess1999_LC/*dat')),
                            glob.glob(os.path.expandvars('$SNDATA_ROOT/lcmerge/Jha2006_LC/*dat')),
                            glob.glob(os.path.expandvars('$SNDATA_ROOT/lcmerge/OTHER_LOWZ_LC/*dat')),
                            glob.glob(os.path.expandvars('$SNDATA_ROOT/lcmerge/SNLS3_LC/*dat')),
                            glob.glob(os.path.expandvars('$SNDATA_ROOT/lcmerge/SDSS/*dat'))))
    for f in files:
        sn = snana.SuperNova(f)
        zhel = float(sn.REDSHIFT_HELIO.split()[0])
        snpcalc = f'/Users/David/Dropbox/research/salt2/trainsalt/v6/lc-{sn.SNID}.list'
        ra,dec = None,None
        if os.path.exists(snpcalc):
            with open(snpcalc) as fin:
                for line in fin:
                    if line.startswith('@RA'): ra = float(line.split()[1])
                    if line.startswith('@DEC'): dec = float(line.split()[1])
        else:
            snpcalc = f'/Users/David/Dropbox/research/salt2/trainsalt/v6/SDSS3_%06i.DAT'%sn.SNID
            if os.path.exists(snpcalc):
                with open(snpcalc) as fin:
                    for line in fin:
                        if line.startswith('@RA'): ra = float(line.split()[1])
                        if line.startswith('@DEC'): dec = float(line.split()[1])
            else:
                print(f"no coords for {sn.SNID}")
        if ra is None or dec is None:
            try:
                ra,dec = sdss['RA'][sdss['ID'] == 'SN%i'%sn.SNID][0],sdss['RA'][sdss['ID'] == 'SN%i'%sn.SNID][0]
                sc = SkyCoord(ra,dec,unit=(u.hour,u.deg))
                ra,dec = sc.ra.deg,sc.dec.deg
            except:
                if sn.SNID == 'sn2006ob':
                    ra,dec = 27.95046, 0.26342
                elif sn.SNID == 'sn2006oa':
                    ra,dec = 320.92892,-0.84347
                elif sn.SNID == 'sn2005hc':
                    ra,dec = 29.1999000,-0.2135056
                elif sn.SNID == 'sn2005ir':
                    ra,dec = 19.1823000,0.7945444
                elif sn.SNID == 'sn2006nz':
                    ra,dec = 14.1217083,-1.2266944
                elif sn.SNID == 'sn2006on':
                    ra,dec = 328.9937500,-1.0702778
                else:
                    import pdb; pdb.set_trace()

        sc = SkyCoord(ra,dec,unit=(u.deg, u.deg))
        gsc = sc.galactic
        zcmb = vnew(ra,dec,zhel) #,0,gsc.l.degree,gsc.b.degree)
        fileutils.replace_line(f,f'RA: {ra}',linestart='RA')
        fileutils.replace_line(f,f'DECL: {dec}',linestart='DECL')
        fileutils.replace_line(f,f'REDSHIFT_FINAL: {zcmb:.5f} +- 0.0001',linestart='REDSHIFT_FINAL')
    
if __name__ == "__main__":
    main()
