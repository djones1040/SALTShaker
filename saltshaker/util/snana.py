from __future__ import print_function

import os
import time
from astropy.io import fits
import numpy as np
import glob
from collections.abc import Sequence
from matplotlib import pyplot as p
from matplotlib import patches
from astropy.time import Time
import gzip

SNTYPEDICT = {1:'Ia',10:'Ia',2:'II',3:'Ibc',32:'Ib',33:'Ic',20:'IIP',21:'IIn',22:'IIL',23:'IIb',42:'Ia',101:'Ia',0:'Ia' }

BANDCOLOR = {'g':'b','r':'g','i':'r','z':'darkorange'}
BANDMARKER = {'g':'o','r':'o','i':'o','z':'o'}

# SNANA NON1A template id conversion table
IBCMODELS = {
    '103': ['Ib' ,'CSP-2004gv' ],
    '104': ['Ib' ,'CSP-2006ep' ],
    '105': ['Ib' ,'CSP-2007Y'  ],
    '202': ['Ib' ,'SDSS-000020'],
    '203': ['Ib' ,'SDSS-002744'],
    '212': ['Ib' ,'SDSS-014492'],
    '234': ['Ib' ,'SDSS-019323'],
    '021': ['Ibc','SNLS-04D1la'],
    '022': ['Ic' ,'SNLS-04D4jv'],
    '101': ['Ic' ,'CSP-2004fe' ],
    '102': ['Ic' ,'CSP-2004gq' ],
    '205': ['Ic' ,'SDSS-004012'],
    '207': ['Ic' ,'SDSS-013195'],
    '211': ['Ic' ,'SDSS-014475'],
    '217': ['Ic' ,'SDSS-015475'],
    '218': ['Ic' ,'SDSS-017548'],
    }
       
IIMODELS = {
    '201':['IIP','SDSS-000018'],
    '204':['IIP','SDSS-003818'],
    '208':['IIP','SDSS-013376'],
    '210':['IIP','SDSS-014450'],
    '213':['IIP','SDSS-014599'],
    '214':['IIP','SDSS-015031'],
    '215':['IIP','SDSS-015320'],
    '216':['IIP','SDSS-015339'],
    '219':['IIP','SDSS-017564'],
    '220':['IIP','SDSS-017862'],
    '221':['IIP','SDSS-018109'],
    '222':['IIP','SDSS-018297'],
    '223':['IIP','SDSS-018408'],
    '224':['IIP','SDSS-018441'],
    '225':['IIP','SDSS-018457'],
    '226':['IIP','SDSS-018590'],
    '227':['IIP','SDSS-018596'],
    '228':['IIP','SDSS-018700'],
    '229':['IIP','SDSS-018713'],
    '230':['IIP','SDSS-018734'],
    '231':['IIP','SDSS-018793'],
    '232':['IIP','SDSS-018834'],
    '233':['IIP','SDSS-018892'],
    '235':['IIP','SDSS-020038'],
    '206':['IIN','SDSS-012842'],
    '209':['IIN','SDSS-013449'],
    '002':['IIL','Nugent+ScolnicIIL'],
    }

class SuperNovaSpectrum( object ) :
    """ object class for a single SN spectrum from spectrum-sname-#.list file
    """
    def __init__( self, datfile=None, verbose=False) : 
        if not (datfile) : 
            if verbose:  print("No spec file provided. Returning an empty SuperNovaSpectrum object.")
        if verbose: print("Reading in data from spec file %s"%(datfile))
        self.readspecfile( datfile )

    @property
    def z(self):
        if 'Redshift' in self.__dict__ :
            return( self.Redshift )
        else : 
            return( '' )

    @property
    def mjdspec(self):
        if 'Date' in self.__dict__ :
            return( self.Date )
        else : 
            return( '' )

    def readspecfile(self, datfile ):
        """ read spectrum from a JLA file.
        Metadata in the header are in "@ value"
        column names are given with "# value", will add manually 
        Observation data lines are the rest 
        """
        from numpy import array,log10,unique,where
        if not os.path.isfile(datfile): raise RuntimeError( "%s does not exist."%datfile) 
        self.datfile = os.path.abspath(datfile)
        fin = open(datfile,'r')
        data = fin.readlines()
        fin.close()
        colnames=['WAVE','FLUX','FLUXERR','VALID']
        for col in colnames:
            self.__dict__[ col ] = []
        for i in range(len(data)):
            line = data[i]
            if(len(line.strip())==0) : continue
            if line.startswith("@") :
                metadata_name = line.split()[0][1:]
                metadata_value = line.split()[1]
                self.__dict__[ metadata_name ] = str2num(metadata_value)
            elif line.startswith("#") : continue
            else :
                obsdat = line.split()
                self.__dict__['WAVE'].append( str2num(obsdat[0]) )
                self.__dict__['FLUX'].append( str2num(obsdat[1]) )
                self.__dict__['FLUXERR'].append( str2num(obsdat[2]) )
                if len(obsdat) >= 4: self.__dict__['VALID'].append( str2num(obsdat[3]) )
                else: self.__dict__['VALID'].append( 1.0 )
        for col in colnames : 
            self.__dict__[col] = array( self.__dict__[col] )
        return( None )

class SuperNova( object ) : 
    """ object class for a single SN extracted from SNANA sim tables
        or from a SNANA-style .DAT file
    """
    
    def __init__(
            self,
            datfile=None,
            headfitsfile=None,
            photfitsfile=None,
            specfitsfile=None,
            headfits=None,
            photfits=None,
            specfits=None,
            snid=None,
            verbose=False,
            simdir=None,
            readspec=True
    ) : 
        """ Read in header info (z,type,etc) and full light curve data.
        For simulated SNe stored in fits tables, user must provide the simname and snid,
        and the data are collected from binary fits tables, assumed to exist 
        within the $SNDATA_ROOT/SIM/ directory tree. 
        For observed or simulated SNe stored in ascii .dat files, user must provide 
        the full path to the datfile.
        """
        if not (datfile or (snid and headfitsfile and photfitsfile)) : 
            if verbose:  print("No datfile or simname provided. Returning an empty SuperNova object.")
        
        if headfitsfile and photfitsfile and snid :
            self.snid = snid
            # read in header and light curve data from binary fits tables 
            gothd = self.getheadfits( headfitsfile )
            gotphot = self.getphotfits( photfitsfile )
            if specfitsfile is not None and readspec: gotspec = self.getspecfits( specfitsfile )
            else: self.SPECTRA = {}
        elif headfits and photfits and snid:
            self.snid = snid
            # read in header and light curve data from binary fits tables 
            gothd = self.getheadfits( headfits=headfits )
            gotphot = self.getphotfits( photfits=photfits )
            if specfits is not None and readspec: gotspec = self.getspecfits( specfits=specfits )
            else: self.SPECTRA = {}
        elif datfile :  
            if verbose : print("Reading in data from light curve file %s"%(datfile))
            self.readdatfile( datfile, readspec=readspec ) 

    @property
    def name(self):
        if 'NAME' in self.__dict__ :
            return( self.NAME )
        elif 'SNID' in self.__dict__ :
            return( self.SNID )
        elif 'NICKNAME' in self.__dict__ :
            return( self.NICKNAME )
        elif 'CID' in self.__dict__ :
            return( self.CID )
        elif 'IAUNAME' in self.__dict__ :
            return( self.IAUNAME )
        else : 
            return( '' )

    @property
    def nickname(self):
        if 'NICKNAME' in self.__dict__ :
            return( self.NICKNAME )
        elif 'NAME' in self.__dict__ :
            return( self.NAME )
        elif 'IAUNAME' in self.__dict__ :
            return( self.IAUNAME )
        elif 'SNID' in self.__dict__ :
            return( self.SNID )
        elif 'CID' in self.__dict__ :
            return( self.CID )
        else : 
            return( '' )
            
    @property
    def bandlist(self):
        if 'FLT' in self.__dict__ :
            return( np.unique( self.FLT ) )
        else : 
            return( np.array([]))

    @property
    def bands(self):
        return( ''.join(self.bandlist) )

    @property 
    def BANDORDER(self):
        return( 'griz' )

    @property 
    def bandorder(self):
        return( self.SURVEYDATA.BANDORDER )

    @property
    def signoise(self):
        """ compute the signal to noise curve"""
        if( 'FLUXCALERR' in self.__dict__ and 
            'FLUXCAL' in self.__dict__  ) :
            return( self.FLUXCAL / np.abs(self.FLUXCALERR) )
        else: 
            return( None)
    
    @property
    def pkmjd(self):
        if 'PEAKMJD' in self.__dict__.keys() :
            return( self.PEAKMJD )
        elif 'SIM_PEAKMJD' in self.__dict__.keys() :
            return( self.SIM_PEAKMJD )
        else : 
            return( self.pkmjdobs )

    @property
    def pkmjderr(self):
        if 'PEAKMJDERR' in self.__dict__.keys() :
            return( self.PEAKMJDERR )
        elif 'SIM_PEAKMJDERR' in self.__dict__.keys() :
            return( self.SIM_PEAKMJDERR )
        elif 'SIM_PEAKMJD_ERR' in self.__dict__.keys() :
            return( self.SIM_PEAKMJD_ERR )
        else : 
            return( max( self.pkmjdobserr, 1.2*abs(self.pkmjdobs-self.pkmjd)  ) )

    @property
    def pkmjdobs(self):
        if 'SEARCH_PEAKMJD' in self.__dict__.keys() :
            return( self.SEARCH_PEAKMJD )
        elif 'SIM_PEAKMJD' in self.__dict__.keys() :
            return( self.SIM_PEAKMJD )
        else : 
            # crude guess at the peak mjd as the date of highest S/N
            return( self.MJD[ self.signoise.argmax() ] )

    @property
    def pkmjdobserr(self):
        if 'SEARCH_PEAKMJDERR' in self.__dict__.keys() :
            return( self.SEARCH_PEAKMJDERR )
        elif 'SEARCH_PEAKMJD_ERR' in self.__dict__.keys() :
            return( self.SEARCH_PEAKMJD_ERR )
        else : 
            # determine the peak mjd uncertainty
            ipk = self.signoise.argmax()
            pkband = self.FLT[ ipk ]
            ipkband = np.where(self.FLT==pkband)[0]
            mjdpkband = np.array( sorted( self.MJD[ ipkband ] ) )
            if len(ipkband)<2 : return( 30 )
            ipkidx = ipkband.tolist().index( ipk )
            if ipkidx == 0 : 
                return( 0.7*(mjdpkband[1]-mjdpkband[0]) )
            elif ipkidx == len(ipkband)-1 : 
                return( 0.7*(mjdpkband[-1]-mjdpkband[-2]) )
            else : 
                return( 0.7*0.5*(mjdpkband[ipkidx+1]-mjdpkband[ipkidx-1]) )

    @property
    def mjdpk(self):
        return( self.pkmjd )

    @property
    def mjdpkerr(self):
        return( self.pkmjderr )

    @property
    def mjdpkobs(self):
        return( self.pkmjdobs )

    @property
    def mjdpkobserr(self):
        return( self.pkmjdobserr )

    @property
    def isdecliner(self):
        if 'DECLINER' in self.__dict__ : 
            if self.DECLINER in ['True','TRUE',1] : return( True )
            else : return( False )
        if self.pkmjd < self.MJD.min() : return( True ) 
        else : return( False )

    @property
    def zphot(self):
        zphot = None
        for key in ['HOST_GALAXY_PHOTO-Z','ZPHOT']:
            if key in self.__dict__.keys() : 
                hostphotz = self.__dict__[key] 
                if type( hostphotz ) == str : 
                    zphot = float(hostphotz.split()[0])
                    break
                else : 
                    zphot = float(hostphotz)
                    break
        if zphot>0 : return( zphot ) 
        else : return( 0 ) 

    @property
    def zphoterr(self):
        zphoterr=None
        for key in ['HOST_GALAXY_PHOTO-Z_ERR','ZPHOTERR']:
            if key in self.__dict__.keys() : 
                zphoterr = float(self.__dict__[key])
                break
        if not zphoterr : 
            for key in ['HOST_GALAXY_PHOTO-Z',]:
                if key in self.__dict__.keys() : 
                    hostphotz = self.__dict__[key] 
                    if type( hostphotz ) == str : 
                        zphoterr = float(hostphotz.split()[2])
                        break 
        if zphoterr>0 : return( zphoterr ) 
        else : return( 0 ) 

    @property
    def zspec(self):
        zspec=None
        for key in ['HOST_GALAXY_SPEC-Z','SN_SPEC-Z','ZSPEC']:
            if key in self.__dict__.keys() : 
                hostspecz = self.__dict__[key] 
                if type( hostspecz ) == str : 
                    zspec = float(hostspecz.split()[0])
                    break
                else : 
                    zspec = float(hostspecz)
                    break
        if zspec>0 : return( zspec ) 
        else : return( 0 ) 

    @property
    def zspecerr(self):
        zspecerr=None
        for key in ['HOST_GALAXY_SPEC-Z_ERR','SN_SPEC-Z_ERR','ZSPECERR']:
            if key in self.__dict__.keys() : 
                zspecerr = float(self.__dict__[key])
                break
        if not zspecerr : 
            for key in ['HOST_GALAXY_SPEC-Z','SN_SPEC-Z']:
                if key in self.__dict__.keys() : 
                    specz = self.__dict__[key] 
                    if type( specz ) == str : 
                        zspecerr = float(specz.split()[2])
                        break 
        if zspecerr>0 : return( zspecerr )
        else : return( 0 ) 

    @property
    def z(self):
        zfin = None
        if 'REDSHIFT_FINAL' in self.__dict__ : 
            if type( self.REDSHIFT_FINAL ) == str : 
                zfin = float(self.REDSHIFT_FINAL.split()[0])
            else : 
                zfin = float(self.REDSHIFT_FINAL)
            if zfin > 0 : return( zfin ) 
        elif 'REDSHIFT' in self.__dict__ : return( self.REDSHIFT ) 
        elif self.zspec > 0 : return( self.zspec ) 
        elif self.zphot > 0 : return( self.zphot ) 
        elif 'SIM_REDSHIFT' in self.__dict__ : return( self.SIM_REDSHIFT ) 
        else : return( 0 )

    @property
    def zerr(self):
        # TODO : better discrimination of possible redshift labels
        if ( 'REDSHIFT_FINAL' in self.__dict__.keys() and
             type( self.REDSHIFT_FINAL ) == str ):
            return( float(self.REDSHIFT_FINAL.split()[2]))
        elif ( 'REDSHIFT_ERR' in self.__dict__.keys() ): 
            if type( self.REDSHIFT_ERR ) == str :
                return( float(self.REDSHIFT_ERR.split()[0]) )
            else : 
                return(self.REDSHIFT_ERR)
        if self.zspecerr > 0 : return( self.zspecerr ) 
        elif self.zphoterr > 0 : return( self.zphoterr ) 
        else : return( 0 )

    @property
    def nobs(self):
        return( len(self.FLUXCAL) )

    @property
    def chi2_ndof(self):
        """ The reduced chi2. 
        !! valid only for models that have been fit to observed data !!
        """
        if 'CHI2VEC' in self.__dict__ and 'NDOF' in self.__dict__ : 
            return( self.CHI2VEC.sum() / self.NDOF )
        elif 'CHI2' in self.__dict__ and 'NDOF' in self.__dict__ : 
            return( self.CHI2 / self.NDOF )
        else : 
            return( 0 ) 

    @property
    def chi2(self):
        """ The raw (unreduced) chi2. 
        !! valid only for models that have been fit to observed data !!
        """
        if 'CHI2VEC' in self.__dict__  : 
            return( self.CHI2VEC.sum() )
        elif 'CHI2' in self.__dict__  : 
            return( self.CHI2 )
        else : 
            return( 0 ) 

    def readdatfile(self, datfile, readspec=True ):
        """ read the light curve data from the SNANA-style .dat file.
        Metadata in the header are in "key: value" pairs
        Observation data lines are marked with OBS: 
        and column names are given in the VARLIST: row.
        Comments are marked with #.
        """
        # TODO : could make the data reading more general: instead of assuming the 6 known 
        #   columns, just iterate over the varlist.
        from numpy import array,log10,unique,where

        self.datfile = os.path.abspath(datfile)
        if not os.path.isfile(datfile) and not os.path.isfile(datfile+'.gz'): raise RuntimeError( "%s does not exist."%datfile)
        elif os.path.isfile(datfile+'.gz'):
            datfile = datfile+'.gz'
            fin = gzip.open(datfile,'rt')
        elif datfile.endswith('.gz'):
            fin = gzip.open(datfile,'rt')
        else:
            fin = open(datfile,'r')


        data = fin.readlines()
        fin.close()
        flt,mjd=[],[]
        fluxcal,fluxcalerr=[],[]
        mag,magerr=[],[]

        # read header data and observation data
        for i in range(len(data)):
            line = data[i]
            if(len(line.strip())==0) : continue
            if line.startswith("#") : continue 
            if line.startswith('END:') : break
            if line.startswith('VARLIST:'):
                colnames = line.split()[1:]
                for col in colnames : 
                    self.__dict__[col] = []
            elif line.startswith('NOBS:'):
                nobslines = int(line.split()[1])
            elif line.startswith('NVAR:'):
                ncol = int(line.split()[1])
            elif 'OBS:' in line : 
                obsdat = line.split()[1:]
                for col in colnames : 
                    icol = colnames.index(col)
                    # removing this ugly hack for now!
                    #if col in ['FLT','BAND'] and len(obsdat[icol]) > 1:
                        # for SALTShaker, we need to just look at the last character
                        # simplest quick fix now that filters in LC files are intelligible
                    #    self.__dict__[col].append( str2num(obsdat[icol]) )
                    #else:
                    self.__dict__[col].append( str2num(obsdat[icol]) )

            else : 
                colon = line.find(':')
                key = line[:colon].strip()
                val = line[colon+1:].strip()
                
                if key.lower() in ['name'  , 'snid'  ,  'nickname'  ,'cid'  , 'iauname' ]:
#                    import pdb; pdb.set_trace()
                    self.__dict__[ key ] = val.split('#')[0].strip()
                else:
                    self.__dict__[ key ] = str2num(val)

        for col in colnames : 
            self.__dict__[col] = array( self.__dict__[col] )

        if readspec: self.readspecfromlcfile(datfile)
        
        return( None )

    def readspecfromlcfile(self,datfile):
        """read spectroscopy from the SNANA lightcurve file"""

        if datfile.endswith('.gz'):
            with gzip.open(datfile,'rt') as fin:
                reader = [x.split()[1:] for x in fin if x.startswith('SPEC:')]
            fin = gzip.open(datfile,'rt')
        else:
            with open(datfile) as fin:
                reader = [x.split()[1:] for x in fin if x.startswith('SPEC:')]
            fin = open(datfile)

        self.SPECTRA = {}
        startSpec = False
        i = 0
        for line in fin:
            if line.startswith('SPECTRUM_ID'):
                startSpec = True
                specid = int(line.split()[1])-1
                self.SPECTRA[specid] = {}
                istart = i
            elif (line.startswith('END_SPECTRUM') or line.startswith('SPECTRUM_END')) and startSpec:
                startSpec = False; iend = i
                j = 0
                for column in zip(*reader):
                    self.SPECTRA[specid][specvarnames[j]] = np.array(column[:][istart:iend]).astype(float)
                    j += 1
            elif line.startswith('VARNAMES_SPEC'):
                specvarnames = line.split()[1:]
            elif startSpec and not line.startswith('SPEC:'):
                try:
                    self.SPECTRA[specid][line.split()[0][:-1]] = float(line.split('#')[0].split()[1])
                except:
                    self.SPECTRA[specid][line.split()[0][:-1]] = line.split('#')[0].split()[1]
            elif startSpec and line.startswith('SPEC'):
                i += 1
        fin.close()
        
    def readnewspec(self,datfile):
        """read spectroscopy from an ascii file with columns wavelength, flux, (fluxerr)
Header must include one of two following lines at the top:

MJD-OBS: <spectrum MJD>
DATE-OBS: <spectrum date (isot format)>

Each file should contain only one spectrum
"""

        if 'SPECTRA' not in self.__dict__.keys():
            self.SPECTRA = {}

        specid = -1
        for i in range(100):
            if i not in self.SPECTRA.keys():
                specid = i
                self.SPECTRA[specid] = {}
                break
        if specid == -1:
            raise RuntimeError('something really weird happened or you have >100 spectra for this object already!')
            
        startSpec = False
        with open(datfile) as fin:
            for line in fin:
                line = line.replace('\n','')
                if 'MJD_OBS:' in line:
                    self.SPECTRA[specid]['SPECTRUM_MJD'] = float(line.split()[-11])
                elif 'DATE_OBS:' in line:
                    self.SPECTRA[specid]['SPECTRUM_MJD'] = date_to_mjd(line.split()[-1])
                elif not startSpec:
                    specvals = line.split()
                    if len(specvals) == 2:
                        specvarnames = ['LAMAVG','FLAM']
                    elif len(specvals) == 3:
                        specvarnames = ['LAMAVG','FLAM','FLAMERR']
                    else:
                        raise RuntimeError('format of file %s not understood'%datfile)
                
                    for v in specvarnames:
                        self.SPECTRA[specid][v] = np.array([])

                    for v,sv in zip(specvarnames,specvals):
                        self.SPECTRA[specid][v] = np.append(self.SPECTRA[specid][v],float(sv))
                    startSpec = True
                elif startSpec:
                    specvals = line.split()
                    for v,sv in zip(specvarnames,specvals):
                        self.SPECTRA[specid][v] = np.append(self.SPECTRA[specid][v],float(sv))
        if 'SPECTRUM_MJD' not in self.SPECTRA[specid].keys():
            raise RuntimeError('MJD_OBS or DATE_OBS keys must be in spectrum header')
                        
    def writedatfile(self, datfile, mag2fluxcal=False, **kwarg ):
        """ write the light curve data into a SNANA-style .dat file.
        Metadata in the header are in "key: value" pairs
        Observation data lines are marked with OBS: 
        and column names are given in the VARLIST: row.
        Comments are marked with #.

        mag2fluxcal : convert magnitudes and errors into fluxcal units
          and update the fluxcal and fluxcalerr arrays before writing
        """
        from numpy import array,log10,unique,where

        if mag2fluxcal : 
            from .. import hstsnphot
            self.FLUXCAL, self.FLUXCALERR = mag2fluxcal( self.MAG, self.MAGERR ) 

        fout = open(datfile,'w')
        for key in ['SURVEY','NICKNAME','SNID','IAUC','PHOTOMETRY_VERSION',
                    'SNTYPE','FILTERS','MAGTYPE','MAGREF','DECLINER',
                    'RA','DECL','MWEBV','REDSHIFT_FINAL',
                    'HOST_GALAXY_PHOTO-Z','HOST_GALAXY_SPEC-Z','REDSHIFT_STATUS',
                    'SEARCH_PEAKMJD','SEARCH_PEAKMJDERR',
                    'PEAKMJD','PEAKMJDERR',
                    'SIM_SALT2c','SIM_SALT2x1','SIM_SALT2mB','SIM_SALT2alpha',
                    'SIM_SALT2beta','SIM_REDSHIFT','SIM_PEAKMJD'] : 
            if key in kwarg : 
                print('%s: %s'%(key,str(kwarg[key])),file=fout)
            elif key in self.__dict__ : 
                print('%s: %s'%(key,str(self.__dict__[key])),file=fout)
        print('\nNOBS: %i'%len(self.FLUXCAL),file=fout)
        if 'MAG' in self.__dict__.keys():
            print('NVAR: 7',file=fout)
            print('VARLIST:  MJD    FLT FIELD   FLUXCAL   FLUXCALERR    MAG     MAGERR\n',file=fout)
            
            for i in range(self.nobs):
                print('OBS: %9.3f  %s  %s %8.3f %8.3f %8.3f %8.3f'%(
                    self.MJD[i], self.FLT[i], self.FIELD[i], self.FLUXCAL[i], 
                    self.FLUXCALERR[i], self.MAG[i], self.MAGERR[i] ),file=fout)
        else:
            print('NVAR: 5',file=fout)
            print('VARLIST:  MJD    FLT FIELD   FLUXCAL   FLUXCALERR\n',file=fout)
            
            for i in range(self.nobs):
                print('OBS: %9.3f  %s  %s %8.3f %8.3f'%(
                    self.MJD[i], self.FLT[i], self.FIELD[i], self.FLUXCAL[i], 
                    self.FLUXCALERR[i] ),file=fout)

        if 'SPECTRA' in self.__dict__.keys():
            print("""\nEND_PHOTOMETRY:

# =============================================
NSPECTRA:  %i
"""%len(self.SPECTRA.keys()),file=fout)
            for k in self.SPECTRA.keys():
                if k == list(self.SPECTRA.keys())[0]:
                    nvar_spec = len(self.SPECTRA[k].keys())
                    print('NVAR_SPEC: %i'%len(self.SPECTRA[k].keys()),file=fout)
                if nvar_spec != len(self.SPECTRA[k].keys()):
                    raise RuntimeError('number of variables for spectrum %s is inconsistent with the other spectra!'%k)
                varnames_line = 'VARNAMES_SPEC: '
                specarray_varnames = []
                specheader = ''
                for ks in ['LAMMIN','LAMMAX','LAMAVG','FLAM','FLAMERR','SIM_GENFLAM','SIM_GENMAG']:
                    if ks in self.SPECTRA[k].keys():
                        varnames_line += '%s '%ks
                        specarray_varnames += [ks]
                for ks in self.SPECTRA[k].keys():
                    if ks not in specarray_varnames:
                        specheader += '%s: %s\n'%(ks,self.SPECTRA[k][ks])
                if k == list(self.SPECTRA.keys())[0]: print('%s\n'%varnames_line,file=fout)
                
                print('SPECTRUM_ID: %i'%k,file=fout)
                print(specheader,file=fout)
                if not len(specarray_varnames):
                    raise RuntimeError('couldn\'t find wavelength/flux keys for spectrum')
                for i in range(len(self.SPECTRA[k][specarray_varnames[0]])):
                    outline = 'SPEC:  '
                    for ks in specarray_varnames:
                        outline += '%8.5e '%self.SPECTRA[k][ks][i]
                    print(outline,file=fout)
                print('END_SPECTRUM:\n',file=fout)
        print('\nEND:',file=fout)
        fout.close()
        return( datfile )

    def appendspec2snanafile(self, outfile, specdir=None, specfile=None, verbose=False, ps=False, mjd=None, **kwarg ):
        """ function to write jla data in snana format. Use as
        sn_lc=jla.SuperNova(full_path_of_jla_data/lc-sn_name.list)
        sn_lc.writesnanafile(full_path_of_jla_data/lc-sn_name.list)
        """
        from numpy import array,log10,unique,where,absolute
        if 'FLT' not in self.__dict__.keys():
            self.FLT = self.BAND[:]
        #if verbose:     print("Writing data from light curve file %s to snana format"%(datfile))
        fout = open(outfile,'w')
        list_snana_headers=['SURVEY','SNID','RA','DEC','MWEBV','REDSHIFT_HELIO','MJDPK']
        list_jla_headers=['SURVEY','SN','RA','DEC','MWEBV','Z_HELIO','DayMax']
        non_header_keys = ['NOBS','NVAR','MJD','FLT','FIELD','FLUXCAL','FLUXCALERR','MAG','MAGERR','datfile']
        for key in self.__dict__.keys(): #list_jla_headers : 
            if key not in non_header_keys:
                #icol = list_jla_headers.index(key)
                if key == 'Z_HELIO' :
                    print('%s: %s +- %s'%(key,str(self.__dict__[key]),str(self.__dict__['Redshift_err'])),file=fout)
                else :
                    print('%s: %s'%(key,str(self.__dict__[key])),file=fout)             
        print('\nNOBS: %i'%len(self.MJD),file=fout)
        print('NVAR: 6',file=fout)
        print('VARLIST:  MJD    FLT FIELD   FLUXCAL   FLUXCALERR\n',file=fout)
        for i in range(self.nobs):
            #Factor_275=(27.5-self.ZP[i])/(-2.5)
            #flux_275=self.Flux[i]/(10**Factor_275)
            #fluxerr_275=absolute(1.0/10**Factor_275)*self.Fluxerr[i]
            print('OBS: %9.3f  %s  %s %8.7f %8.7f'%(
                self.MJD[i], self.FLT[i], 'NULL', self.FLUXCAL[i], 
                self.FLUXCALERR[i]),file=fout)
        print('END_PHOTOMETRY:\n',file=fout)
        print('# =============================================',file=fout)
        #self.datfile = os.path.abspath(datfile)
        #folder_data=datfile[0:datfile.rfind('/')]+'/'
        #import pdb; pdb.set_trace()
        if ps: self.SNID = '%06i'%self.SNID
        if type(self.SNID) == float: self.SNID == str(int(self.SNID))

        if specfile is None:
            list_file_spec=glob.glob(specdir+'/spectrum*'+str(self.__dict__['SNID'])+'*.list')
        else:
            list_file_spec = np.atleast_1d(specfile)
        
        num_spec = [int(l.split('-')[-1].split('.')[0]) for l in list_file_spec]
        if not len(list_file_spec) and isinstance(self.SNID,str):
            snid2 = self.SNID[:]
            snid2 = snid2[:-1]+snid2[-1].lower()
            list_file_spec=glob.glob(specdir+'/spectrum*'+snid2+'*.list')
            num_spec = [int(l.split('-')[-1].split('.')[0]) for l in list_file_spec]
        
        if not len(list_file_spec):
            if verbose:
                print('warning: no spectrum for SNID %s'%self.SNID)
            return( None )
        #else:
        #   print(self.SNID,list_file_spec[0])
        print('\nNSPECTRA: %i \n'%len(list_file_spec),file=fout)
        print('\nNVAR_SPEC: 5',file=fout)
        print('VARNAMES_SPEC: LAMMIN LAMMAX  FLAM  FLAMERR DQ\n',file=fout)
        counter=0
        for specfile in np.array(list_file_spec)[np.argsort(num_spec)]:
            counter=counter+1
            sn_spectrum=SuperNovaSpectrum(specfile)
            print('SPECTRUM_ID: %i'%counter,file=fout)
            print('SPECTRUM_MJD: %9.2f'%sn_spectrum.mjdspec,file=fout)
            resolution=sn_spectrum.WAVE[1]-sn_spectrum.WAVE[0]
            for wl,fl,flerr,dq in zip(sn_spectrum.WAVE,sn_spectrum.FLUX,sn_spectrum.FLUXERR,sn_spectrum.VALID):
                wl_l=wl-resolution/2.0
                wl_u=wl+resolution/2.0
                if fl == fl and flerr == flerr: print('SPEC: %9.2f %9.2f %9.5e %9.5e %i'%(wl_l,wl_u,fl,flerr,dq),file=fout)
            print('SPECTRUM_END:\n',file=fout)  
        fout.close()

        return( None )

    
    def getheadfits( self, headfitsfile=None, headfits=None ) :
        """ read header data for the given SN from a binary fits table
        generated using the SNANA Monte Carlo simulator (GENSOURCE=RANDOM)
        """
        # read in the fits bin table headers and data
        #sndataroot = os.environ['SNDATA_ROOT']
        #simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simdir )
        #headfitsfiles = glob.glob(os.path.join( simdatadir, '%s_*-????_HEAD.FITS.gz'%self.simname))
        #print(headfitsfiles)
        #self.simnum = None
        #if not len( headfitsfiles ) : return(False)
        #for headfits in headfitsfiles:
        if headfitsfile is not None:
            hhead = fits.getheader( headfitsfile, ext=1 ) 
            hdata = fits.getdata( headfitsfile, ext=1 )
            self.headfitswithsnid = headfitsfile
        else:
            hhead = headfits[1].header
            hdata = headfits[1].data

        #self.simnum = int(headfitsfile.split('-')[-1].split('_HEAD')[0])
 

        # collect header data into object properties.
        Nsn = hhead['NAXIS2']    # num. of rows in the table = num. of simulated SNe
        Nhcol = hhead['TFIELDS'] # num. of table columns  = num. of data arrays
        snidlist = np.array([ int( hdata[isn]['SNID'] ) for isn in range(Nsn) ])

        if self.snid not in snidlist : 
            raise RuntimeError(f'SNID {self.snid} not in list')
        isn = np.where( snidlist== self.snid )[0][0]
        for ihcol in range( Nhcol ) : 
            self.__dict__[ hhead['TTYPE%i'%(ihcol+1)] ] = hdata[isn][ihcol] 

        # Make some shortcut aliases to most useful metadata
        for alias, fullname in ( [['z','SIM_REDSHIFT'], ['type','SNTYPE'],['mjdpk','SIM_PEAKMJD'],
                                  ['Hpk','SIM_PEAKMAG_H'],['Jpk','SIM_PEAKMAG_J'],['Wpk','SIM_PEAKMAG_W'],
                                  ['Zpk','SIM_PEAKMAG_Z'],['Ipk','SIM_PEAKMAG_I'],['Xpk','SIM_PEAKMAG_X'],['Vpk','SIM_PEAKMAG_V'],
        ]) : 
            if fullname in self.__dict__.keys() : 
                if alias=='type' : self.__dict__[alias] = SNTYPEDICT[self.__dict__[fullname]]
                else : self.__dict__[alias] = self.__dict__[fullname]
        return(True)
        raise RuntimeError( "SNID %s is not in %s"%(self.snid,headfits) )
        
    def getphotfits( self, photfitsfile=None, photfits=None ) :
        """ read phot data for the given SN from the photometry fits table
        generated using the SNANA Monte Carlo simulator (GENSOURCE=RANDOM)
        """
        # read in the fits bin table headers and data
        #sndataroot = os.environ['SNDATA_ROOT']
        #simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simdir )
        #photfitsfiles = glob.glob(os.path.join( simdatadir, self.headfitswithsnid.replace('HEAD','PHOT')))
        #if not len( photfitsfiles ) : return(False)
        #for photfits in photfitsfiles
        if photfitsfile is not None:
            phead = fits.getheader( photfitsfile, ext=1 ) 
            pdata = fits.getdata( photfitsfile, ext=1 ) 
        else:
            phead = photfits[1].header
            pdata = photfits[1].data
        
        # find pointers to beginning and end of the obs sequence
        sndata = pdata[ self.PTROBS_MIN-1:self.PTROBS_MAX ]

        # collect phot data into object properties.
        Npcol = phead['TFIELDS'] # num. of table columns  = num. of data arrays
        for ipcol in range( Npcol ) : 
            self.__dict__[ phead['TTYPE%i'%(ipcol+1)] ] = np.array( [ sndata[irow][ipcol] for irow in range(self.NOBS) ] )
        return( True )

    def getspecfits( self, specfitsfile=None, specfits=None ) :
        """ read phot data for the given SN from the photometry fits table
        generated using the SNANA Monte Carlo simulator (GENSOURCE=RANDOM)
        """
        # read in the fits bin table headers and data
        #sndataroot = os.environ['SNDATA_ROOT']
        #simdatadir = os.path.join( sndataroot,'SIM/%s'%self.simdir )
        #photfitsfiles = glob.glob(os.path.join( simdatadir, self.headfitswithsnid.replace('HEAD','PHOT')))
        #if not len( photfitsfiles ) : return(False)
        #for photfits in photfitsfiles:

        if specfitsfile is not None:
            hdu = fits.open(specfitsfile)
        else:
            hdu = specfits
            
        #to adapt to snana spectra format change
        headertbl_idx = [hdu[i].name for i in range(len(hdu))].index('SPECTRO_HEADER')
        fluxtbl_idx = [hdu[i].name for i in range(len(hdu))].index('SPECTRO_FLUX')
        if 'SPECTRO_LAMINDEX' in [hdu[i].name for i in range(len(hdu))]:
            lamidxtbl_idx =  [hdu[i].name for i in range(len(hdu))].index('SPECTRO_LAMINDEX')
        else:
            lamidxtbl_idx = None

        # collect phot data into object properties.
        #Npcol = phead['TFIELDS'] # num. of table columns  = num. of data arrays
        #for ipcol in range( Npcol ) : 
        #   self.__dict__[ phead['TTYPE%i'%(ipcol+1)] ] = np.array( [ sndata[irow][ipcol] for irow in range(self.NOBS) ] )
        #import pdb; pdb.set_trace()

        self.SPECTRA = {}
        specid = 0
        for i,snid in enumerate(hdu[headertbl_idx].data['SNID']):
            if self.SNID != snid: continue
            if hdu[headertbl_idx].data['NBIN_LAM'][i] == 0: continue
            self.SPECTRA[specid] = {}
            #self.SPECTRA[specid]['LAMINDEX'] = hdu[lamidxtbl_idx].data['LAMINDEX']

            if lamidxtbl_idx is None:
                self.SPECTRA[specid]['LAMMIN'] = hdu[fluxtbl_idx].data['LAMMIN'][hdu[headertbl_idx].data['PTRSPEC_MIN'][i]-1:hdu[headertbl_idx].data['PTRSPEC_MAX'][i]]
                self.SPECTRA[specid]['LAMMAX'] = hdu[fluxtbl_idx].data['LAMMAX'][hdu[headertbl_idx].data['PTRSPEC_MIN'][i]-1:hdu[headertbl_idx].data['PTRSPEC_MAX'][i]]
            else:
                self.SPECTRA[specid]['LAMMIN'] = np.array([hdu[lamidxtbl_idx].data['LAMMIN'][hdu[lamidxtbl_idx].data['LAMINDEX'] == hdu[fluxtbl_idx].data['LAMINDEX'][i]][0] \
                    for i in range(hdu[headertbl_idx].data['PTRSPEC_MIN'][i]-1,hdu[headertbl_idx].data['PTRSPEC_MAX'][i])])
                self.SPECTRA[specid]['LAMMAX'] = np.array([hdu[lamidxtbl_idx].data['LAMMAX'][hdu[lamidxtbl_idx].data['LAMINDEX'] == hdu[fluxtbl_idx].data['LAMINDEX'][i]][0] \
                    for i in range(hdu[headertbl_idx].data['PTRSPEC_MIN'][i]-1,hdu[headertbl_idx].data['PTRSPEC_MAX'][i])])
            self.SPECTRA[specid]['FLAM'] = hdu[fluxtbl_idx].data['FLAM'][hdu[headertbl_idx].data['PTRSPEC_MIN'][i]-1:hdu[headertbl_idx].data['PTRSPEC_MAX'][i]]
            self.SPECTRA[specid]['FLAMERR'] = hdu[fluxtbl_idx].data['FLAMERR'][hdu[headertbl_idx].data['PTRSPEC_MIN'][i]-1:hdu[headertbl_idx].data['PTRSPEC_MAX'][i]]
            self.SPECTRA[specid]['SPECTRUM_MJD'] = hdu[headertbl_idx].data['MJD'][i]
            specid += 1

        return( True )

    
    def plotLightCurve(self, ytype='flux', xtype='mjd', bands='all', mjdpk=None,
                       showlegend=False, showpkmjdrange=False, 
                       showsalt2fit=False, showclassfit=False, 
                       showclasstable=True, 
                       filled=False, autozoom=True, 
                       savefig='', verbose=False,  **kwarg ) : 
        """ Plot the observed multi-color light curve data. 
        WFC3-IR filters are plotted as circles with solid lines
        ACS-WFC bands are squares with dashed lines
        WFC3-UVIS filters are shown as triangles with dotted lines
                
          OPTIONS 
        ytype : 'flux', 'mag', 'chi2Ia', 'chi2Ibc', 'chi2II'  
            - Default of 'flux' uses SNANA's FLUXCAL units, ZPT:27.5
            - The 'chi2' options presume an existing maxLikeModel, and plot
              the chi2 contribution from each light curve point 
        xtype : 'mjd', 'tobs', 'trest'  (tobs and trest are days rel. to peak)
        bands : list of SNANA filter IDs or 'all'  (e.g.  bands='HJW')

        showlegend : put a legend in the upper corner
        showpkmjdrange : add a vertical line and bar marking the range of PKMJD
        showsalt2fit : overplot the best-fit SALT2 model (if available)
        showclassfit : 'Ia.maxlike', 'II.maxlike', 'Ibc.maxlike', 'Ia.maxprob', 'II.maxprob', 'Ibc.maxprob' 
            overplot the best-fit model (either max likelihood or max posterior probability) 
            for the given class from a (previously executed) classification simulation. 
        showclasstable : print a table of parameter values and chi2 statistics for the 
            best-fit model on the right side of the figure
        savefig : filename for saving the figure directly to disk (extension sets the filetype)

           (The following options are typically used for plotting finely sampled models, 
            like a SALT2 model fit or a max likelihood model from a classification sim)
        filled : plot semi-transparent filled curves instead of points and connecting lines
        autozoom : True/False to toggle on/off the automatic rescaling 

        Any additional keyword args are passed to the matplotlib.pyplot.plot() function 
          (e.g: ms=10, ls=' '  to plot large markers with no lines)
        """ 
        from matplotlib.patches import FancyArrowPatch
        fig = p.gcf()
        ax = fig.gca()

        if showpkmjdrange :
            if ytype=='flux':
                ymax=self.FLUXCAL.max() + 3*self.FLUXCALERR.max()
                ymin=self.FLUXCAL.min() - 3*self.FLUXCALERR.max()
            else : 
                ymax=self.MAG.max() + 3*self.MAGERR.max()
                ymin=self.MAG.min() - 3*self.MAGERR.max()
            if xtype=='mjd' : 
                ax.axvline( self.pkmjd, color='0.5',lw=0.7,ls='--' )
                pkmjdbar = patches.Rectangle( [ self.pkmjd-self.pkmjderr, ymin], 2*self.pkmjderr, ymax+abs(ymin), 
                                              color='0.5', alpha=0.3, zorder=-100 )
                ax.add_patch( pkmjdbar )
            elif xtype=='trest' : 
                ax.axvline( 0.0, color='0.5',lw=0.7,ls='--' )
                pkmjdbar = patches.Rectangle( [ -self.pkmjderr/(1+self.z), ymin], 2*self.pkmjderr/(1+self.z), ymax+abs(ymin), 
                                              color='0.5', alpha=0.3, zorder=-100 )
                ax.add_patch( pkmjdbar )
            elif xtype=='tobs' : 
                ax.axvline( 0.0, color='0.5',lw=0.7,ls='--' )
                pkmjdbar = patches.Rectangle( [ -self.pkmjderr, ymin], 2*self.pkmjderr, ymax+abs(ymin), 
                                              color='0.5', alpha=0.3, zorder=-100 )
                ax.add_patch( pkmjdbar )

        
        if( showsalt2fit ) : 
            if ( 'salt2fitModel' in self.__dict__ and 'salt2fit' in self.__dict__ ) : 
                self.salt2fitModel.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                                   showlegend=False, autozoom=autozoom, 
                                                   showsalt2fit=False, filled=True, marker=' ', ls='-')
               
                ax = p.gca()
                if showclasstable : 
                    ax.text(0.95,0.95, "Ia (SALT2)\nz=%.3f\nx1=%.2f\nc=%.2f\nmB=%.2f\nchi2=%.2f/%i=%.2f\np=%.3f"%(
                            self.salt2fit.z,self.salt2fit.x1,self.salt2fit.c,self.salt2fit.mB,self.salt2fit.CHI2,
                            self.salt2fit.NDOF,self.salt2fit.CHI2/self.salt2fit.NDOF,
                            self.salt2fit.FITPROB), transform=ax.transAxes,ha='right',va='top' )
            else : 
                print("No SALT2 fit model available.  Use self.getSALT2fit()" )

        if str(showclassfit).startswith('Ia') :
            if 'ClassMC' in self.__dict__ :  
                bestFitModelIa = self.ClassMC.maxLikeModels.Ia
                PIa = self.ClassMC.PIa
            elif 'ClassSim' in self.__dict__ : 
                if showclassfit.endswith('like'): bestFitModelIa = self.maxLikeIaModel
                else : bestFitModelIa = self.maxProbIaModel
                PIa = self.PIa
            else :  bestFitModelIa = None               
            if bestFitModelIa == None : 
                print("The max likelihood Ia model is not available.")
                print("Run doGridClassify or doClassifyMC + getMaxLikeModelsMC" )
            else : 
                bestFitModelIa.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                               showlegend=False, autozoom=autozoom, 
                                               showsalt2fit=False, showclassfit=False,
                                               filled=True, marker=' ')
                if self.ClassSim.Ia.USELUMPRIOR==0 : magoffsetString = '$\Delta$m=%.1f\n'%bestFitModelIa.MAGOFF 
                else : magoffsetString = ''
                ax = p.gca()
                if showclasstable : 
                    ax.text(0.95, 0.95, "%s(%s)\nz=%.3f\n$x_{1}$=%.2f\n$\mathcal{C}$=%.2f\n$\\beta$=%.1f\nMJD$_{pk}$=%.1f\n%s$\chi^2_{\\nu}$=%.1f/%i=%.1f\nP(Ia)=%.2f"%(
                            bestFitModelIa.TYPE, bestFitModelIa.TEMPLATE.replace('_','').replace('+','').split('.')[0], 
                            bestFitModelIa.REDSHIFT,bestFitModelIa.LUMIPAR,bestFitModelIa.COLORPAR,bestFitModelIa.COLORLAW,
                            bestFitModelIa.PEAKMJD, magoffsetString, 
                            bestFitModelIa.chi2, bestFitModelIa.NDOF, bestFitModelIa.chi2_ndof, PIa), 
                            transform=ax.transAxes,ha='right',va='top' )
                mjdpk = bestFitModelIa.mjdpk

        elif str(showclassfit).startswith('Ibc'):
            if 'ClassMC' in self.__dict__ :  
                bestFitModelIbc = self.ClassMC.maxLikeModels.Ibc
                PIbc = self.ClassMC.PIbc
            elif 'ClassSim' in self.__dict__ : 
                if str(showclassfit).endswith('like'): bestFitModelIbc = self.maxLikeIbcModel
                else : bestFitModelIbc = self.maxProbIbcModel
                PIbc = self.PIbc
            else :  bestFitModelIbc = None               
            if bestFitModelIbc == None : 
                print("The max likelihood Ibc model is not available.")
                print("Run doGridClassify or doClassifyMC + getMaxLikeModelsMC" )
            else : 
                bestFitModelIbc.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                                showlegend=False, autozoom=autozoom, 
                                                showsalt2fit=False, showclassfit=False,
                                                filled=True, marker=' ')
                ax = p.gca()
                if showclasstable : 
                    ax.text(0.95, 0.95, "%s(%s)\nz=%.3f\n$A_V$=%.2f\n$R_V$=%.1f\nMJD$_{pk}$=%.1f\n$\Delta$m=%.1f\n$\chi^2_{\\nu}$=%.1f/%i=%.1f\nP(Ib/c)=%.2f"%(
                            bestFitModelIbc.TYPE, bestFitModelIbc.TEMPLATE.replace('_','').replace('+',''), 
                            bestFitModelIbc.REDSHIFT,bestFitModelIbc.COLORPAR,bestFitModelIbc.COLORLAW,
                            bestFitModelIbc.PEAKMJD,bestFitModelIbc.MAGOFF,
                            bestFitModelIbc.chi2, bestFitModelIbc.NDOF, bestFitModelIbc.chi2_ndof, PIbc), 
                            transform=ax.transAxes,ha='right',va='top' )
                mjdpk = bestFitModelIbc.mjdpk
        elif str(showclassfit).startswith('II') :
            if 'ClassMC' in self.__dict__ :  
                bestFitModelII = self.ClassMC.maxLikeModels.II
                PII = self.ClassMC.PII
            elif 'ClassSim' in self.__dict__ : 
                if str(showclassfit).endswith('like'): bestFitModelII = self.maxLikeIIModel
                else : bestFitModelII = self.maxProbIIModel
                PII = self.PII
            else :  bestFitModelII = None               
            if bestFitModelII == None : 
                print("The max likelihood II model is not available.")
                print("Run doGridClassify or doClassifyMC + getMaxLikeModelsMC" )
            else : 
                bestFitModelII.plotLightCurve( ytype=ytype, xtype=xtype, bands=bands, 
                                               showlegend=False, autozoom=autozoom, 
                                               showsalt2fit=False, showclassfit=False,
                                               filled=True, marker=' ' )
                ax = p.gca()
                if showclasstable :
                    ax.text(0.95, 0.95, "%s(%s)\nz=%.3f\n$A_V$=%.2f\n$R_V$=%.1f\nMJD$_{pk}$=%.1f\n$\Delta$m=%.1f\n$\chi^2_{\\nu}$=%.1f/%i=%.1f\nP(II)=%.2f"%(
                            bestFitModelII.TYPE, bestFitModelII.TEMPLATE.replace('_','').replace('+',''), 
                            bestFitModelII.REDSHIFT,bestFitModelII.COLORPAR,bestFitModelII.COLORLAW,
                            bestFitModelII.PEAKMJD, bestFitModelII.MAGOFF,
                            bestFitModelII.chi2, bestFitModelII.NDOF, bestFitModelII.chi2_ndof, PII), 
                            transform=ax.transAxes,ha='right',va='top' )
                mjdpk = bestFitModelII.mjdpk

        if bands=='all' : bands = ''.join(self.bandlist)
        magmin,magmax=27,22
        fluxmin,fluxmax=0,2
        for band in reversed(self.BANDORDER) : 
            if band not in bands : continue
            iband = np.where( self.FLT == band )[ 0 ]
            mag = self.MAG[ iband ] 
            magerr = self.MAGERR[ iband ] 
            flux = self.FLUXCAL[ iband ] 
            fluxerr = self.FLUXCALERR[ iband ] 
            mjd = self.MJD[ iband ] 
#            camera = self.SURVEYDATA.band2camera( band )

            if mjdpk==None : mjdpk = self.pkmjd
            if xtype=='tobs' : x = mjd - mjdpk
            elif xtype=='trest' : x = ( mjd - mjdpk ) / (1+self.z)
            else : x = mjd
                       
            if ytype=='flux' : 
                y = flux
                yerr = fluxerr
                yUL = []
                if y.max() > fluxmax and y.max() != 99 : fluxmax = y.max()
                if y.min() < fluxmin and y.min() > -90 : fluxmin = y.min()
            elif ytype.startswith('chi2'): 
                if ytype.endswith('Ia'): maxLikeModel = self.maxLikeIaModel
                elif ytype.endswith('Ibc'): maxLikeModel = self.maxLikeIbcModel
                elif ytype.endswith('II'): maxLikeModel = self.maxLikeIIModel
                ibandchi2 = np.where( maxLikeModel.CHI2FLT == band )[0]
                if not len(ibandchi2) : continue
                y = maxLikeModel.CHI2VEC[ ibandchi2 ]
                yerr = np.zeros( len(y) )
                yUL = []
                mjd = maxLikeModel.CHI2MJD[ ibandchi2 ]
                if xtype=='tobs' : x = mjd - mjdpk
                elif xtype=='trest' : x = ( mjd - mjdpk ) / (1+self.z)
                else : x = mjd
            else : 
                iUpperLim = np.where( magerr<-8 )[0]
                iGoodMags = np.where( magerr>-8 )[0]
                y = mag[ iGoodMags ]
                yerr = magerr[ iGoodMags ]
                yUL = mag[ iUpperLim ]
                xUL = x[  iUpperLim ]
                x = x[ iGoodMags ]
                if len(y) : 
                    if y.max() > magmax : magmax = min(y.max(),28)
                    if y.min() < magmin : magmin = max(y.min(),13)

            lstyle,alpha = ' ',1
            if showsalt2fit or showclassfit : lstyle=' '

            defaultarg = { 'label':band, 'ls':lstyle, 'marker':'o','alpha':alpha,'color':'k' }
            fillarg = {'alpha': 0.3}
#            if 'BANDCOLOR' in self.SURVEYDATA.__dict__ : 
            defaultarg['color'] = BANDCOLOR[band]
            fillarg['color'] = BANDCOLOR[band]
#            if 'BANDMARKER' in self.SURVEYDATA.__dict__ : 
            defaultarg['marker'] = BANDMARKER[band] 
            plotarg = dict(defaultarg.items()+kwarg.items())
            if plotarg['marker']==' ': ax.plot( x, y, **plotarg )
            if filled : 
                if ytype=='mag': 
                    yerr = np.where( (yerr<8) & (yerr>-8), yerr, np.abs(yerr).min() )
                ax.fill_between( x, y-yerr, y+yerr, **fillarg)
            else : 
                if len(y) : 
                    ax.errorbar( x, y, yerr, **plotarg )
                if len(yUL) : 
                    if not len(y) : 
                        plotarg['marker']='.'
                        ax.plot( xUL, yUL, **plotarg )
                    for xx,yy in zip( xUL, yUL ):
                        arr = FancyArrowPatch( [xx,yy], [xx,yy+0.75], arrowstyle='-|>', mutation_scale=25, ls='solid', fc=plotarg['color'] )
                        ax.add_patch( arr )                    

        if ytype=='mag': 
            ax.set_ylabel( 'Vega mag' )
            if not autozoom : ax.set_ylim( magmax+0.1, magmin-0.1) 
            if not ax.yaxis_inverted() : ax.invert_yaxis()
        elif ytype=='flux':
            ax.set_ylabel( 'Flux' )
            if not autozoom : ax.set_ylim( fluxmin*0.9, fluxmax*1.1) 
        elif ytype.startswith('chi2'):
            ax.set_ylabel( r'$\chi^2$', rotation='horizontal' )

        if xtype=='mjd' : 
            ax.set_xlabel( 'obs frame time (MJD)' )
            if self.mjdpk > 0 and not autozoom : 
                ax.set_xlim( self.mjdpk - 20 * (1+self.z), self.mjdpk + 60 * (1+self.z) )
        elif xtype=='trest' : 
            ax.set_xlabel( 'rest frame time (days)' )
            ax.set_xlim( -20, 60 )
        elif xtype=='tobs' : 
            ax.set_xlabel( 'obs frame time (days rel. to peak)' )
        if showlegend : 
            leg = ax.legend( loc='upper left', frameon=False, numpoints=1, handlelength=0.2, handletextpad=0.4, labelspacing=0.2 )

        if savefig : p.savefig( savefig )
        p.draw()

def mag2fluxcal( mag, magerr=0 ):
    """ convert magnitudes into SNANA-style FLUXCAL units
    (fixed zero point of 27.5 for all bands) """
    from numpy import iterable, abs, array, zeros, any
    
    if not iterable( mag ) : 
        mag = array( [ mag ] )
        magerr = array( [ magerr ] )
    if not iterable( magerr ) : 
        magerr = zeros( len(mag) ) 

    fluxcal, fluxcalerr = [],[]
    for m,me in zip( mag, magerr) : 
        if me < 0 : 
            fluxcal.append( 0 ) 
            fluxcalerr.append( 10**(-0.4*(m-27.5)) )
        else : 
            fluxcal.append( 10**(-0.4*(m-27.5)) )
            fluxcalerr.append( 0.92103 * me * fluxcal[-1] )
    fluxcal = array( fluxcal )
    fluxcalerr = array( fluxcalerr )

    if len(mag)==1 : 
        fluxcal = fluxcal[0]
        fluxcalerr = fluxcalerr[0] 

    if any( magerr ) : return( fluxcal, fluxcalerr ) 
    else : return( fluxcal )

def str2num(s) :
    """ convert a string to an int or float, as appropriate.
    If neither works, return the string"""
    try: return int(s)
    except ValueError:
        try: return float(s)
        except ValueError: return( s )

def date_to_mjd(date):
    time = Time(date,scale='utc')
    return time.mjd
