import os
import numpy as np
from saltshaker.util import snana
from saltshaker.util.estimate_tpk_bazin import estimate_tpk_bazin
from astropy.io import fits
from saltshaker.initfiles import init_rootdir
from saltshaker.data import data_rootdir
from astroquery.irsa_dust import IrsaDust
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
from time import time
import scipy.stats as ss
import astropy.table as at
import logging
import abc
from copy import deepcopy

log=logging.getLogger(__name__)

class SNDataReadError(ValueError):
        pass
class BreakLoopException(RuntimeError):
        pass

def checksize(a,b):
        assert(a.size==b.size)
        
class SALTtrainingdata(metaclass=abc.ABCMeta):

    __slots__=[]
    
    @property
    @abc.abstractmethod
    def __listdatakeys__(self):
        pass
                
    def clip(self,clipcriterion):
        copy=deepcopy(self)
        for key in self.__listdatakeys__:
            setattr(copy,key,getattr(self,key)[clipcriterion])
        return copy
        
class SALTtraininglightcurve(SALTtrainingdata):
        
        __slots__=['mjd','tobs','phase','fluxcal','fluxcalerr','filt']
        
        def __init__(self,z,tpk_guess,flt,sn ):
                assert((sn.FLT==flt).sum()>0)
                inlightcurve= (sn.FLT==flt)
                self.mjd=sn.MJD[inlightcurve]
                sortinds=np.argsort(self.mjd)
                self.mjd=self.mjd[sortinds]
                self.tobs=(self.mjd-tpk_guess)
                self.phase=self.tobs/(1+z)
                self.fluxcal=sn.FLUXCAL[inlightcurve][sortinds]
                self.fluxcalerr=sn.FLUXCALERR[inlightcurve][sortinds]
                self.filt=flt
                checksize(self.tobs,self.mjd)
                checksize(self.mjd,self.fluxcal)
                checksize(self.mjd,self.fluxcalerr)
                
        __listdatakeys__={'tobs','mjd','phase','fluxcal','fluxcalerr'}
                
        def __len__(self):
                return len(self.tobs)

                
class SALTtrainingspectrum(SALTtrainingdata):

        __slots__=['flux', 'phase', 'wavelength', 'fluxerr', 'tobs','restwavelength','mjd','n_specrecal']

        def __init__(self,snanaspec,z,tpk_guess,n_specrecal=None,binspecres=None ):
                        m=snanaspec['SPECTRUM_MJD']
                        if snanaspec['FLAM'].size==0:
                                raise SNDataReadError(f'Spectrum has no observations')
                                
                        if 'LAMAVG' in snanaspec:
                                wavelength = snanaspec['LAMAVG']
                        elif 'LAMMIN' in snanaspec and 'LAMMAX' in snanaspec:
                                wavelength = (snanaspec['LAMMIN']+snanaspec['LAMMAX'])/2
                        else:
                                raise SNDataReadError('couldn\'t find wavelength data')
                        self.wavelength=wavelength
                        self.fluxerr=snanaspec['FLAMERR']
                        self.flux=snanaspec['FLAM']
                        self.tobs=m -tpk_guess
                        self.mjd=m
                        self.phase=self.tobs/(1+z)
                        self.n_specrecal = n_specrecal

                        if 'DQ' in snanaspec:
                                iGood=(snanaspec['DQ']==1)
                        elif 'SPECFLAG' in snanaspec:
                                iGood=(snanaspec['SPECFLAG']==1)
                        else:
                                iGood=np.ones(len(self),dtype=bool)
                        iGood = iGood & (~np.isnan(self.flux))
                        if ('DQ' in snanaspec and (snanaspec['DQ']==1).sum() == 0) or np.all(np.isnan(self.flux)):
                                raise SNDataReadError('Spectrum is all marked as invalid data')
                        if binspecres is not None:
                                flux = self.flux[iGood]
                                wavelength = self.wavelength[iGood]
                                fluxerr = self.fluxerr[iGood]
                                fluxmax = np.max(flux)
                                weights = 1/(fluxerr/fluxmax)**2.
                                
                                def weighted_avg(indices):
                                        """
                                        Return the weighted average and standard deviation.
                                        indices, weights -- Numpy ndarrays with the same shape.
                                        """

                                        average = np.average(flux[indices]/fluxmax, weights=weights[indices])
                                        variance = np.average((flux[indices]/fluxmax-average)**2, weights=weights[indices])      # Fast and numerically precise

                                        return average

                                def weighted_err(indices):
                                        """
                                        Return the weighted average and standard deviation.
                                        indices, weights -- Numpy ndarrays with the same shape.
                                        """
                                        average = np.average(flux[indices]/fluxmax, weights=weights[indices])
                                        variance = np.average((flux[indices]/fluxmax-average)**2, weights=weights[indices])      # Fast and numerically precise
                                        return np.sqrt(variance)

                                wavebins = np.linspace(np.min(wavelength),np.max(wavelength),int((np.max(wavelength)-np.min(wavelength))/(binspecres*(1+z))))
                                binned_flux = ss.binned_statistic(wavelength,range(len(flux)),bins=wavebins,statistic=weighted_avg).statistic
                                binned_fluxerr = ss.binned_statistic(wavelength,range(len(flux)),bins=wavebins,statistic=weighted_err).statistic

                                iGood = (binned_flux == binned_flux)

                                self.flux = binned_flux[iGood]
                                self.wavelength = (wavebins[1:][iGood]+wavebins[:-1][iGood])/2.
                                self.fluxerr = binned_fluxerr[iGood]
                        else:

                                self.flux = self.flux[iGood]
                                self.wavelength = self.wavelength[iGood]
                                self.fluxerr = self.fluxerr[iGood]

                        # error floor
                        self.fluxerr = np.hypot(self.fluxerr, 5e-3*np.max(self.flux))
                        self.restwavelength= self.wavelength/ (1+z)

                        for key in self.__listdatakeys__: 
                            checksize(self.wavelength,getattr(self,key) )
        
        __listdatakeys__={'wavelength','flux','fluxerr','restwavelength'}

        def __len__(self):
                return len(self.wavelength)
                        
class SALTtrainingSN:

        __slots__=['survey', 'zHelio', 'MWEBV', 'snid', 'tpk_guess', 'salt2fitprob', 'photdata','specdata','SIM_SALT2x0','SIM_SALT2x1','SIM_SALT2c']
        def __init__(self,sn,
                     estimate_tpk=False,snpar=None,
                     pkmjddict={},n_specrecal=None,binspecres=None):

                if 'FLT' not in sn.__dict__.keys():
                    raise SNDataReadError('can\'t find SN filters!')

                if 'SURVEY' in sn.__dict__.keys():
                    self.survey=sn.SURVEY
                else:
                    raise SNDataReadError('SN %s has no SURVEY key, which is needed to find the filter transmission curves'%sn.SNID[0])
                if not 'REDSHIFT_HELIO' in sn.__dict__.keys():
                    raise SNDataReadError('SN %s has no heliocentric redshift information in the header'%sn.SNID)

                if 'PEAKMJD' in sn.__dict__.keys(): sn.SEARCH_PEAKMJD = sn.PEAKMJD
                # FITS vs. ASCII format issue in the parser
                if isinstance(sn.REDSHIFT_HELIO,str): self.zHelio = float(sn.REDSHIFT_HELIO.split('+-')[0])
                else: self.zHelio = sn.REDSHIFT_HELIO

                if estimate_tpk:
                    if 'B' in sn.FLT:
                        tpk,tpkmsg = estimate_tpk_bazin(
                            sn.MJD[sn.FLT == 'B'],sn.FLUXCAL[sn.FLT == 'B'],sn.FLUXCALERR[sn.FLT == 'B'],max_nfev=100000,t0=sn.SEARCH_PEAKMJD)
                    elif 'g' in sn.FLT:
                        tpk,tpkmsg = estimate_tpk_bazin(
                            sn.MJD[sn.FLT == 'g'],sn.FLUXCAL[sn.FLT == 'g'],sn.FLUXCALERR[sn.FLT == 'g'],max_nfev=100000,t0=sn.SEARCH_PEAKMJD)
                    elif 'c' in sn.FLT:
                        tpk,tpkmsg = estimate_tpk_bazin(
                            sn.MJD[sn.FLT == 'c'],sn.FLUXCAL[sn.FLT == 'c'],sn.FLUXCALERR[sn.FLT == 'c'],max_nfev=100000,t0=sn.SEARCH_PEAKMJD)
                    else:
                        raise SNDataReadError(f'need a blue filter to estimate tmax')
                elif len(pkmjddict.keys()):
                    try:
                        tpk = pkmjddict[sn.SNID]
                        tpkmsg = 'success: peak MJD provided'
                    except KeyError:
                        tpkmsg = f'can\'t find tmax in pkmjd file'
                else:
                    tpk = sn.SEARCH_PEAKMJD
                    if type(tpk) == str:
                        tpk = float(sn.SEARCH_PEAKMJD.split()[0])
                    tpkmsg = 'success: peak MJD found in LC file'
                if 'success' not in tpkmsg:
                    raise SNDataReadError(f'can\'t estimate t_max for SN {sn.SNID}: {tpkmsg}')

                # to allow a fitprob cut
                if snpar is not None:
                    if 'FITPROB' in snpar.keys() and str(sn.SNID) in snpar['SNID']:
                        fitprob = snpar['FITPROB'][str(sn.SNID) == snpar['SNID']][0]
                    else:
                        fitprob = -99
                else:
                    fitprob = -99
                if 'SIM_SALT2x0' in sn.__dict__.keys():
                    self.SIM_SALT2x0 = sn.SIM_SALT2x0
                else:
                    self.SIM_SALT2x0 = None
                if 'SIM_SALT2x1' in sn.__dict__.keys():
                    self.SIM_SALT2x1 = sn.SIM_SALT2x1
                else:
                    self.SIM_SALT2x1 = None
                if 'SIM_SALT2c' in sn.__dict__.keys():
                    self.SIM_SALT2c = sn.SIM_SALT2c
                else:
                    self.SIM_SALT2c = None
                    
                #Find E(B-V) from Milky Way
                if 'MWEBV' in sn.__dict__.keys():
                    try: self.MWEBV = float(sn.MWEBV.split()[0])
                    except: self.MWEBV= float(sn.MWEBV)
                elif 'RA' in sn.__dict__.keys() and 'DEC' in sn.__dict__.keys():
                    log.warning('determining MW E(B-V) from IRSA for SN %s using RA/Dec in file'%sn.SNID)
                    sc = SkyCoord(sn.RA,sn.DEC,frame="fk5",unit=u.deg)
                    self.MWEBV = IrsaDust.get_query_table(sc)['ext SandF mean'][0]
                else:
                    raise SNDataReadError('Could not determine E(B-V) from files.   Set MWEBV keyword in input file header for SN %s'%sn.SNID)

                self.snid=sn.SNID
                self.tpk_guess=tpk
                self.salt2fitprob=fitprob
                
                self.photdata = {flt:SALTtraininglightcurve(self.zHelio,tpk_guess= self.tpk_guess,flt=flt, sn=sn) for flt in np.unique(sn.FLT)}
                try: assert(len(self.photdata)>0)
                except AssertionError:
                    raise SNDataReadError(f'No lightcurves for SN {sn.SNID}')
                self.specdata = {}

                if 'SPECTRA' in sn.__dict__:
                    for speccount,k in enumerate(sn.SPECTRA):
                        if n_specrecal is not None and k+1 in n_specrecal['N']:
                            n_recal_pars = n_specrecal['ncalib'][n_specrecal['N'] == k+1][0]
                        else: n_recal_pars = None
                        try:
                            self.specdata[speccount]=SALTtrainingspectrum(
                                sn.SPECTRA[k],self.zHelio,self.tpk_guess,
                                n_specrecal=n_recal_pars,
                                binspecres=binspecres)
                        except SNDataReadError as e:
                            if hasattr(e,'message'):
                                log.warning(f'{e.message}, skipping spectrum {k} for SN {self.snid}')
                            else:
                                log.warning(f'DataReadError, skipping spectrum {k} for SN {self.snid}')

        @property
        def num_lc(self):
            return len(self.photdata)
        
        @property
        def num_photobs(self):
            return sum([len(self.photdata[filt]) for filt in self.photdata])

        @property
        def num_specobs(self):
            return sum([len(self.specdata[key]) for key in self.specdata])
                
        @property
        def num_spec(self):
            return len(self.specdata)
                
        @property
        def filt(self):
            return list(self.photdata.keys())
        
def rdkcor(surveylist,options):

    kcordict = {}
    for survey in surveylist:
        kcorfile = options.__dict__['%s_kcorfile'%survey]
        subsurveys = options.__dict__['%s_subsurveylist'%survey].split(',')
        kcorfile = os.path.expandvars(kcorfile)
        if not os.path.exists(kcorfile):
            log.info('kcor file %s does not exist.   Checking %s/kcor'%(kcorfile,data_rootdir))
            kcorfile = '%s/kcor/%s'%(data_rootdir,kcorfile)
        if not os.path.exists(kcorfile):
            raise RuntimeError('kcor file %s does not exist'%kcorfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                hdu = fits.open(kcorfile)
                zpoff = hdu[1].data
                snsed = hdu[2].data
                filtertrans = hdu[5].data
                primarysed = hdu[6].data
                hdu.close()
            except:
                raise RuntimeError('kcor file format is non-standard for kcor file %s'%kcorfile)

        for subsurvey in subsurveys:
            kcorkey = '%s(%s)'%(survey,subsurvey)
            if not subsurvey: kcorkey = survey[:]
            kcordict[kcorkey] = {}
            kcordict[kcorkey]['primarywave'] = np.array(primarysed['wavelength (A)'])
            kcordict[kcorkey]['snflux'] =  np.array(snsed['SN Flux (erg/s/cm^2/A)'])

            if 'AB' in primarysed.names:
                kcordict[kcorkey]['AB'] =  np.array(primarysed['AB'])
            if 'Vega' in primarysed.names:
                kcordict[kcorkey]['Vega'] =      np.array(primarysed['Vega'])
            if 'VEGA' in primarysed.names:
                kcordict[kcorkey]['Vega'] =      np.array(primarysed['VEGA'])
            if 'BD17' in primarysed.names:
                kcordict[kcorkey]['BD17'] =      np.array(primarysed['BD17'])
            for filt in zpoff['Filter Name']:
                #log.warning('Using only the last character of kcor-provided filter names')
                if not options.filters_use_lastchar_only:
                    internalfiltname=filt[:] #[-1]
                else:
                    internalfiltname=filt[-1]

                kcordict[kcorkey][internalfiltname] = {}
                kcordict[kcorkey][internalfiltname]['filtwave'] = np.array(filtertrans['wavelength (A)'])
                kcordict[kcorkey][internalfiltname]['fullname'] = filt #.split('/')[0].replace(' ','')
                kcordict[kcorkey][internalfiltname]['filttrans'] = np.array(filtertrans[filt])

                lambdaeff = np.sum(kcordict[kcorkey][internalfiltname]['filtwave']*filtertrans[filt])/np.sum(filtertrans[filt])
                kcordict[kcorkey][internalfiltname]['lambdaeff'] = lambdaeff
                kcordict[kcorkey][internalfiltname]['magsys'] = \
                    zpoff['Primary Name'][zpoff['Filter Name'] == filt][0]
                kcordict[kcorkey][internalfiltname]['primarymag'] = \
                    zpoff['Primary Mag'][zpoff['Filter Name'] == filt][0] - zpoff['ZPoff(SNpot)'][zpoff['Filter Name'] == filt][0]
                kcordict[kcorkey][internalfiltname]['zpoff'] = \
                    zpoff['ZPoff(Primary)'][zpoff['Filter Name'] == filt][0] - zpoff['ZPoff(SNpot)'][zpoff['Filter Name'] == filt][0]
    if (options.calibrationshiftfile):
        log.info('Calibration shift file provided, applying offsets:')
        #Calibration dictionary:
        with open(options.calibrationshiftfile) as file:
            for line in file:
                log.info(f'Applying shift: {line}')
                try:
                    line=line.split('#')[0].split()
                    if len(line)==0:
                        continue
                    shifttype,survey,filter,shift=line
                    shift=float(shift)
                    #filter=filter[-1]#filter=filter[filter.index('/')+1:]
                    if not options.calib_survey_ignore:
                        if shifttype=='MAGSHIFT':
                            kcordict[survey][filter]['zpoff'] +=shift
                            kcordict[survey][filter]['primarymag']+=shift
                        elif shifttype=='LAMSHIFT' or shifttype=='WAVESHIFT':
                            kcordict[survey][filter]['filtwave']+=shift
                            kcordict[survey][filter]['lambdaeff']+=shift
                        else:
                            raise ValueError(f'Invalid calibration shift: {shifttype}')
                    else:
                        has_filter = False
                        for survey_forfilt in kcordict.keys():
                            if filter in kcordict[survey_forfilt].keys():
                                if shifttype=='MAGSHIFT':
                                    kcordict[survey_forfilt][filter]['zpoff'] +=shift
                                    kcordict[survey_forfilt][filter]['primarymag']+=shift
                                    has_filter = True
                                elif shifttype=='LAMSHIFT' or shifttype=='WAVESHIFT':
                                    kcordict[survey_forfilt][filter]['filtwave']+=shift
                                    kcordict[survey_forfilt][filter]['lambdaeff']+=shift
                                    has_filter = True
                                else:
                                    raise ValueError(f'Invalid calibration shift: {shifttype}')
                            if not has_filter:
                                raise ValueError(f'could not find filter {filter} in any kcor files')
                except Exception as e:
                    log.critical(f'Could not apply calibration offset \"{line[:-1]}\"')
                    raise e
                log.info('Calibration offsets applied')
    else:
        log.info('No calibration shift file provided, continuing')
    primarywave,primarysed = np.genfromtxt('%s/flatnu.dat'%init_rootdir,unpack=True)
        
    kcordict['default'] = {}
    initBfilt = '%s/Bessell90_B.dat'%init_rootdir
    filtwave,filttp = np.genfromtxt(initBfilt,unpack=True)
    kcordict['default']['Bwave'] = filtwave
    kcordict['default']['Btp'] = filttp
        
    initVfilt = '%s/Bessell90_V.dat'%init_rootdir
    filtwave,filttp = np.genfromtxt(initVfilt,unpack=True)
    kcordict['default']['Vwave'] = filtwave
    kcordict['default']['Vtp'] = filttp
        
    kcordict['default']['AB']=primarysed
    kcordict['default']['primarywave']=primarywave
    return kcordict


        
def rdAllData(snlists,estimate_tpk,
              dospec=False,peakmjdlist=None,
              waverange=[2000,9200],binspecres=None,snparlist=None,specrecallist=None,maxsn=None):
    datadict = {}
    if peakmjdlist:
        pksnid,pkmjd = np.loadtxt(peakmjdlist,unpack=True,dtype=str,usecols=[0,1])
        pkmjd = pkmjd.astype('float')
        pkmjddict={key:val for key,val in zip(pksnid,pkmjd)}
    else: pkmjd,pksnid,pkmjddict=[],[],{}
    if snparlist:
        snpar = at.Table.read(snparlist,format='ascii')
        snpar['SNID'] = snpar['SNID'].astype(str)
    else: snpar = None
    if specrecallist:
        src = at.Table.read(specrecallist,format='ascii')
        src['SNID'] = src['SNID'].astype(str)
    
    nsnperlist = []
    for snlist in snlists.split(','):

        snlist = os.path.expandvars(snlist)
        if not os.path.exists(snlist):
            log.info('SN list file %s does not exist.       Checking %s/trainingdata/%s'%(snlist,data_rootdir,snlist))
            snlist = '%s/trainingdata/%s'%(data_rootdir,snlist)
        if not os.path.exists(snlist):
            raise RuntimeError('SN list file %s does not exist'%snlist)
        snfiles = np.genfromtxt(snlist,dtype='str')
        snfiles = np.atleast_1d(snfiles)

        nsnperlist += [len(snfiles)]
    nsnperlist=np.array(nsnperlist)
    skipcount = 0
    rdstart = time()
    #If there is a maximum number of SNe to be taken in total, take an equal number from each snlist
    if maxsn is not None: maxcount = nsnperlist*maxsn/nsnperlist.sum()
    else: maxcount = [np.inf]*len(snlists.split(','))

    #Check whether to add the supernova to a dictionary of results; if not return False, otherwise do so and return True 
    def processsupernovaobject(outputdict,sn,maxnum,n_specrecal):

            if 'FLT' not in sn.__dict__.keys() and \
               'BAND' in sn.__dict__.keys():
                sn.FLT = sn.BAND
                    
            sn.SNID=str(sn.SNID)

            if sn.SNID in datadict: duplicatesurvey=datadict[sn.SNID].survey
            elif sn.SNID in outputdict: duplicatesurvey=outputdict[sn.SNID].survey
            else: duplicatesurvey=None
            if not duplicatesurvey is None:
                log.warning(f'SNID {sn.SNID} is a duplicate! Keeping version from survey {duplicatesurvey}, discarding version from survey {sn.SURVEY}')
                return False

            try:
                saltformattedsn=SALTtrainingSN(
                    sn,estimate_tpk=estimate_tpk,
                    pkmjddict=pkmjddict,snpar=snpar,
                    n_specrecal=n_specrecal,
                    binspecres=binspecres)
            except SNDataReadError as e:
                log.warning(e.args[0])
                return False
            if len(saltformattedsn.specdata) == 0:
                log.debug(f'SN {sn.SNID} has no supernova spectra')
            outputdict[saltformattedsn.snid]=saltformattedsn
            if len(outputdict)      >= maxnum:
                raise BreakLoopException('Maximum number of SNe read in')

            return True
            

    for snlist,maxct in zip(snlists.split(','),maxcount):
        tsn = time()
        snlist = os.path.expandvars(snlist)
        snfiles = np.genfromtxt(snlist,dtype='str')
        snfiles = np.atleast_1d(snfiles)
        snreadinfromlist={}

        try:
            for f in snfiles:
                if '/' not in f:
                    f = os.path.join(os.path.dirname(snlist),f)

                #If this is a fits file, read the list of snids and read them out one at a time
                if f.lower().endswith('.fits') or f.lower().endswith('.fits.gz'):

                    if f.lower().endswith('.fits') and not os.path.exists(f) and os.path.exists('{}.gz'.format(f)):
                        f = '{}.gz'.format(f)
                    # get list of SNIDs
                    hdata = fits.getdata( f, ext=1 )
                    survey = fits.getval( f, 'SURVEY')
                    Nsn = fits.getval( f, 'NAXIS2', ext=1 )
                    snidlist = np.array([ int( hdata[isn]['SNID'] ) for isn in range(Nsn) ])
                    if os.path.exists(f.replace('_HEAD.FITS','_SPEC.FITS')):
                        specfitsfile = f.replace('_HEAD.FITS','_SPEC.FITS')
                    else: specfitsfile = None

                    for snid in snidlist:

                        sn = snana.SuperNova(
                            snid=snid,headfitsfile=f,photfitsfile=f.replace('_HEAD.FITS','_PHOT.FITS'),
                            specfitsfile=specfitsfile,readspec=dospec)
                        if 'SUBSURVEY' in sn.__dict__.keys() and not (len(np.unique(sn.SUBSURVEY))==1 and survey.strip()==np.unique(sn.SUBSURVEY)[0].strip()) \
                        and sn.SUBSURVEY.strip() != '':
                            sn.SURVEY = f"{survey}({sn.SUBSURVEY})"
                        else:
                            sn.SURVEY = survey
                        if specrecallist:
                            n_specrecal = src[src['SNID'] == snid]
                        else:
                            n_specrecal = None
                        skipcount+=not processsupernovaobject(snreadinfromlist,sn,maxct,n_specrecal)
                else:
                    if '/' not in f:
                        f = '%s/%s'%(os.path.dirname(snlist),f)
                    sn = snana.SuperNova(f,readspec=dospec)
                    if specrecallist:
                        n_specrecal = src[src['SNID'] == sn.SNID]
                    else:
                        n_specrecal = None
                        
                    skipcount+=not processsupernovaobject(snreadinfromlist,sn,maxct,n_specrecal)
        except BreakLoopException:
            pass

        datadict.update(snreadinfromlist)
    log.info(f'read in {len(datadict)} SNe, {skipcount} SNe were not read')
    log.info('reading data files took %.1f'%(time()-rdstart))
    if not len(datadict.keys()):
        raise RuntimeError('no light curve data to train on!!')

    return datadict
        
