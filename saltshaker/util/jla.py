from __future__ import print_function

import os
import time
import numpy as np
import glob
from astropy.time import Time

class SuperNova( object ) : 
	""" object class for a single SN extracted from JLA lc-snname.list file
	"""
	def __init__( self, datfile=None, verbose=False) : 
		if not (datfile) : 
			if verbose:	 print("No datfile provided. Returning an empty SuperNova object.")
		if verbose: print("Reading in data from light curve file %s"%(datfile))
		self.readdatfile( datfile )

	@property
	def name(self):
		if 'SN' in self.__dict__ :
			return( self.SN )
		else : 
			return( '' )
	@property
	def z(self):
		if 'Z_HELIO' in self.__dict__ :
			return( self.Z_HELIO )
		elif 'Redshift' in self.__dict__ :
			return(self.Redshift)
		else : 
			return( '' )

	@property
	def zerr(self):
		if 'Redshift_err' in self.__dict__ :
			return( self.Redshift_err )
		else : 
			return( '' )

	@property
	def ra(self):
		if 'RA' in self.__dict__ :
			return( self.RA )
		else : 
			return( '' )

	@property
	def dec(self):
		if 'DEC' in self.__dict__ :
			return( self.DEC )
		else : 
			return( '' )

	@property
	def cov_mat_file(self):
		if 'COVMAT' in self.__dict__ :
			return( self.COVMAT )
		else : 
			return( '' )

	@property
	def survey(self):
		if 'SURVEY' in self.__dict__ :
			return( self.SURVEY )
		else : 
			return( '' )

	@property
	def mw_ebv(self):
		if 'MWEBV' in self.__dict__ :
			return( self.MWEBV )
		else : 
			return( '' )

	@property
	def mjdpk(self):
		if 'DayMax' in self.__dict__ :
			return( self.DayMax )
		else : 
			return( '' )

	@property
	def nobs(self):
		return( len(self.Mjd) )

	def readdatfile(self, datfile ):
		""" read the light curve data from a JLA file.
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
		colnames=['Mjd','Flux','Fluxerr','ZP','Camera','Filter','MagSys']
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
				self.__dict__['Mjd'].append( str2num(obsdat[0]) )
				self.__dict__['Flux'].append( str2num(obsdat[1]) )
				self.__dict__['Fluxerr'].append( str2num(obsdat[2]) )
				self.__dict__['ZP'].append( str2num(obsdat[3]) )
				dcolon = obsdat[4].find('::')
				self.__dict__['Camera'].append( str2num(obsdat[4][:dcolon].strip()) )
				self.__dict__['Filter'].append( str2num(obsdat[4][dcolon+2:].strip()) )
				self.__dict__['MagSys'].append( str2num(obsdat[5]) )
		for col in colnames : 
			self.__dict__[col] = array( self.__dict__[col] )
		return( None )

	def writesnanafile(self, datfile,verbose=False, **kwarg ):
		""" function to write jla data in snana format. Use as
		sn_lc=jla.SuperNova(full_path_of_jla_data/lc-sn_name.list)
		sn_lc.writesnanafile(full_path_of_jla_data/lc-sn_name.list)
		"""
		from numpy import array,log10,unique,where,absolute
		if verbose:	 print("Writing data from light curve file %s to snana format"%(datfile))
		fout = open(datfile+'.snana.dat','w')
		list_snana_headers=['SURVEY','SNID','RA','DEC','MWEBV','REDSHIFT_HELIO','MJDPK']
		list_jla_headers=['SURVEY','SN','RA','DEC','MWEBV','Z_HELIO','DayMax']
		for key in list_jla_headers : 
			if key in self.__dict__ :
				icol = list_jla_headers.index(key)
				if key == 'Z_HELIO' :
					print('%s: %s +- %s'%(list_snana_headers[icol],str(self.__dict__[key]),str(self.__dict__['Redshift_err'])),file=fout)
				else :
					print('%s: %s'%(list_snana_headers[icol],str(self.__dict__[key])),file=fout)				
		print('\nNOBS: %i'%len(self.Mjd),file=fout)
		print('NVAR: 6',file=fout)
		print('VARLIST:  MJD	FLT FIELD	FLUXCAL	  FLUXCALERR ZPT\n',file=fout)
		for i in range(self.nobs):
			Factor_275=(27.5-self.ZP[i])/(-2.5)
			flux_275=self.Flux[i]/(10**Factor_275)
			fluxerr_275=absolute(1.0/10**Factor_275)*self.Fluxerr[i]
			print('OBS: %9.3f  %s  %s %8.7f %8.7f %s'%(
					self.Mjd[i], self.Filter[i], 'NULL', flux_275, 
					fluxerr_275,'27.5' ),file=fout)
		print('END_PHOTOMETRY:\n',file=fout)
		print('# =============================================',file=fout)
		self.datfile = os.path.abspath(datfile)
		folder_data=datfile[0:datfile.rfind('/')]+'/'
		list_file_spec=glob.glob(folder_data+'spectrum*'+self.__dict__['SN']+'*.list')
		print('\nNSPECTRA: %i \n'%len(list_file_spec),file=fout)
		print('\nNVAR_SPEC: 5',file=fout)
		print('VARNAMES_SPEC: LAMMIN LAMMAX  FLAM  FLAMERR DQ\n',file=fout)
		counter=0
		for specfile in sorted(list_file_spec):
			counter=counter+1
			sn_spectrum=SuperNovaSpectrum(specfile,verbose=False)
			print('SPECTRUM_ID: %i'%counter,file=fout)
			print('SPECTRUM_MJD: %9.2f'%sn_spectrum.mjdspec,file=fout)
			resolution=sn_spectrum.WAVE[1]-sn_spectrum.WAVE[0]
			for wl,fl,flerr,dq in zip(sn_spectrum.WAVE,sn_spectrum.FLUX,sn_spectrum.FLUXERR,sn_spectrum.VALID):
				wl_l=wl-resolution/2.0
				wl_u=wl+resolution/2.0
				print('SPEC: %9.2f %9.2f %9.5e %9.5e %i'%(wl_l,wl_u,fl,flerr,dq),file=fout)
			print('SPECTRUM_END:\n',file=fout)	
		fout.close()

		return( None )
	
class SuperNovaSpectrum( object ) :
	""" object class for a single SN spectrum from spectrum-sname-#.list file
	"""
	def __init__( self, datfile=None, verbose=False) : 
		if not (datfile) : 
			if verbose:	 print("No spec file provided. Returning an empty SuperNovaSpectrum object.")
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
				self.__dict__['VALID'].append( str2num(obsdat[3]) )
		for col in colnames : 
			self.__dict__[col] = array( self.__dict__[col] )
		return( None )

def str2num(s) :
	try: return int(s)
	except ValueError:
		try: return float(s)
		except ValueError: return( s )
