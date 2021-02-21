#!/usr/bin/env python
# D. Jones - 5/14/14
# diffimmagstats.py --cmpfile=/datascope/ps1sn1/data/v10.0/GPC1v3/eventsv1/workspace/PSc360052/tmpl/g/PSc360052.md04s047.g.stack_44.sw.icmp --psffile=/datascope/ps1sn1/data/v10.0/GPC1v3/eventsv1/workspace/PSc360052/g/PSc360052.md04s047.g.ut091126f.648816_44.sw.dao.psf.fits --diffim=/datascope/ps1sn1/data/v10.0/GPC1v3/eventsv1/workspace/PSc360052_tmpl/g/PSc360052.md04s047.g.ut091126f.648816_44_md04s047.g.stack_44.diff.fits
"""Calculate increase in uncertainty due
to bright host galaxies

Usage: diffimmagstats.py --cmpfile=cmpfile --psffile=psffile --diffim=diffimfile

"""
import glob
import os
import numpy as np
import astropy.io.fits as pyfits

class txtobj:
	def __init__(self,filename,allstring=False,
				 cmpheader=False,sexheader=False,
				 useloadtxt=True,fitresheader=False,
				 delimiter=' ',skiprows=0,tabsep=False,
				 rowprfx='SN'):
		if cmpheader:
			hdu = pyfits.open(filename)
			hdu.verify('silentfix')
			hdr = hdu[0].header
			skiprows=1
		#if fitresheader: skiprows=6

		coldefs = np.array([])
		if cmpheader:
			for k,v in zip(hdr.keys(),hdr.values()):
				if 'COLTBL' in k and k != 'NCOLTBL':
					coldefs = np.append(coldefs,v)
		elif sexheader:
			fin = open(filename,'r')
			lines = fin.readlines()
			for l in lines:
				if l.startswith('#'):
					coldefs = np.append(coldefs,l.replace('#','').split()[1])
		elif fitresheader:
			self.rdfitres(filename,rowprfx=rowprfx)
			return
		else:
			fin = open(filename,'r')
			lines = fin.readlines()
			if not tabsep:
				if delimiter == ' ':
					coldefs = np.array(lines[0].replace('#','').split())
					#coldefs = coldefs[np.where(coldefs != '#')]
				else:
					coldefs = np.array(lines[0].replace('#','').split(delimiter))[np.where(np.array(lines[0].split(delimiter)) != '')[0]]
					#coldefs = coldefs[np.where(coldefs != '#')]
			else:
				l = lines[0].replace('#','').replace('\n','')
				coldefs = np.array(l.replace('#','').split('\t')[l.split('\t') != ''])
				#coldefs = coldefs[np.where(coldefs != '#')]

		for i in range(len(coldefs)):
			coldefs[i] = coldefs[i].replace('\n','').replace('\t','').replace(' ','')
			if coldefs[i]:
				self.__dict__[coldefs[i].replace('\r','')] = np.array([])

		self.filename = np.array([])
		if useloadtxt:
			for c,i in zip(coldefs,range(len(coldefs))):
				c = c.replace('\n','')
				if c:
					if not delimiter or delimiter == ' ':
						self.__dict__[c] = np.atleast_1d(np.genfromtxt(filename,unpack=True,usecols=[i],dtype='str',skip_header=skiprows))
						try: self.__dict__[c] = self.__dict__[c].astype(float)
						except: continue
							
					else:
						self.__dict__[c] = np.atleast_1d(np.genfromtxt(filename,unpack=True,usecols=[i],dtype='str',delimiter=delimiter,skip_header=skiprows))
						try: self.__dict__[c] = self.__dict__[c].astype(float)
						except: continue
			self.filename = np.array([filename]*len(self.__dict__[c]))

		else:
			fin = open(filename,'r')
			count = 0
			for line in fin:
				if count >= 1 and not line.startswith('#'):
					entries = line.split()
					for e,c in zip(entries,coldefs):
						e = e.replace('\n','')
						c = c.replace('\n','')
						if not allstring:
							try:
								self.__dict__[c] = np.append(self.__dict__[c],float(e))
							except:
								self.__dict__[c] = np.append(self.__dict__[c],e)
						else:
							self.__dict__[c] = np.append(self.__dict__[c],e)
						self.filename = np.append(self.filename,filename)
				else: count += 1
			fin.close()

	def rdfitres(self,filename,rowprfx='SN'):
		import numpy as np
		fin = open(filename,'r')
		lines = fin.readlines()
		for l in lines:
			if l.startswith('VARNAMES:'):
				l = l.replace('\n','')
				coldefs = l.split()
				break
				
		with open(filename) as f:
			reader = [x.split() for x in f if x.startswith('%s:'%rowprfx)]

		i = 0
		for column in zip(*reader):
			try:
				self.__dict__[coldefs[i]] = np.array(column[:]).astype(float)
			except:
				self.__dict__[coldefs[i]] = np.array(column[:])
			i += 1
			
	def addcol(self,col,data):
		self.__dict__[col] = data
	def cut_inrange(self,col,minval,maxval,rows=[]):
		if not len(rows):
			rows = np.where((self.__dict__[col] > minval) &
							(self.__dict__[col] < maxval))[0]
			return(rows)
		else:
			rows2 = np.where((self.__dict__[col][rows] > minval) &
							(self.__dict__[col][rows] < maxval))[0]
			return(rows[rows2])

	def cut_byVar(self,varname,val_or_range):
		if isinstance(val_or_range,(tuple,list,np.ndarray)):
			iCut = np.where(np.logical_and(self.__dict__[varname]>=val_or_range[0],self.__dict__[varname]<=val_or_range[1]))[0]
			
		else:
			iCut = self.__dict__[varname]==val_or_range

		for k in self.__dict__.keys():
			self.__dict__[k] = self.__dict__[k][iCut]
			
	def appendfile(self,filename,usegenfromtxt=False):
		if usegenfromtxt:
			fin = open(filename,'r')
			for line in fin:
				if line.startswith('#'):
					coldefs = line.split('#')[1].split('\n')[0].split()
					break
			fin.close()
			for c,i in zip(coldefs,range(len(coldefs))):
				try:
					self.__dict__[c] = np.concatenate((self.__dict__[c],np.genfromtxt(filename,unpack=True,usecols=[i])))
				except:
					self.__dict__[c] = np.concatenate((self.__dict__[c],np.genfromtxt(filename,unpack=True,
																				   usecols=[i],dtype='str')))
			self.filename = np.append(self.filename,np.array([filename]*len(np.genfromtxt(filename,unpack=True,usecols=[i],dtype='str'))))
			
			return()
		fin = open(filename,'r')
		for line in fin:
			if line.startswith('#'):
				coldefs = line.split('#')[1].split('\n')[0].split()
			else:
				entries = line.split()
				for e,c in zip(entries,coldefs):
					e = e.replace('\n','')
					c = c.replace('\n','')
					try:
						self.__dict__[c] = np.append(self.__dict__[c],float(e))
					except:
						self.__dict__[c] = np.append(self.__dict__[c],e)
				self.filename = np.append(self.filename,filename)
