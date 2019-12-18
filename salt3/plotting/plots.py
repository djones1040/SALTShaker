from astropy.cosmology import Planck15 as cosmo
import numpy as np
import matplotlib.pyplot as plt
import os,scipy,optparse
import scipy.stats
from copy import deepcopy

from .txtobj import txtobj
from .getmu import *
from .util import *
from .ovdatamc import *

usagestring="""
	ovdatamc.py <DataFitresFile> <SimFitresFile>  <varName1:varName2:varName3....>  [--cutwin NN_ITYPE 1 1 --cutwin x1 -3 3]

	Given a FITRES file for both data and an SNANA simulation, 
	ovdatamc.py creates histograms that compare fit parameters or other 
	variables between the two.  If distance moduli/residuals are not 
	included in the fitres file, specify MU/MURES as the varName and 
	they will be computed with standard SALT2 nuisance parameters.  To 
	specify multiple variable names, use colons to separate them.

	use -h/--help for full options list
	"""

def getObj(fitresfile, fitresheader = True, makeCuts = True, version=None):
	fr = txtobj(fitresfile,fitresheader = fitresheader)
	if makeCuts:
		fr = mkcuts(fr)
	fr.filename=os.path.splitext(os.path.basename(fitresfile))[0]
	fr.version=version
	return(fr)

def calcMu(fr,alpha=0.14,beta=3.1,M=-19.36):
	fr.MU = fr.mB + alpha*fr.x1 - beta*fr.c - M
	fr.MUERR = np.sqrt(fr.mBERR**2 + alpha**2.*fr.x1ERR**2. + beta**2.*fr.cERR**2)
	return(fr)

def plot_hubble_diff(fr1,fr2,multisurvey=False,nbins=6):
	if multisurvey:
		surveys=np.unique(fr1.FIELD)
		col_dict={s:np.random.rand(3,) for s in surveys}
		
	else:
		surveys1=[None]
		col_dict1={None:'b'}
	ax = None
	for survey in surveys:
		if survey is None:
			survey='ALL'
			zdata1=fr1.zCMB
			mudata1=fr1.MU
			muerrdata1=fr1.MUERR
			zdata2=fr2.zCMB
			mudata2=fr2.MU
			muerrdata2=fr2.MUERR
		else:
			zdata1=fr1.zCMB[fr1.FIELD==survey]
			mudata1=fr1.MU[fr1.FIELD==survey]
			muerrdata1=fr1.MUERR[fr1.FIELD==survey]
			zdata2=fr2.zCMB[fr2.FIELD==survey]
			mudata2=fr2.MU[fr2.FIELD==survey]
			muerrdata2=fr2.MUERR[fr2.FIELD==survey]

		stats1,edges1,bins1 = scipy.stats.binned_statistic(zdata1,mudata1,'mean',bins=np.arange(np.min(zdata1),np.max(zdata1)+.001,.05))
		stats2,edges2,bins2 = scipy.stats.binned_statistic(zdata2,mudata2,'mean',bins=edges1)
		stat_err1,edges2,bins2 = scipy.stats.binned_statistic(zdata1,mudata1,'std',bins=edges1)
		stat_err2,edges2,bins2 = scipy.stats.binned_statistic(zdata2,mudata2,'std',bins=edges1)
		bin_data1=[]
		bin_data2=[]

		final_inds=[]
		for i in range(1,len(edges1)):
			inds1=np.where(bins1==i)[0]
			inds2=np.where(bins2==i)[0]
			if len(inds1)==0 or len(inds2)==0:
				continue
			final_inds.append(i-1)
			
			stat_err1[i-1]/=np.sqrt(len(inds1))
			stat_err2[i-1]/=np.sqrt(len(inds2))
			bin_data1.append(np.average(mudata1[inds1],weights=1./muerrdata1[inds1]))
			bin_data2.append(np.average(mudata2[inds2],weights=1./muerrdata2[inds2]))
		bin_data1=np.array(bin_data1)
		bin_data2=np.array(bin_data2)

		if ax is None:
			ax=plot('errorbar',[(edges1[i]+edges1[i+1])/2 for i in final_inds],bin_data1-bin_data2,yerr=np.sqrt(stat_err1[final_inds]**2+stat_err2[final_inds]**2),
				x_lab=r'$z_{\rm{CMB}}$',y_lab=r'$\mu$ Residual',fmt='o',color=col_dict[survey],label=survey)
			
		else:
			ax.errorbar([(edges1[i]+edges1[i+1])/2 for i in final_inds],bin_data1-bin_data2,yerr=np.sqrt(stat_err1[final_inds]**2+stat_err2[final_inds]**2),
				fmt='o',color=col_dict[survey],label=survey)
				
		
		
	ax.legend(fontsize=16)
	lims=ax.get_xlim()
	ax.plot(lims,[0,0],'k--',linewidth=3)
	
		
	if not os.path.exists('figures'):
		os.makedirs('figures')
	if fr1.version is not None:
		fname1=fr1.version
	else:
		fname1=fr1.filename
	if fr2.version is not None:
		fname2=fr2.version
	else:
		fname2=fr2.filename
	if os.path.exists(os.path.join('figures',fname1+'-'+fname2+'_hubble_diagram.pdf')):
		ext=1
		while os.path.exists(os.path.join('figures',fname1+'-'+fname2+'_hubble_diagram_'+str(ext)+'.pdf')):
			ext+=1
		outname=os.path.join('figures',fname1+'-'+fname2+'_hubble_diagram_'+str(ext)+'.pdf')
	else:
		outname=os.path.join('figures',fname1+'-'+fname2+'_hubble_diagram.pdf')
	plt.tight_layout()
	plt.savefig(outname,format='pdf')

	plt.clf()

def plot_hubble(fr,binned=True,multisurvey=False,nbins=6):
	if multisurvey:
		surveys=np.unique(fr.FIELD)
		col_dict={s:np.random.rand(3,) for s in surveys}
	else:
		surveys=[None]
		col_dict={None:'b'}
	ax = None
	for survey in surveys:
		if survey is None:
			survey='ALL'
			zdata=fr.zCMB
			mudata=fr.MU
			muerrdata=fr.MUERR
		else:
			zdata=fr.zCMB[fr.FIELD==survey]
			mudata=fr.MU[fr.FIELD==survey]
			muerrdata=fr.MUERR[fr.FIELD==survey]

		if binned:
			stats,edges,bins = scipy.stats.binned_statistic(zdata,mudata,'mean',bins=np.arange(np.min(zdata),np.max(zdata)+.001,.05))
			stat_err,edges2,bins2 = scipy.stats.binned_statistic(zdata,mudata,'std',bins=edges)
			bin_data=[]
			final_inds=[]
			for i in range(1,len(edges)):
				inds=np.where(bins==i)[0]
				if len(inds)==0:
					continue
				final_inds.append(i-1)
				stat_err[i-1]/=np.sqrt(len(inds))
				bin_data.append(np.average(mudata[inds],weights=1./muerrdata[inds]))
			bin_data=np.array(bin_data)

			if ax is None:
				ax=plot('errorbar',[(edges[i]+edges[i+1])/2 for i in final_inds],bin_data,yerr=stat_err[final_inds],y_lab=r'$\mu$',fmt='o',color=col_dict[survey],label=survey)
				ax,ax2=split_plot(ax,'errorbar',[(edges[i]+edges[i+1])/2 for i in final_inds],
					y=bin_data-cosmo.distmod([(edges[i]+edges[i+1])/2 for i in final_inds]).value,yerr=stat_err[final_inds],x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o',color=col_dict[survey])
			else:
				ax.errorbar([(edges[i]+edges[i+1])/2 for i in final_inds],bin_data,yerr=stat_err[final_inds],fmt='o',color=col_dict[survey],label=survey)
				ax2.errorbar([(edges[i]+edges[i+1])/2 for i in final_inds],bin_data-cosmo.distmod([(edges[i]+edges[i+1])/2 for i in final_inds]).value,yerr=stat_err[final_inds],
					fmt='o',color=col_dict[survey])
			
		else:
			ax=plot('errorbar',zdata,y=mudata,yerr=muerrdata,y_lab=r'$\mu$',fmt='o',color=col_dict[survey],label=survey)
			ax,ax2=split_plot(ax,'errorbar',zdata,y=mudata-cosmo.distmod(zdata).value,yerr=muerrdata,x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o',color=col_dict[survey])
		zinterp=np.arange(np.min(zdata),np.max(zdata),.01)
		ax.plot(zinterp,cosmo.distmod(zinterp).value,color='k',linewidth=3)
	ax.legend(fontsize=16)
	lims=ax.get_xlim()
	ax2.plot(lims,[0,0],'k--',linewidth=3)
	
		
	if not os.path.exists('figures'):
		os.makedirs('figures')
	if fr.version is not None:
		fname=fr.version
	else:
		fname=fr.filename
	if os.path.exists(os.path.join('figures',fname+'_hubble_diagram.pdf')):
		ext=1
		while os.path.exists(os.path.join('figures',fname+'_hubble_diagram_'+str(ext)+'.pdf')):
			ext+=1
		outname=os.path.join('figures',fname+'_hubble_diagram_'+str(ext)+'.pdf')
	else:
		outname=os.path.join('figures',fname+'_hubble_diagram.pdf')
	plt.tight_layout()
	plt.savefig(outname,format='pdf')

	plt.clf()

def plot_zdepend(datafile,simfile,fitvars=['x1','c'],survey=None,zstep=.05,version='',alpha=1,beta=1,**kwargs):
	data=deepcopy(datafile)
	sim=deepcopy(simfile)
	#data = txtobj_abv(datafile)
	#sim = txtobj_abv(simfile)

	if survey is not None:
		data.cut_byVar('FIELD',survey)
		sim.cut_byVar('FIELD',survey)
	else:
		survey=''
	data.version=version
	sim.version=version

	ax = None
	for var in fitvars:
		if var=='x1':
			const=alpha
		elif var=='c':
			const=beta
		else:
			const=1
		stats,edges,bins = scipy.stats.binned_statistic(data.zCMB,data.__dict__[var],'mean',bins=np.arange(np.min(data.zCMB),np.max(data.zCMB)+.001,zstep))
		stats_err,edges_err,bins_err = scipy.stats.binned_statistic(data.zCMB,data.__dict__[var],'std',bins=edges)
		stats1,edges1,bins1 = scipy.stats.binned_statistic(sim.zCMB,sim.__dict__[var],'mean',bins=edges)
		stats1_err,edges1_err,bins1_err = scipy.stats.binned_statistic(sim.zCMB,sim.__dict__[var],'std',bins=edges)
		bin_data1=[]
		bin_data2=[]

		final_inds=[]
		for i in range(1,len(edges)):
			inds1=np.where(bins==i)[0]
			inds2=np.where(bins1==i)[0]
			if len(inds1)==0 or len(inds2)==0:
				continue
			final_inds.append(i-1)
			
			stats_err[i-1]/=np.sqrt(len(inds1))
			stats1_err[i-1]/=np.sqrt(len(inds2))
			bin_data1.append(np.average(data.__dict__[var][inds1],weights=1./data.__dict__[var+'ERR'][inds1]))
			bin_data2.append(np.average(sim.__dict__[var][inds2],weights=1./sim.__dict__[var+'ERR'][inds2]))
		bin_data1=np.array(bin_data1)
		bin_data2=np.array(bin_data2)
		if ax is None:
			ax=plot('errorbar',[(edges[i]+edges[i+1])/2 for i in final_inds],(bin_data1-bin_data2)*const,yerr=const*np.sqrt(stats_err[final_inds]**2+stats1_err[final_inds]**2),
				x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o',label=survey+'_%s'%var)
			
		else:
			ax.errorbar([(edges1[i]+edges1[i+1])/2 for i in final_inds],const*(bin_data1-bin_data2),yerr=const*np.sqrt(stats_err[final_inds]**2+stats1_err[final_inds]**2),
				fmt='o',label=survey+'_%s'%var)

	ax.legend(fontsize=16)
	lims=ax.get_xlim()
	ax.plot(lims,[0,0],'k--',linewidth=3)
	
		
	if not os.path.exists('figures'):
		os.makedirs('figures')
	if data.version is not None:
		fname1=data.version
	else:
		fname1=data.filename
	if survey is not None:
		fname1+='_'+survey
	if os.path.exists(os.path.join('figures',fname1+'_zdepend.pdf')):
		ext=1
		while os.path.exists(os.path.join('figures',fname1+'_zdepend'+str(ext)+'.pdf')):
			ext+=1
		outname=os.path.join('figures',fname1+'_zdepend'+str(ext)+'.pdf')
	else:
		outname=os.path.join('figures',fname1+'_zdepend.pdf')
	plt.tight_layout()
	plt.savefig(outname,format='pdf')

	plt.clf()



def plot_fits(simfile,datafile=None,fitvars=['x1','c'],version='',cuts={},xlimits=None,survey=None,**kwargs):
	
	ovhist_obj=ovhist()
	parser = ovhist_obj.add_options(usage=usagestring)
	options,  args = parser.parse_args()
	options.histvar = fitvars#['x1','c']

	ovhist_obj.options = options
	for k in kwargs.keys():
		ovhist_obj.options.__dict__[k]=kwargs.get(k)
	ovhist_obj.version=version
	if datafile is None:
		datafile=simfile
	
	data = txtobj_abv(datafile)
	sim = txtobj_abv(simfile)
	if survey is not None:
		data.cut_byVar('FIELD',survey)
		sim.cut_byVar('FIELD',survey)
	for cut in cuts.keys():
		data.cut_inrange(cut,cuts[cut][0],cuts[cut][1])
		sim.cut_inrange(cut,cuts[cut][0],cuts[cut][1])


	#sys.exit()
	# getting distance modulus is slow, so don't do it unless necessary
	getMU = False
	if len(ovhist_obj.options.cutwin):
		for cutopt in ovhist_obj.options.cutwin:
			if 'MU' in cutopt[0]: getMU = True
	for h in ovhist_obj.options.histvar:
		if 'MU' in h: getMU = True
			
	if 'MU' in ovhist_obj.options.histvar or getMU:
		if not 'MU' in data.__dict__:
			data.MU,data.MUERR = salt2mu(x1=data.x1,x1err=data.x1ERR,c=data.c,cerr=data.cERR,mb=data.mB,mberr=data.mBERR,
										 cov_x1_c=data.COV_x1_c,cov_x1_x0=data.COV_x1_x0,cov_c_x0=data.COV_c_x0,
										 alpha=ovhist_obj.options.alpha,beta=ovhist_obj.options.beta,
										 x0=data.x0,sigint=ovhist_obj.options.sigint,z=data.zHD,M=ovhist_obj.options.dataM)
			from astropy.cosmology import Planck13 as cosmo
			if not 'MURES' in data.__dict__:
				data.MURES = data.MU - cosmo.distmod(data.zHD).value
		if not 'MU' in sim.__dict__:
			sim.MU,sim.MUERR = salt2mu(x1=sim.x1,x1err=sim.x1ERR,c=sim.c,cerr=sim.cERR,mb=sim.mB,mberr=sim.mBERR,
									   cov_x1_c=sim.COV_x1_c,cov_x1_x0=sim.COV_x1_x0,cov_c_x0=sim.COV_c_x0,
									   alpha=ovhist_obj.options.alpha,beta=ovhist_obj.options.beta,
									   x0=sim.x0,sigint=ovhist_obj.options.sigint,z=sim.zHD,M=ovhist_obj.options.simM)
			from astropy.cosmology import Planck13 as cosmo
			if not 'MURES' in sim.__dict__:
				sim.MURES = sim.MU - cosmo.distmod(sim.zHD).value

	if ovhist_obj.options.scaleb4cuts:
		cols_CC = np.where((sim.SIM_TYPE_INDEX != 1))[0]
		cols_Ia = np.where((sim.SIM_TYPE_INDEX == 1))[0]
		lenCC = float(len(cols_CC))
		lenIa = float(len(cols_Ia))

	sim = ovhist_obj.mkcuts(sim,fitresfile=simfile)
	data = ovhist_obj.mkcuts(data,fitresfile=datafile)

	if ovhist_obj.options.journal:
		mf = factors(len(ovhist_obj.options.histvar))
		if ovhist_obj.options.nplots[0]: ysubplot = ovhist_obj.options.nplots[1]; xsubplot = ovhist_obj.options.nplots[0]
		else:
			ysubplot = mf[len(mf)/2]
			xsubplot = len(ovhist_obj.options.histvar)/ysubplot
		plt.rcParams['figure.figsize'] = (xsubplot*7,ysubplot*7)
		if not ovhist_obj.options.outfile:
			ovhist_obj.options.outfile = 'ovplot_%s.png'%("_".join(ovhist_obj.options.histvar))
	else:
		plt.rcParams['figure.figsize'] = (8.5,11)
		from matplotlib.backends.backend_pdf import PdfPages
		if not ovhist_obj.options.outfile:
			if not os.path.exists('figures'):
				os.makedirs('figures')
			if ovhist_obj.version is not None:
				fname=ovhist_obj.version
			else:
				fname=''
			if survey is not None:
				fname+='_'+survey
			
			ovhist_obj.options.outfile = os.path.join("figures",'ovplot_%s_%s.pdf'%(fname,"_".join(ovhist_obj.options.histvar)))
			if os.path.exists(ovhist_obj.options.outfile):
				ext=1
				while os.path.exists(os.path.join("figures",'ovplot_%s_%s_%i.pdf'%(fname,"_".join(ovhist_obj.options.histvar),ext))):
					ext+=1
				ovhist_obj.options.outfile=os.path.join("figures",'ovplot_%s_%s_%i.pdf'%(fname,"_".join(ovhist_obj.options.histvar),ext))
			
			
		if not os.path.exists(ovhist_obj.options.outfile) or ovhist_obj.options.clobber:
			pdf_pages = PdfPages(ovhist_obj.options.outfile)
		else:
			print('File %s exists!  Not clobbering...'%ovhist_obj.options.outfile)
			return(1)

	for histvar,i in zip(ovhist_obj.options.histvar,
						 np.arange(len(ovhist_obj.options.histvar))+1):
		if ovhist_obj.options.journal:
			ax = plt.subplot(ysubplot,xsubplot,i)
			import string
			ax.text(-0.1, 1.05, '%s)'%string.ascii_uppercase[i-1], transform=ax.transAxes, 
					 size=20, weight='bold')
		else:
			if i%3 == 1: fig = plt.figure()
			if i == 3: subnum = 3
			else: subnum = i%3
			if len(ovhist_obj.options.histvar) >= 3:
				ax = plt.subplot(3,1,subnum)
			else:
				ax = plt.subplot(len(ovhist_obj.options.histvar),1,subnum)
		if not ovhist_obj.options.journal:
			ax.set_xlabel(histvar,labelpad=0)
		else:
			try:
				if '$' in histvardict[histvar]:
					ax.set_xlabel(histvardict[histvar],fontsize=40)
				else:
					ax.set_xlabel(histvardict[histvar],fontsize=30)
			except KeyError:
				ax.set_xlabel(histvar)
			ax.set_ylabel('$N_{SNe}$',labelpad=0,fontsize=30)
			if 'vzCMB' in histvar: 
				ax.set_ylabel(histvardict[histvar],fontsize=30)
				ax.set_xlabel('$z_{CMB}$',fontsize=30)
			elif 'vmB' in histvar: 
				ax.set_ylabel(histvardict[histvar],fontsize=30)
				ax.set_xlabel('$m_{B}$',fontsize=30)

		if 'vzCMB' in histvar:
			ovhist_obj.plt2var(data,sim,ax,histvar)
			continue
		if 'vmB' in histvar:
			ovhist_obj.plt2mB(data,sim,ax,histvar)
			continue
			
			
		ovhist_obj.options.histmin,ovhist_obj.options.histmax = None,None
		if len(ovhist_obj.options.cutwin):
			for cutopt in ovhist_obj.options.cutwin:
				var,min,max = cutopt[0],cutopt[1],cutopt[2]; min,max = float(min),float(max)
			if var == histvar:
				ovhist_obj.options.histmin = min; ovhist_obj.options.histmax = max
		if not ovhist_obj.options.histmin:
			ovhist_obj.options.histmin = np.min(np.append(sim.__dict__[histvar],data.__dict__[histvar]))
			ovhist_obj.options.histmax = np.max(np.append(sim.__dict__[histvar],data.__dict__[histvar]))


		cols_CC = np.where((sim.SIM_TYPE_INDEX != 1) & 
						   (sim.__dict__[histvar] >= ovhist_obj.options.histmin) &
						   (sim.__dict__[histvar] <= ovhist_obj.options.histmax))[0]
		cols_Ia = np.where((sim.SIM_TYPE_INDEX == 1) & 
						   (sim.__dict__[histvar] >= ovhist_obj.options.histmin) &
						   (sim.__dict__[histvar] <= ovhist_obj.options.histmax))[0]
		if not ovhist_obj.options.scaleb4cuts:
			lenCC = float(len(cols_CC))
			lenIa = float(len(cols_Ia))
		
		# bins command options
		if ovhist_obj.options.bins[0] != None: ovhist_obj.options.nbins = ovhist_obj.options.bins[0]
		if ovhist_obj.options.bins[1] != None: ovhist_obj.options.histmin = ovhist_obj.options.bins[1]
		if ovhist_obj.options.bins[2] != None: ovhist_obj.options.histmax = ovhist_obj.options.bins[2]
		print(histvar,ovhist_obj.options.histmin,ovhist_obj.options.histmax)

		histint = (ovhist_obj.options.histmax - ovhist_obj.options.histmin)/ovhist_obj.options.nbins
		histlen = float(len(np.where((data.__dict__[histvar] > ovhist_obj.options.histmin) &
									 (data.__dict__[histvar] < ovhist_obj.options.histmax))[0]))
		n_nz = np.histogram(data.__dict__[histvar],bins=np.linspace(ovhist_obj.options.histmin,ovhist_obj.options.histmax,ovhist_obj.options.nbins))
		
		errl,erru = poisson_interval(n_nz[0])
		ax.plot(n_nz[1][:-1]+(n_nz[1][1]-n_nz[1][0])/2.,n_nz[0],'o',color='k',lw=2,label='data')
		ax.errorbar(n_nz[1][:-1]+(n_nz[1][1]-n_nz[1][0])/2.,n_nz[0],yerr=[n_nz[0]-errl,erru-n_nz[0]],color='k',fmt=' ',lw=2)
		import copy
		n_nz_chi2 = copy.deepcopy(n_nz)
		n_nz = np.histogram(sim.__dict__[histvar],bins=np.linspace(ovhist_obj.options.histmin,ovhist_obj.options.histmax,ovhist_obj.options.nbins))
		ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
				color='k',drawstyle='steps-mid',lw=4,label='All Sim. SNe',ls='--')
		chi2 = np.sum((n_nz[0]/float(lenIa+lenCC)*histlen-n_nz_chi2[0])**2./((erru-errl)/2.)**2.)/float(len(n_nz[0])-1)
		print('chi2 = %.3f for %s'%(chi2,histvar))

		n_nz = np.histogram(sim.__dict__[histvar][cols_CC],bins=np.linspace(ovhist_obj.options.histmin,ovhist_obj.options.histmax,ovhist_obj.options.nbins))
		ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
				color='b',drawstyle='steps-mid',lw=4,label='Sim. CC SNe',ls='-.')
		n_nz = np.histogram(sim.__dict__[histvar][cols_Ia],bins=np.linspace(ovhist_obj.options.histmin,ovhist_obj.options.histmax,ovhist_obj.options.nbins))
		ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
				color='r',drawstyle='steps-mid',lw=2,label='Sim. SNe Ia')

		if ovhist_obj.options.ylim[0] or ovhist_obj.options.ylim[1]: ax.set_ylim([ovhist_obj.options.ylim[0],ovhist_obj.options.ylim[1]])
		if ovhist_obj.options.ylog == 'all' or histvar in ovhist_obj.options.ylog:
			ax.set_yscale('log')
			if not ovhist_obj.options.ylim[0]: ax.set_ylim(bottom=0.5)

		print('Variable: %s'%histvar)
		print('NDATA: %i'%len(data.CID))
		print('MC Scale: %.1f'%(histlen/float(lenIa+lenCC)))
		if lenIa:
			print('N(CC Sim.)/N(Ia Sim.): %.3f'%(lenCC/float(lenIa)))
		else:
			print('N(CC Sim.)/N(Ia Sim.): inf')

		if xlimits is not None:
			ax.set_xlim(xlimits)
		if not ovhist_obj.options.journal and i%3 == 1:
			box = ax.get_position()
			ax.set_position([box.x0, box.y0,# + box.height * 0.15,
							 box.width, box.height * 0.85])
			# Put a legend below current axis
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6),
					  fancybox=True, ncol=2,numpoints=1)
		if ovhist_obj.options.journal and i == 2:
			ax.legend(loc='upper center', bbox_to_anchor=(0.625, 1.0),
					  fancybox=True,numpoints=1,prop={'size':23})
	

		if ovhist_obj.options.interact:
			plt.show()
		
		if not ovhist_obj.options.journal:
			if i%3 == 1: ovhist_obj.plottitle(ax)
			if not i%3:
				if not os.path.exists(ovhist_obj.options.outfile) or ovhist_obj.options.clobber:
					pdf_pages.savefig(fig)

	if not ovhist_obj.options.journal:
		if i%3:
			pdf_pages.savefig(fig)
		pdf_pages.close()
	else:
		if not ovhist_obj.options.outfile:
			outfile = 'ovplot_%s.png'%("_".join(ovhist_obj.options.histvar))
		else: outfile = ovhist_obj.options.outfile
		if not os.path.exists(outfile) or ovhist_obj.options.clobber:
			plt.savefig(outfile)
		else:
			print('File %s exists!  Not clobbering...'%outfile)

if __name__=='__main__':
	getObj('test')