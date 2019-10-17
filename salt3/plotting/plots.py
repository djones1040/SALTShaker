from astropy.cosmology import Planck15 as cosmo
import numpy as np
import matplotlib.pyplot as plt
import os,scipy
import scipy.stats


from .txtobj import txtobj
from .getmu import *
from .util import *
from .ovdatamc import *

def getObj(fitresfile, fitresheader = True, makeCuts = True):
	fr = txtobj(fitresfile,fitresheader = fitresheader)
	if makeCuts:
		fr = mkcuts(fr)
	fr.filename=os.path.splitext(os.path.basename(fitresfile))[0]
	return(fr)

def calcMu(fr,alpha=0.14,beta=3.1,M=-19.36):
	fr.MU = fr.mB + alpha*fr.x1 - beta*fr.c - M
	fr.MUERR = np.sqrt(fr.mBERR**2 + alpha**2.*fr.x1ERR**2. + beta**2.*fr.cERR**2)
	return(fr)

def plot_hubble(fr,binned=True):
	if binned:
		stats,edges,bins = scipy.stats.binned_statistic(fr.zCMB,fr.MU,'mean',bins=np.arange(np.min(fr.zCMB),np.max(fr.zCMB)+.001,.1))
		stat_err,edges2,bins2 = scipy.stats.binned_statistic(fr.zCMB,fr.MU,'std',bins=edges)
		bin_data=[]
		for i in range(1,len(edges)):
			inds=np.where(bins==i)
			bin_data.append(np.average(fr.MU[inds],weights=1./fr.MUERR[inds]))
		bin_data=np.array(bin_data)
		ax=plot('errorbar',[(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)],bin_data,yerr=stat_err,y_lab=r'$\mu$',fmt='o')
		ax,ax2=split_plot(ax,'errorbar',[(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)],
			y=bin_data-cosmo.distmod([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]).value,yerr=stat_err,x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o')
		lims=ax.get_xlim()
		ax2.plot(lims,[0,0],'k--',linewidth=3)
	else:
		ax=plot('errorbar',fr.zCMB,y=fr.MU,yerr=fr.MUERR,y_lab=r'$\mu$',fmt='o')
		ax,ax2=split_plot(ax,'errorbar',fr.zCMB,y=fr.MU-cosmo.distmod(fr.zCMB).value,yerr=fr.MUERR,x_lab=r'$z_{\rm{CMB}}$',y_lab='Residual',fmt='o')
		lims=ax.get_xlim()
		ax2.plot(lims,[0,0],'k--',linewidth=3)
	zinterp=np.arange(np.min(fr.zCMB),np.max(fr.zCMB),.01)
	ax.plot(zinterp,cosmo.distmod(zinterp).value,color='k',linewidth=3)
		
	if not os.path.exists('figures'):
		os.makedirs('figures')
	if os.path.exists(os.path.join('figures',fr.filename+'_hubble_diagram.pdf')):
		ext=1
		while os.path.exists(os.path.join('figures',fr.filename+'_hubble_diagram_'+str(ext)+'.pdf')):
			ext+=1
		outname=os.path.join('figures',fr.filename+'_hubble_diagram_'+str(ext)+'.pdf')
	else:
		outname=os.path.join('figures',fr.filename+'_hubble_diagram.pdf')
	plt.tight_layout()
	plt.savefig(outname,format='pdf')

	plt.clf()

def plot_fits(self,simfile):
    data = txtobj(simfile)
    sim = txtobj(simfile)

    # getting distance modulus is slow, so don't do it unless necessary
    getMU = False
    if len(self.options.cutwin):
        for cutopt in self.options.cutwin:
            if 'MU' in cutopt[0]: getMU = True
    for h in self.options.histvar:
        if 'MU' in h: getMU = True
            
    if 'MU' in self.options.histvar or getMU:
        if not 'MU' in data.__dict__:
            data.MU,data.MUERR = salt2mu(x1=data.x1,x1err=data.x1ERR,c=data.c,cerr=data.cERR,mb=data.mB,mberr=data.mBERR,
                                         cov_x1_c=data.COV_x1_c,cov_x1_x0=data.COV_x1_x0,cov_c_x0=data.COV_c_x0,
                                         alpha=self.options.alpha,beta=self.options.beta,
                                         x0=data.x0,sigint=self.options.sigint,z=data.zHD,M=self.options.dataM)
            from astropy.cosmology import Planck13 as cosmo
            if not 'MURES' in data.__dict__:
                data.MURES = data.MU - cosmo.distmod(data.zHD).value
        if not 'MU' in sim.__dict__:
            sim.MU,sim.MUERR = salt2mu(x1=sim.x1,x1err=sim.x1ERR,c=sim.c,cerr=sim.cERR,mb=sim.mB,mberr=sim.mBERR,
                                       cov_x1_c=sim.COV_x1_c,cov_x1_x0=sim.COV_x1_x0,cov_c_x0=sim.COV_c_x0,
                                       alpha=self.options.alpha,beta=self.options.beta,
                                       x0=sim.x0,sigint=self.options.sigint,z=sim.zHD,M=self.options.simM)
            from astropy.cosmology import Planck13 as cosmo
            if not 'MURES' in sim.__dict__:
                sim.MURES = sim.MU - cosmo.distmod(sim.zHD).value

    if self.options.scaleb4cuts:
        cols_CC = np.where((sim.SIM_TYPE_INDEX != 1))[0]
        cols_Ia = np.where((sim.SIM_TYPE_INDEX == 1))[0]
        lenCC = float(len(cols_CC))
        lenIa = float(len(cols_Ia))

    sim = self.mkcuts(sim,fitresfile=simfile)
    data = self.mkcuts(data,fitresfile=datafile)

    if self.options.journal:
        mf = factors(len(self.options.histvar))
        if self.options.nplots[0]: ysubplot = self.options.nplots[1]; xsubplot = self.options.nplots[0]
        else:
            ysubplot = mf[len(mf)/2]
            xsubplot = len(self.options.histvar)/ysubplot
        plt.rcParams['figure.figsize'] = (xsubplot*7,ysubplot*7)
        if not self.options.outfile:
            self.options.outfile = 'ovplot_%s.png'%("_".join(self.options.histvar))
    else:
        plt.rcParams['figure.figsize'] = (8.5,11)
        from matplotlib.backends.backend_pdf import PdfPages
        if not self.options.outfile:
            self.options.outfile = 'ovplot_%s.pdf'%("_".join(self.options.histvar))
        if not os.path.exists(self.options.outfile) or self.options.clobber:
            pdf_pages = PdfPages(self.options.outfile)
        else:
            print('File %s exists!  Not clobbering...'%self.options.outfile)
            return(1)

    for histvar,i in zip(self.options.histvar,
                         np.arange(len(self.options.histvar))+1):
        if self.options.journal:
            ax = plt.subplot(ysubplot,xsubplot,i)
            import string
            ax.text(-0.1, 1.05, '%s)'%string.ascii_uppercase[i-1], transform=ax.transAxes, 
                     size=20, weight='bold')
        else:
            if i%3 == 1: fig = plt.figure()
            if i == 3: subnum = 3
            else: subnum = i%3
            if len(self.options.histvar) >= 3:
                ax = plt.subplot(3,1,subnum)
            else:
                ax = plt.subplot(len(self.options.histvar),1,subnum)
        if not self.options.journal:
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
            if 'vzHD' in histvar: 
                ax.set_ylabel(histvardict[histvar],fontsize=30)
                ax.set_xlabel('$z_{CMB}$',fontsize=30)
            elif 'vmB' in histvar: 
                ax.set_ylabel(histvardict[histvar],fontsize=30)
                ax.set_xlabel('$m_{B}$',fontsize=30)

        if 'vzHD' in histvar:
            self.plt2var(data,sim,ax,histvar)
            continue
        if 'vmB' in histvar:
            self.plt2mB(data,sim,ax,histvar)
            continue
            
            
        self.options.histmin,self.options.histmax = None,None
        if len(self.options.cutwin):
            for cutopt in self.options.cutwin:
                var,min,max = cutopt[0],cutopt[1],cutopt[2]; min,max = float(min),float(max)
            if var == histvar:
                self.options.histmin = min; self.options.histmax = max
        if not self.options.histmin:
            self.options.histmin = np.min(np.append(sim.__dict__[histvar],data.__dict__[histvar]))
            self.options.histmax = np.max(np.append(sim.__dict__[histvar],data.__dict__[histvar]))


        cols_CC = np.where((sim.SIM_TYPE_INDEX != 1) & 
                           (sim.__dict__[histvar] >= self.options.histmin) &
                           (sim.__dict__[histvar] <= self.options.histmax))[0]
        cols_Ia = np.where((sim.SIM_TYPE_INDEX == 1) & 
                           (sim.__dict__[histvar] >= self.options.histmin) &
                           (sim.__dict__[histvar] <= self.options.histmax))[0]
        if not self.options.scaleb4cuts:
            lenCC = float(len(cols_CC))
            lenIa = float(len(cols_Ia))
        
        # bins command options
        if self.options.bins[0] != None: self.options.nbins = self.options.bins[0]
        if self.options.bins[1] != None: self.options.histmin = self.options.bins[1]
        if self.options.bins[2] != None: self.options.histmax = self.options.bins[2]
        print(histvar,self.options.histmin,self.options.histmax)

        histint = (self.options.histmax - self.options.histmin)/self.options.nbins
        histlen = float(len(np.where((data.__dict__[histvar] > self.options.histmin) &
                                     (data.__dict__[histvar] < self.options.histmax))[0]))
        n_nz = np.histogram(data.__dict__[histvar],bins=np.linspace(self.options.histmin,self.options.histmax,self.options.nbins))
        
        errl,erru = poisson_interval(n_nz[0])
        ax.plot(n_nz[1][:-1]+(n_nz[1][1]-n_nz[1][0])/2.,n_nz[0],'o',color='k',lw=2,label='data')
        ax.errorbar(n_nz[1][:-1]+(n_nz[1][1]-n_nz[1][0])/2.,n_nz[0],yerr=[n_nz[0]-errl,erru-n_nz[0]],color='k',fmt=' ',lw=2)
        import copy
        n_nz_chi2 = copy.deepcopy(n_nz)
        n_nz = np.histogram(sim.__dict__[histvar],bins=np.linspace(self.options.histmin,self.options.histmax,self.options.nbins))
        ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
                color='k',drawstyle='steps-mid',lw=4,label='All Sim. SNe',ls='--')
        chi2 = np.sum((n_nz[0]/float(lenIa+lenCC)*histlen-n_nz_chi2[0])**2./((erru-errl)/2.)**2.)/float(len(n_nz[0])-1)
        print('chi2 = %.3f for %s'%(chi2,histvar))

        n_nz = np.histogram(sim.__dict__[histvar][cols_CC],bins=np.linspace(self.options.histmin,self.options.histmax,self.options.nbins))
        ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
                color='b',drawstyle='steps-mid',lw=4,label='Sim. CC SNe',ls='-.')
        n_nz = np.histogram(sim.__dict__[histvar][cols_Ia],bins=np.linspace(self.options.histmin,self.options.histmax,self.options.nbins))
        ax.plot((n_nz[1][:-1]+n_nz[1][1:])/2.,n_nz[0]/float(lenIa+lenCC)*histlen,
                color='r',drawstyle='steps-mid',lw=2,label='Sim. SNe Ia')

        if self.options.ylim[0] or self.options.ylim[1]: ax.set_ylim([self.options.ylim[0],self.options.ylim[1]])
        if self.options.ylog == 'all' or histvar in self.options.ylog:
            ax.set_yscale('log')
            if not self.options.ylim[0]: ax.set_ylim(bottom=0.5)

        print('Variable: %s'%histvar)
        print('NDATA: %i'%len(data.CID))
        print('MC Scale: %.1f'%(histlen/float(lenIa+lenCC)))
        if lenIa:
            print('N(CC Sim.)/N(Ia Sim.): %.3f'%(lenCC/float(lenIa)))
        else:
            print('N(CC Sim.)/N(Ia Sim.): inf')


        if not self.options.journal and i%3 == 1:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0,# + box.height * 0.15,
                             box.width, box.height * 0.85])
            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.6),
                      fancybox=True, ncol=2,numpoints=1)
        if self.options.journal and i == 2:
            ax.legend(loc='upper center', bbox_to_anchor=(0.625, 1.0),
                      fancybox=True,numpoints=1,prop={'size':23})
    

        if self.options.interact:
            plt.show()
        
        if not self.options.journal:
            if i%3 == 1: self.plottitle(ax)
            if not i%3:
                if not os.path.exists(self.options.outfile) or self.options.clobber:
                    pdf_pages.savefig(fig)

    if not self.options.journal:
        if i%3:
            pdf_pages.savefig(fig)
        pdf_pages.close()
    else:
        if not self.options.outfile:
            outfile = 'ovplot_%s.png'%("_".join(self.options.histvar))
        else: outfile = self.options.outfile
        if not os.path.exists(outfile) or self.options.clobber:
            plt.savefig(outfile)
        else:
            print('File %s exists!  Not clobbering...'%outfile)

if __name__=='__main__':
	getObj('test')