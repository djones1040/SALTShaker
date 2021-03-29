import sys,scipy
sys.path.append('/project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/saltshaker')
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
import pipeline.pipeline
#reload(pipeline.pipeline)
from pipeline.pipeline import *
import plotting.plots as plots
import util.adjfitres as adjfitres
import numpy
def run_wfirst_pipeline():
	
	#adjfitres.cutFitRes('A1_SHALLOW/ALLSurveys_Spec_DMS_SALT2/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='MEDIUM')
	#adjfitres.cutFitRes('A1_SHALLOW/ALLSurveys_Spec_DMS_SALT2/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='DEEP')
	#adjfitres.cutFitRes('A1_SHALLOW/AllSurveys_Spec_DMS_BYO/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='DEEP')
	#adjfitres.cutFitRes('A1_SHALLOW/AllSurveys_Spec_DMS_BYO/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='DEEP')
	#sys.exit()
	version='AllSurveys_Spec_DMS_BYO'
	outdir='/project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/examples/wfirst/A1_SHALLOW/'+version
	version2='ALLSurveys_Spec_DMS_SALT2'
	outdir2='/project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/examples/wfirst/A1_SHALLOW/'+version2
	fr=plots.calcMu(plots.getObj(os.path.join(outdir,'FITOPT000.FITRES'),version=version))
	fr2=plots.calcMu(plots.getObj(os.path.join(outdir2,'FITOPT000.FITRES'),version=version2))
	#'''
	zind=np.where(fr.zCMB<.25)[0]
	zind2=np.where(np.logical_and(fr.zCMB>=.25,fr.zCMB<.6))[0]
	ind=np.where((fr.BYOSED_COLOR[zind]-fr.c[zind])>.05)[0]
	inds2=np.where((fr.BYOSED_COLOR[zind2]-fr.c[zind2])<-.15)[0]
	for i in ind:
		print(fr.CID[zind][i],fr.BYOSED_COLOR[zind][i],fr.c[zind][i])
	print('high')
	for i in inds2:
		print(fr.CID[zind2][i],fr.BYOSED_COLOR[zind2][i],fr.c[zind2][i])
	
	#os.system('plot_snana.py -i %s -v %s -f A1_SHALLOW/WFIRST_ALLSURVEYS.NML'%(','.join([str(int(fr.CID[zind][i])) for i in ind]),version))
	#os.system('plot_snana.py -i %s -v %s -f A1_SHALLOW/WFIRST_ALLSURVEYS.NML'%(','.join([str(int(fr.CID[zind2][i])) for i in inds2]),version))
	#sys.exit()
	#print(len(fr.zCMB),np.median(fr.BYOSED_COLOR[fr.zCMB<=1.4]),np.median(fr.BYOSED_COLOR[fr.zCMB>1.4]))
	cutz=.8
	
	plt.scatter(fr2.SIM_c[fr2.zCMB<=cutz],fr2.SIM_c[fr2.zCMB<=cutz]-fr2.c[fr2.zCMB<=cutz],s=1,alpha=.3,label='$z<=%.1f$'%cutz)
	plt.scatter(fr2.SIM_x1[fr2.zCMB<=cutz],fr2.SIM_x1[fr2.zCMB<=cutz]-fr2.x1[fr2.zCMB<=cutz],s=1,alpha=.3,label='$z<=%.1f$'%cutz)
	stats1,edges1,bins1 = scipy.stats.binned_statistic(fr2.SIM_c[fr2.zCMB<=cutz],fr2.SIM_c[fr2.zCMB<=cutz]-fr2.c[fr2.zCMB<=cutz],
													   'mean',bins=np.linspace(np.min(fr2.SIM_c),np.max(fr2.SIM_c),15))

	plt.scatter(edges1[:-1],stats1,color="r",label='binned')
	plt.legend()
	plt.xlabel('SALT2 c')
	plt.ylabel('SALT2 c-fitted c')
	plt.savefig('salt_c.pdf',overwrite=True)
	plt.clf()
	plt.scatter(fr.BYOSED_COLOR[fr.zCMB<=cutz],fr.BYOSED_COLOR[fr.zCMB<=cutz]-fr.c[fr.zCMB<=cutz],s=1,alpha=.3,label='$z<=%.1f$'%cutz)
	stats1,edges1,bins1 = scipy.stats.binned_statistic(fr.BYOSED_COLOR[fr.zCMB<=cutz],fr.BYOSED_COLOR[fr.zCMB<=cutz]-fr.c[fr.zCMB<=cutz],
													   'mean',bins=np.linspace(np.min(fr.BYOSED_COLOR),np.max(fr.BYOSED_COLOR),15))


	plt.scatter(edges1[:-1],stats1,color="r",label='binned')
	plt.legend()
	plt.xlabel('BYOSED c')
	plt.ylabel('BYOSED c-fitted c')
	plt.savefig('test_c.pdf',overwrite=True)
	plt.clf()
	plt.scatter(fr.BYOSED_COLOR[fr.zCMB>cutz],fr.BYOSED_COLOR[fr.zCMB>cutz]-fr.c[fr.zCMB>cutz],s=1,alpha=.3,label='$z>%.1f$'%cutz)
	stats1,edges1,bins1 = scipy.stats.binned_statistic(fr.BYOSED_COLOR[fr.zCMB>cutz],fr.BYOSED_COLOR[fr.zCMB>cutz]-fr.c[fr.zCMB>cutz],
													   'mean',bins=np.linspace(np.min(fr.BYOSED_COLOR),np.max(fr.BYOSED_COLOR),15))


	plt.scatter(edges1[:-1],stats1,color="r",label='binned')
	plt.legend()
	plt.xlabel('BYOSED c')
	plt.ylabel('BYOSED c-fitted c')
	plt.savefig('test_c1.pdf',overwrite=True)
	#'''
	#sys.exit()
	#fr.cut_inrange('SNRMAX1',18,1500)
	#fr2.cut_inrange('SNRMAX1',18,1500)
	for s in ['MEDIUM','DEEP']:
		#plots.plot_zdepend(os.path.join(outdir,'FITOPT000.FITRES'),os.path.join(outdir2,'FITOPT000.FITRES'),survey=s,fitvars=['c','x1','mB'],version=version)
		
		plots.plot_zdepend(fr,fr2,survey=s,fitvars=['c','x1','mB'],alpha=.14,beta=3.1,version=version)
	#sys.exit()
	plots.plot_hubble(fr,multisurvey=True)
	plots.plot_hubble(fr2,multisurvey=True)
	#plots.plot_hubble_diff(fr,fr2,multisurvey=True)
	#sys.exit()
	lim_dict={'SHALLOW':[10,70],'MEDIUM':[10,30],'DEEP':[10,40]}
	bin_dict={'SHALLOW':70,'MEDIUM':150,'DEEP':30}
	lim_dict2={'SHALLOW':[20,24],'MEDIUM':[23.5,27],'DEEP':[23.5,27.5]}
	cuts={'zCMB':[1,2]}
	for s in ['MEDIUM']:
		plots.plot_fits(os.path.join(outdir,'FITOPT000.FITRES'),fitvars=['MURES'],datafile=os.path.join(outdir2,'FITOPT000.FITRES'),version=version,xlimits=[-.5,.3],survey=s,cuts=cuts)
		plots.plot_fits(os.path.join(outdir,'FITOPT000.FITRES'),fitvars=['c'],datafile=os.path.join(outdir2,'FITOPT000.FITRES'),version=version,xlimits=[-.2,.2],survey=s,cuts=cuts)
		plots.plot_fits(os.path.join(outdir,'FITOPT000.FITRES'),fitvars=['x1'],datafile=os.path.join(outdir2,'FITOPT000.FITRES'),version=version,xlimits=[-2,2],survey=s,cuts=cuts)
		#plots.plot_fits(os.path.join(outdir,'FITOPT000.FITRES'),fitvars=['x1vzCMB'],datafile=os.path.join(outdir2,'FITOPT000.FITRES'),version=version,xlimits=lim_dict2[s],survey=s,cuts=cuts)
		plots.plot_fits(os.path.join(outdir,'FITOPT000.FITRES'),fitvars=['SNRMAX1'],datafile=os.path.join(outdir2,'FITOPT000.FITRES'),version=version,xlimits=lim_dict[s],survey=s,nbins=bin_dict[s],cuts=cuts)
		plots.plot_fits(os.path.join(outdir,'FITOPT000.FITRES'),fitvars=['BYOSED_COLOR'],datafile=os.path.join(outdir,'FITOPT000.FITRES'),version=version,xlimits=[-.2,.2],survey=s,cuts=cuts)
		#plots.plot_fits(os.path.join(outdir,'FITOPT000.FITRES'),fitvars=['BYOSED_STRETCH'],datafile=os.path.join(outdir,'FITOPT000.FITRES'),version=version,xlimits=[-2,2],survey=s,cuts=cuts)
	#plots.plot_fits(os.path.join(outdir2,'FITOPT000.FITRES'),version=version2)
	sys.exit()
	
	pipe = SALT3pipe(finput='run_wfirst.txt')
	pipe.build(data=False,mode='customize',onlyrun=['sim','lcfit','getmu'])
	#pipe.build(data=False,mode='customize',onlyrun=['biascorsim','biascorlcfit'])
	pipe.configure()
	
	pipe.glue(['sim','lcfit'],on='phot')

	pipe.glue(['lcfit','getmu'])
	pipe.run(onlyrun=['sim','lcfit','getmu'])
	#adjfitres.cutFitRes('A1_SHALLOW/AllSurveys_Spec_DMS_BYO/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='MEDIUM')
	#adjfitres.cutFitRes('A1_SHALLOW/AllSurveys_Spec_DMS_BYO/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='DEEP')
	#adjfitres.cutFitRes('A1_SHALLOW/ALLSurveys_Spec_DMS_SALT2/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='MEDIUM')
	#adjfitres.cutFitRes('A1_SHALLOW/ALLSurveys_Spec_DMS_SALT2/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='DEEP')
	#pipe.run(onlyrun=['getmu'])
	
	#pipe.run(onlyrun=['biascorsim','biascorlcfit'])
	#adjfitres.cutFitRes('biascor_output/BIASCOR_Spec_DMS/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='MEDIUM')
	#adjfitres.cutFitRes('biascor_output/BIASCOR_Spec_DMS/FITOPT000.FITRES',cuts=[['zCMB','>1.0']],field='DEEP')

if __name__=='__main__':
	run_wfirst_pipeline()
