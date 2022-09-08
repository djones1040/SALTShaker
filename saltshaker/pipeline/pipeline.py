##SALT3 pipeline
##sim -> training -> lcfitting -> salt2mu -> wfit

import subprocess
import configparser
import pandas as pd
import os
import sys
import numpy as np
import time
import glob
import warnings
import copy
import shutil
import psutil
import yaml
import shlex
import pickle
from saltshaker.pipeline.validplot import ValidPlots
cwd = os.getcwd()

def config_error():
    raise RuntimeError("'configure' stage has not been run yet")
def build_error():
    raise RuntimeError("'build' stage has not been run yet")

def boolean_string(s):
    if s not in {'False', 'True', '1', '0', None}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') | (s == '1')

def finput_abspath(finput):
    finput = finput.strip()
    if not finput.startswith('/') and not finput.startswith('$') and \
       '/' in finput: finput = '%s/%s'%(cwd,finput)
    return finput

def abspath_for_getmu(finput):
    finput = finput.strip()
    if not finput.startswith('/') and not finput.startswith('$'): finput = '%s/%s'%(cwd,finput)
    return finput

def nmlval_to_abspath(key,value):
    if not isinstance(value,list):
        valuelist = [value]
    else:
        valuelist = value
    newvlist = []
    for value in valuelist:
        if isinstance(value,str) and key.lower() in ['kcor_file','vpec_file'] and is_not_abspath(value.replace("'","")):               
            if key.lower() == 'kcor_file' and os.path.exists(os.path.expandvars('$SNDATA_ROOT/kcor/%s'%value)):
                newvlist.append(value)
            else:
                value = '%s/%s'%(cwd,value)
                newvlist.append(value)
        else:
            newvlist.append(value)
    if len(newvlist) == 1:
        return newvlist[0]
    else:
        return newvlist

def is_not_abspath(value):
    if not value.startswith('/') and not value.startswith('$') and '/' in value:
        return True
    else:
        return False
    
class SALT3pipe():
    def __init__(self,finput=None):
        self.finput = finput
        self.BYOSED = BYOSED()
        self.TrainSim = Simulation()
        self.InitLCFit = LCFitting()
        self.Training = Training()
        self.TestSim = Simulation()
        self.LCFitting = LCFitting()
        self.GetMu = GetMu()
        self.CosmoFit = CosmoFit()
        self.Data = Data()
        self.BiascorSim = Simulation(biascor=True)
        self.BiascorLCFit = LCFitting(biascor=True)

        self.build_flag = False
        self.config_flag = False
        self.glue_flag = False
        
        self.gluepairs = []
        
        self.timestamp = str(time.time())

    def gen_input(self):
        pass

    def build(self,mode='default',data=True,skip=None,onlyrun=None):
        self.build_flag = True

        if data:
            pipe_default = ['data','train','lcfit','getmu','cosmofit']
        else:
            pipe_default = ['byosed','sim','train','lcfit','getmu','cosmofit']
        if mode.startswith('default'):
            pipepros = pipe_default             
        elif mode.startswith('customize'):
            if skip is not None and onlyrun is None:
                if isinstance(skip,str):
                    skip = [skip]
                pipepros = [x for x in pipe_default if x not in skip]
            elif skip is None and onlyrun is not None:
                if isinstance(onlyrun,str):
                    onlyrun = [onlyrun]
                pipepros = onlyrun
            else:
                raise ValueError("skip and onlyrun cannot be used together")
        self.pipepros = pipepros
        print("Current procedures: ", self.pipepros)

    def configure(self):
        if not self.build_flag: build_error()
        self.config_flag = True

        config = configparser.ConfigParser()
        config.read(self.finput)
        m2df = self._multivalues_to_df
        
        if not hasattr(self, 'pipepros'):
            raise ValueError("Pipeline stages are not specified, call self.build() first.")
        
        n_lcfit = self._get_config_option(config,'pipeline','n_lcfit',dtype=int)
        n_biascorlcfit = self._get_config_option(config,'pipeline','n_biascorlcfit',dtype=int)
        plotdir = self._get_config_option(config,'pipeline','plotdir',dtype=str)
        self.n_lcfit = n_lcfit
        self.n_biascorlcfit = n_biascorlcfit
        self.plotdir = plotdir
        while not os.path.exists(plotdir): 
            try: os.mkdir(plotdir)
            except: time.sleep(2)
        self.genversion_split = self._get_config_option(config,'pipeline','genversion_split')
        self.genversion_split_biascor = self._get_config_option(config,'pipeline','genversion_split_biascor')

        if n_lcfit >= 1:
            self.InitLCFit = [LCFitting() for i in range(n_lcfit)]
            self.LCFitting = [LCFitting() for i in range(n_lcfit)]
        if n_biascorlcfit >= 1:
            self.BiascorLCFit = [LCFitting(biascor=True) for i in range(n_biascorlcfit)]
        
        for prostr in self.pipepros:
            sectionname = [x for x in config.sections() if x.startswith(prostr)]
            if len(sectionname) == 1:
                prostr = sectionname[0]
            pipepro = self._get_pipepro_from_string(prostr)
            setkeys = self._get_config_option(config,prostr,'set_key')
            if setkeys is not None:
                if isinstance(pipepro,list):
                    for i in range(len(pipepro)):
                        df = m2df(setkeys)
                        if df is not None:
                            pipepro[i].setkeys = df.set_index('label').loc[str(i)]
                            if isinstance(pipepro[i].setkeys,pd.Series):
                                pipepro[i].setkeys = pd.DataFrame([pipepro[i].setkeys])
                        else:
                            pipepro[i].setkeys = None
                else:
                    pipepro.setkeys = m2df(setkeys)
            else:
                pipepro.setkeys = None
                
            if prostr.startswith('lcfit') or prostr.startswith('initlcfit'):
                niter = n_lcfit
            elif prostr.startswith('biascorlcfit'):
                niter = n_biascorlcfit
            else:
                niter = 1
            if not isinstance(pipepro,list):
                pipepro = [pipepro]
            for i in range(niter):
                pipepro[i] = pipepro[i]
                baseinput = self._get_config_option(config,prostr,'baseinput').split(',')
                baseinput = self._drop_empty_string(baseinput)
                outname = self._get_config_option(config,prostr,'outinput').split(',')
                outname = self._drop_empty_string(outname)
                outname = [x+'.temp.{}'.format(self.timestamp) for x in outname]
                pro = self._get_config_option(config,prostr,'pro')
                batch = self._get_config_option(config,prostr,'batch',dtype=boolean_string)
                batch_info = self._get_config_option(config,prostr,'batch_info')
                translate = self._get_config_option(config,prostr,'translate',dtype=boolean_string)
                validplots = self._get_config_option(config,prostr,'validplots',dtype=boolean_string)
                proargs = self._get_config_option(config,prostr,'proargs')
                prooptions = self._get_config_option(config,prostr,'prooptions')
                snlists = self._get_config_option(config,prostr,'snlists')
                labels = self._get_config_option(config,prostr,'labels')
                drop_sim_versions = self._get_config_option(config,prostr,'drop_sim_versions')
                byosed_dir = self._get_config_option(config,prostr,'byosed_dir')
                
                if labels is not None:
                    labels = labels.split(',')
                    labels = self._drop_empty_string(labels)     
                if isinstance(baseinput,(list,np.ndarray)):
                    if 'lcfit' in prostr and not len(baseinput) == niter:
                        raise ValueError("length of input list [{}] must match n_lcfit/n_biascorlcfit [{}]".format(len(baseinput),niter))
                    if labels is None:
                        if len(baseinput)>0:
                            baseinput = baseinput[i]
                        outname=outname[i]
                if byosed_dir is not None:
                    byosed_default = '{}/byosed.params'.format(byosed_dir)
                    print("BYOSED param location: {}".format(byosed_default))
                else:
                    byosed_default = None
                pipepro[i].configure(baseinput=baseinput,
                                     setkeys=pipepro[i].setkeys,
                                     outname=outname,
                                     pro=pro,
                                     proargs=proargs,
                                     prooptions=prooptions,
                                     snlists=snlists,
                                     batch=batch,
                                     translate=translate,
                                     batch_info = batch_info,
                                     validplots=validplots,
                                     plotdir=self.plotdir,
                                     labels=labels,
                                     drop_sim_versions=drop_sim_versions,
                                     byosed_dir=byosed_dir
                                     )
#                 if hasattr(pipepro[i], 'biascor') and pipepro[i].biascor:
#                     pipepro[i].done_file = "{}_{}".format(pipepro[i].done_file,'biascor')
#                 if niter > 1:
#                     pipepro[i].done_file = "{}_{}".format(pipepro[i].done_file,i)

    def run(self,onlyrun=None):
        if not self.build_flag: build_error()
        if not self.config_flag: config_error()

        if onlyrun is not None:
            if isinstance(onlyrun,str):
                onlyrun = [onlyrun]
        
#         self.lastpipepro = self._get_pipepro_from_string(self.pipepros[-1])
        self.success = False
        for prostr in self.pipepros:
            print("Current stage: ",prostr)
            self.lastpipepro = self._get_pipepro_from_string(prostr)
            if onlyrun is not None and prostr not in onlyrun:
                continue
            try:
                i = -9
                pipepro = self._get_pipepro_from_string(prostr)
                if not isinstance(pipepro,list):
                    pipepro.run(batch=pipepro.batch,translate=pipepro.translate)
                    if pipepro.success:
                        pipepro.extract_gzfitres()
                        if pipepro.validplots:
                            print('making validation plots in %s/'%self.plotdir)
                            pipepro.validplot_run()
                    else:
                        if 'sim' in prostr.lower() and 'byosed' in self.pipepros:
                            # rerun sim to see if byosed seg fault resolves
                            for i in range(0,3):
                                print("Rerun byosed sim due to unknown seg fault, {} try".format(i))
                                pipepro.run(batch=pipepro.batch,translate=pipepro.translate)
                                if pipepro.success:
                                    pipepro.extract_gzfitres()
                                    if pipepro.validplots:
                                        print('making validation plots in %s/'%self.plotdir)
                                        pipepro.validplot_run()
                                    break
                                else:                           
                                    raise RuntimeError("Something went wrong..")
                else:
                    if 'initlcfit' in prostr.lower():
                        saltpars_dflist = []
                    for i in range(len(pipepro)):
                        pipepro[i].run(batch=pipepro[i].batch,translate=pipepro[i].translate)
                        if pipepro[i].success:
                            pipepro[i].extract_gzfitres()
                            if 'initlcfit' in prostr.lower(): #read in saltpars from initial fit
                                saltpars = pipepro[i].get_init_saltpars()
                                saltpars_dflist.append(saltpars)
                            if pipepro[i].validplots:
                                print('making validation plots in %s/'%self.plotdir)
                                pipepro[i].validplot_run()
                        else:
                            raise RuntimeError("Something went wrong..")
            except:                       
                # pickle pipeline status where it failed for continue from this point
                picklename = "pipeline.{}.pickle".format(self.timestamp)
                pickle.dump(self, open(picklename, "wb" ) )
                print("Wrote pipeline object as {}".format(picklename))
                raise RuntimeError("Something went wrong..")
            if 'initlcfit' in prostr.lower():
                #write out initpar file
                df = pd.concat(saltpars_dflist)
                outdir = self.Training._get_output_info().loc[self.Training._get_output_info().key=='outputdir','value'].values[0]
                fname = os.path.join(outdir,'snparlist.txt')
                fname_tpk = os.path.join(outdir,'tmaxlist.txt')
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                if os.path.exists(fname):
                    print("removing old file {}".format(fname))
                    os.system("rm {}".format(fname))
                with open(fname, 'w') as f:
                    f.write('# SNID zHelio x0 x1 c FITPROB\n')
                    df[['SNID','zHelio','x0','x1','c','FITPROB']].to_csv(f,index=False,sep=' ',header=None)   
                df[['SNID','PKMJD']].to_csv(fname_tpk,sep=' ',index=False,header=None)
                
        if not isinstance(self.lastpipepro,list):
            self.success = self.lastpipepro.success
        else:
            if i == len(self.lastpipepro)-1:
                self.success = np.all([p.success for p in self.lastpipepro])
            else:
                self.success = self.lastpipepro[i].success
                
        #tar temp files
        if self.success:
            print("Packing temp files")
            
            try:
                outnamelist = []
                for p in self.pipepros:
                    prooutname = self._get_pipepro_from_string(p).outname if not isinstance(self._get_pipepro_from_string(p),list) else [pi.outname for pi in self._get_pipepro_from_string(p)]                       
                    if prooutname is None:
                        continue
                    if isinstance(prooutname,list):
                        outnamelist += prooutname
                    elif isinstance(prooutname,dict):
                        outnamelist += [prooutname[key] for key in prooutname.keys()]
                    else:
                        outnamelist += [prooutname]

                dirnames = []
                for i,outname_unique in enumerate(np.unique(list(outnamelist))):
                    dirnames.append(os.path.dirname(outname_unique))
                # print(dirnames)

                for i,dirname in enumerate(np.unique(dirnames)):
                    if dirname != '':
                        tarcommand = 'tar -zcvf {}/tempfiles.{}_{}.tar.gz {}/*.temp.{}*'.format(dirname,self.timestamp,i,dirname,self.timestamp)
                    else:
                        tarcommand = 'tar -zcvf tempfiles.{}_{}.tar.gz *.temp.{}*'.format(self.timestamp,i,self.timestamp)
                    tarcommand += ' --remove-files'
                    print(tarcommand)      
                    os.system(tarcommand)
            except Exception as e:
                print("[WARNING] Unable to pack all temp files")
                print(str(e))
        else:
            raise RuntimeError("Something went wrong..")
                    
    def glue(self,pipepros=None,on='phot'):
        if not self.build_flag: build_error()
        if not self.config_flag: config_error()

        if pipepros is None:
            return
        elif not isinstance(pipepros,list) or len(pipepros) !=2:
            raise ValueError("pipepros must be list of length 2, {} of {} was given".format(type(pipepros),len(pipepros))) 
        elif not set(pipepros).issubset(self.pipepros):
            raise ValueError("one or more stages are not configured, check options in self.build()")
            
        
        print("Connecting ",pipepros)
              
        pro1 = self._get_pipepro_from_string(pipepros[0])
        if not isinstance(pro1,list):
            pro1list = [pro1]
        else:
            pro1list = pro1
        pro2 = self._get_pipepro_from_string(pipepros[1])
        if not isinstance(pro2,list):
            pro2list = [pro2]
        else:
            pro2list = pro2
            
        for i,pro1 in enumerate(pro1list):
            for j,pro2 in enumerate(pro2list):
                pro1_out = pro1.glueto(pro2)
                if isinstance(pro1, Simulation):
                    pro1_out_dict = pro1_out.copy()
                    if isinstance(pro2, LCFitting):
                        pro2_in = pro2._get_input_info().loc[on]
                        if pro1.biascor:
                            split_arr = self.genversion_split.split(',')
                        else:
                            split_arr = self.genversion_split_biascor.split(',')
                        split_idx = [int(x) for x in split_arr]
                        for tag in ['io','kcor']:
                            pro1_out = pro1_out_dict[tag][split_idx[j:j+1]] #:split_idx[j+1]] 
                            if isinstance(pro1_out,list) or isinstance(pro1_out,np.ndarray): 
                                pro2_in.loc[pro2_in['tag']==tag,'value'] = ', '.join(pro1_out)
                            else:
                                pro2_in.loc[pro2_in['tag']==tag,'value'] = pro1_out
                    elif isinstance(pro2, Training):
                        pro2_in = pro2._get_input_info()
                        for tag in ['io','kcor','subsurvey_list','ignore_filters']:     
                            pro1_out = pro1_out_dict[tag]
                            if isinstance(pro1_out,list) or isinstance(pro1_out,np.ndarray): 
                                if tag == 'io':
                                    if pro2.drop_sim_versions is not None:
                                        pro1_out = [pro1_out[i] for i in range(len(pro1_out)) if str(i) not in pro2.drop_sim_versions.split(',')]
                                    pro2_in.loc[pro2_in['tag']==tag,'value'] = ','.join(pro1_out)
                                elif tag == 'kcor':
                                    for i,survey in zip(pro1_out_dict['ind'],pro1_out_dict['survey']):
                                        if pro2.drop_sim_versions is not None and str(i) in pro2.drop_sim_versions.split(','):
                                            continue
                                        section = 'survey_{}'.format(survey.strip())
                                        if section not in pro2_in.loc[pro2_in['tag']==tag,'section'].values:
                                            df_newrow = pd.DataFrame([{'section':'survey_{}'.format(survey),'key':'kcorfile',
                                                                       'value':'','tag':tag,'label':'main'}])
                                            pro2_in = pd.concat([pro2_in,df_newrow])
                                        pro2_in.loc[(pro2_in['tag']==tag) & (pro2_in['section']==section),'value'] = pro1_out[int(i)]
                                elif tag == 'subsurvey_list':
                                    for i,survey in zip(pro1_out_dict['ind'],pro1_out_dict['survey']):
                                        if pro2.drop_sim_versions is not None and str(i) in pro2.drop_sim_versions.split(','):
                                            continue
                                        section = 'survey_{}'.format(survey.strip())
                                        if section not in pro2_in.loc[pro2_in['tag']==tag,'section'].values:
                                            df_newrow = pd.DataFrame([{'section':'survey_{}'.format(survey),'key':'subsurveylist',
                                                                       'value':'','tag':tag,'label':'main'}])
                                            pro2_in = pd.concat([pro2_in,df_newrow])
                                        if pro1_out[int(i)] is not None:
                                            pro2_in.loc[(pro2_in['tag']==tag) & (pro2_in['section']==section),'value'] = pro1_out[int(i)]
                            else:
                                if tag == 'ignore_filters':
                                    for i,survey in zip(pro1_out_dict['ind'],pro1_out_dict['survey']):
                                        if pro2.drop_sim_versions is not None and str(i) in pro2.drop_sim_versions.split(','):
                                            continue
                                        section = 'survey_{}'.format(survey.strip())
                                        if section not in pro2_in.loc[pro2_in['tag']==tag,'section'].values:
                                            df_newrow = pd.DataFrame([{'section':'survey_{}'.format(survey),'key':'ignore_filters',
                                                                       'value':'','tag':tag,'label':'main'}])
                                            pro2_in = pd.concat([pro2_in,df_newrow])
                                else:
                                    pro2_in.loc[pro2_in['tag']==tag,'value'] = pro1_out

                elif isinstance(pro1, Training) and isinstance(pro2, LCFitting):
                    pro2_in = pro2._get_input_info().loc[on]
#                     pro2_in['value'] = pro1_out
                    pro2_in['value'] = os.path.join(os.getcwd(),pro1_out)
    
                elif isinstance(pro1,LCFitting) and isinstance(pro2, Training):
                    outdir = pro2._get_output_info().loc[pro2._get_output_info().key=='outputdir','value'].values[0]
                    fname = os.path.join(outdir,'snparlist.txt')
                    fname_tpk = os.path.join(outdir,'tmaxlist.txt')
                    pro2_in = pd.DataFrame([{'label':'main','section':'iodata','key':'snparlist','value':fname},
                               {'label':'main','section':'iodata','key':'tmaxlist','value':fname_tpk}])
                    
                elif isinstance(pro2, GetMu):
                    if pro1.biascor:
#                         pro2_in = pro2._get_input_info().loc['biascor']
#                         pro2_in['value'] = pro1_out
                        if i == 0:
                            pro2_in = pro2._get_input_info().loc['biascor']
                            pro2_in['value'] = [pro1_out]
                            if len(pro1list)>1:
                                continue
                        else:
                            pro2_in['value'] += [pro1_out]
                    else:
                        if i == 0:
                            pro2_in = pro2._get_input_info().loc['normal']
                            pro2_in['value'] = [pro1_out]
                            if len(pro1list)>1:
                                continue                            
                        else:
                            pro2_in['value'] += [pro1_out]
                            
                elif isinstance(pro1, BYOSED):
                    pro2_in = pro2._get_input_info()
                    pro2_in['value'] = pro1_out
#                     print(pro2_in)
                    
                else:
#                     pro2_in = pro2._get_input_info().loc[0]
                    pro2_in = pro2._get_input_info()
                    pro2_in['value'] = pd.Series([pro1_out]*len(pro2_in['value']))
                if isinstance(pro2_in,pd.DataFrame):
                    setkeys = pro2_in
                else:
                    setkeys = pd.DataFrame([pro2_in])
                    
                if isinstance(pro1,Training):
                    # need to define the output directory *before* running training
                    pro1.configure(setkeys = pd.DataFrame([pro1._get_output_info().loc[0]]),
                                   pro=pro1.pro,
                                   proargs=pro1.proargs,
                                   baseinput=pro1.outname,
                                   prooptions=pro1.prooptions,
                                   outname=pro1.outname,
                                   batch=pro1.batch,
                                   translate=pro1.translate,
                                   validplots=pro1.validplots,
                                   plotdir=pro1.plotdir,
                                   labels=pro1.labels,
                                   drop_sim_versions=pro1.drop_sim_versions)

                if not pipepros[1].lower().startswith('cosmofit'):
                    pro2.configure(setkeys = setkeys,
                                   pro=pro2.pro,
                                   proargs=pro2.proargs,
                                   baseinput=pro2.outname,
                                   prooptions=pro2.prooptions,
                                   outname=pro2.outname,
                                   batch=pro2.batch,
                                   translate=pro2.translate,
                                   validplots=pro2.validplots,
                                   done_file=pro2.done_file,
                                   plotdir=pro2.plotdir,
                                   labels=pro2.labels,
                                   drop_sim_versions=pro2.drop_sim_versions)
                else:
#                     version_photometry = '/'+self.LCFitting[0].keys['SNLCINP']['VERSION_PHOTOMETRY']+'/'
#                     vinput = version_photometry.strip().join(setkeys['value'].values[0])
                    version_photometry = '/OUTPUT_BBCFIT/'
                    vinput = version_photometry.strip().join(setkeys['value'].values[0])
                    print("cosmofit input file = ",vinput)
                    pro2.configure(pro=pro2.pro,
                                   prooptions=pro2.prooptions,
                                   outname=vinput,
                                   batch=pro2.batch,
                                   translate=pro2.translate,
                                   validplots=pro2.validplots,
                                   plotdir=pro2.plotdir,
                                   labels=pro2.labels,
                                   drop_sim_versions=pro2.drop_sim_versions)

        self.gluepairs.append(pipepros)

    def _get_config_option(self,config,prostr,option,dtype=None):
        if config.has_option(prostr, option):
            option_value = config.get(prostr,option)
        else:
            option_value = None

        if dtype is not None:
            option_value = dtype(option_value)

        return option_value

    def _get_pipepro_from_string(self,pipepro_str):
        if pipepro_str.lower().startswith("sim"):
            pipepro = self.Simulation
        elif pipepro_str.lower().startswith("train") and "sim" not in pipepro_str.lower():
            pipepro = self.Training
        elif pipepro_str.lower().startswith("lcfit"):
            pipepro = self.LCFitting
        elif pipepro_str.lower().startswith("getmu"):
            pipepro = self.GetMu
        elif pipepro_str.lower().startswith("cosmofit"):
            pipepro = self.CosmoFit
        elif pipepro_str.lower().startswith("data"):
            pipepro = self.Data
        elif pipepro_str.lower().startswith("byosed"):
            pipepro = self.BYOSED
        elif pipepro_str.lower().startswith("biascorsim"):
            pipepro = self.BiascorSim
        elif pipepro_str.lower().startswith("biascorlcfit"):
            pipepro = self.BiascorLCFit
        elif pipepro_str.lower().startswith("trainsim"):
            pipepro = self.TrainSim
        elif pipepro_str.lower().startswith("testsim"):
            pipepro = self.TestSim
        elif pipepro_str.lower().startswith("initlcfit"):
            pipepro = self.InitLCFit
        else:
            raise ValueError("Unknow pipeline procedure:",pipepro.strip())
        return pipepro

    def _multivalues_to_df(self,values,colnames=None,stackvalues=False):
        df = pd.DataFrame([s.split() for s in values.split('\n')[1:]])
        if df.empty:
            return None
        if colnames is None:
            ncol = int(values.split('\n')[0])
            if ncol == 2:
                colnames = ['key','value']
            elif ncol == 3:
                colnames = ['section','key','value']
            elif ncol == 4:
                colnames = ['label','section','key','value']
            else:
                raise ValueError("column number for set_key must be between 2 and 4")
            if df.shape[1] > ncol:
                stackvalues = True
            # if df.shape[1] == 2:
            #     colnames = ['key','value']
            # elif df.shape[1] == 3:
            #     if np.any(df.isna()):
            #         colnames = ['key','value']
            #         stackvalues = True
            #     else:
            #         colnames=['section','key','value']
        if stackvalues and df.shape[1] > len(colnames):
            numbercol = [colnames[-1]+'.'+str(i) for i in range(df.shape[1]-len(colnames)+1)]
            df.columns = colnames[0:-1] + numbercol
            lastcol = colnames[-1]
            df[lastcol] = df[[col for col in df.columns if col.startswith(lastcol)]].values.tolist()
            df = df.drop(numbercol,axis=1)
        else:
            df.columns = colnames
        return df
    
    def _drop_empty_string(self,arr):
        return [x for x in arr if x != '']
    

class PipeProcedure():
    def __init__(self):
        self.pro = None
        self.baseinput = None
        self.setkeys = None
        self.proargs = None
        self.outname = None

    def configure(self,pro=None,baseinput=None,setkeys=None,
                  proargs=None,prooptions=None,batch=False,batch_info=None,
                  translate=False,drop_sim_versions=None,byosed_dir=None,
                  validplots=False,plotdir=None,labels=None,**kwargs):  
        if pro is not None and "$" in pro:
            self.pro = os.path.expandvars(pro)
        else:
            self.pro = pro
        if baseinput is not None and '$' in baseinput:
            self.baseinput = os.path.expandvars(baseinput)
        else:
            self.baseinput = baseinput
        self.setkeys = setkeys
        self.proargs = proargs
        self.prooptions = prooptions
        self.batch = batch
        self.translate = translate
        self.batch_info = batch_info
        self.validplots = validplots
        self.plotdir = plotdir
        self.labels = labels
        self.drop_sim_versions = drop_sim_versions
        self.byosed_dir = byosed_dir

        if self.outname is not None:
#             print(self.outname)
            if not isinstance(self.outname,(list,dict)) and not os.path.isdir(os.path.split(self.outname)[0]) and os.path.split(self.outname)[0] != '':
                os.mkdir(os.path.split(self.outname)[0])
            else:
                outname_list = self.outname if not isinstance(self.outname,dict) else self.outname.values()
                for item in outname_list:
                    if not os.path.isdir(os.path.split(item)[0]) and os.path.split(item)[0] != '':
                        os.mkdir(os.path.split(item)[0])
        self.gen_input(outname=self.outname)

    def gen_input(self,outname=None):
        pass

    def run(self,batch=None,translate=None):
#         arglist = [self.proargs] + [finput_abspath(self.finput)] +[self.prooptions]
        if hasattr(self,'success') and self.success:
            print("Skip a previously successful stage")
            return
        else:
            self.success = False
            
        time_start = time.time()
        if not os.path.split(self.finput)[0] == '':
            finput_nopath = os.path.split(self.finput)[1]
            arglist = [self.proargs] + [finput_nopath] +[self.prooptions] #input can't be absolute path for new snana submission script
#         else:
#             arglist = [self.proargs] + [finput_abspath(self.finput)] +[self.prooptions]
        arglist = list(filter(None,arglist))
        args = []
        for arg in arglist:
            if arg is not None:
                for argitem in arg.split(' '):
                    args.append(argitem)

        if translate:
            print("Translating old snana inputs")
            inputdir = os.path.split(finput_abspath(self.finput))[0]
            inputfile = os.path.split(finput_abspath(self.finput))[1]
            currentdir = os.getcwd()
            print("Entering input dir: {}".format(inputdir))
            os.chdir(inputdir)
            print("Current dir: ",os.getcwd())
            shellcommand = "submit_batch_jobs.sh --opt_translate 2 {}".format(inputfile) 
            shellrun = subprocess.run(args=shlex.split(shellcommand),capture_output=True)
            if shellrun.returncode == 1 and 'Exit after input file translation' in str(shellrun.stderr):
                print("Finished translation - {}".format(shellcommand))
                self.translate = False
            else:
                raise RuntimeError("Error occured:\n{}".format(shellrun.stdout))
            print("Going back to original dir: {}".format(currentdir))
            os.chdir(currentdir)            
            print("Current dir: ",os.getcwd())           
        
        #copy self.finput to current dir:
        currentdir = os.getcwd()
        os.system('cp {} {}'.format(finput_abspath(self.finput),currentdir))
        
        if batch: self.success = _run_batch_pro(self.pro, args, done_file=self.done_file)
        else: self.success = _run_external_pro(self.pro, args)
            
        #delete self.finput in currentdir
        if self.success:
            os.system('rm {}'.format(finput_nopath))
            
        time_end = time.time()
        time_taken = (time_end - time_start)/60. #in minutes
        print("this took {} minutes".format(time_taken))
        
    def validplot_run(self):
        pass
    
    def extract_gzfitres(self):
        pass
        
    def _get_input_info(self):
        pass
    
    def _get_output_info(self):
        pass


class PyPipeProcedure(PipeProcedure):

    def gen_input(self,outname="pipeline_generalpy_input.input"):
        self.outname = outname
        self.finput,self.keys = _gen_general_python_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                          outname=outname)

class Data(PipeProcedure):

    def configure(self,snlists=None,**kwargs):
        self.keys = {'snlists':snlists}

    def _get_output_info(self):
        df = {}
        key = 'snlists'
        df['key'] = key
        df['value'] = self.keys[key]
        return pd.DataFrame([df])

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        snlists = self._get_output_info().value.values[0]
        if not os.path.exists(snlists):
            raise ValueError("Path does not exists",snlists)             
        if pipepro.lower().startswith('train'):
            return snlists
        elif pipepro.lower().startswith('lcfit'):
            simpath = os.path.join(os.environ['SNDATA_ROOT'],'SIM/')
            idx = snlists.find(simpath)
            if idx !=0:
                raise ValueError("photometry must be in $SNDATA_ROOT/SIM")
            else:
                return os.path.dirname(snlists[len(simpath):]) 
        else:
            raise ValueError("data can only glue to training or lcfitting")
    
    def run(self,**kwargs):
        pass


class BYOSED(PyPipeProcedure):

    def configure(self,baseinput=None,setkeys=None,
                  outname="pipeline_byosed_input.input",byosed_dir="BYOSED/",
                  bkp_orig_param=False,**kwargs):   
        self.done_file = None
        self.outname = outname
        self.byosed_dir = byosed_dir
        super().configure(pro=None,baseinput=baseinput,setkeys=setkeys,byosed_dir=byosed_dir)
        byosed_default = '{}/byosed.params'.format(byosed_dir)
        #rename current byosed param
        if os.path.exists(os.path.dirname(byosed_default)):
            byosed_default = byosed_default
        elif os.path.exists(os.path.dirname(byosed_default.lower())):
            byosed_default = byosed_default.lower()
        else:
            os.makedirs(os.path.dirname(byosed_default))
#             raise ValueError("Directory {} does not exists".format(os.path.dirname(byosed_default)))
        self.byosed_default = byosed_default

        if bkp_orig_param:
            byosed_rename = "{}.{}".format(byosed_default,int(time.time()))
            if os.path.isfile(byosed_default):
                shellcommand = "cp {} {}".format(byosed_default,byosed_rename) 
                shellrun = subprocess.run(list(shellcommand.split()))
                if shellrun.returncode != 0:
                    raise RuntimeError(shellrun.stdout)
                else:
                    print("{} copied as {}".format(byosed_default,byosed_rename))
        #copy new byosed input to BYOSED folder
        shellcommand = "cp {} {}".format(outname,byosed_default)
        shellrun = subprocess.run(list(shellcommand.split()))
        if shellrun.returncode != 0:
            raise RuntimeError(shellrun.stdout)
        else:
            print("{} is copied to {}".format(outname,byosed_default))

    def run(self,**kwargs):
        self.success = True
    
    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('sim'):
            return "BYOSED {}/".format(os.path.dirname(self.byosed_default))
        else:
            raise ValueError("byosed can only glue to sim")
           
    def gen_input(self,outname="pipeline_generalpy_input.input"):
        self.outname = outname
        self.finput,self.keys = _gen_general_python_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                          outname=outname,delimiter=' ')

class Simulation(PipeProcedure):
    
    def __init__(self,biascor=False):
        self.biascor = biascor
        super().__init__()

    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  batch=False,batch_info=None,translate=False,validplots=False,
                  outname="pipeline_byosed_input.input",done_file='ALL.DONE',**kwargs):
#         print(baseinput)
#         self.done_file = finput_abspath('%s/Sim.DONE'%os.path.dirname(baseinput))
        self.done_file = done_file
        self.outname = outname
        self.prooptions = prooptions
        self.batch = batch
        self.translate = translate
        self.batch_info = batch_info
        self.validplots = validplots
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          prooptions=prooptions,batch=batch,batch_info=batch_info,
                          translate=translate,
                          validplots=validplots)
        setkeys_add = self._append_genmodel_abspath()
        setkeys = setkeys.append(setkeys_add).drop_duplicates(subset=['key'],keep='last')
        print("Re-writing sim input after appending GENMODEL to absolute path")
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          prooptions=prooptions,batch=batch,batch_info=batch_info,
                          translate=translate,
                          validplots=validplots)

    def gen_input(self,outname="pipeline_sim_input.input"):
        self.outname = outname
        self.finput,self.keys,self.done_file = _gen_snana_sim_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                                    outname=outname,done_file=self.done_file,
                                                                    batch_info=self.batch_info)

    def _get_input_info(self):
        df = pd.DataFrame()
        key = 'GENMODEL'
        keystrs = [x.split('[')[0] for x in self.keys.keys()]
        keyarr = [x for x in self.keys.keys() if x.startswith(key) and '_' not in x]  
        df0 = {}
        if len(keyarr ) > 0:
            for ki in keyarr:
                df0['key'] = key
                if '[' in ki:
                    ind = ki.split('[')[1].split(']')[0]
                    df0['value'] = self.keys[ki]
                    df0['ind'] = ind
                else:
                    df0['value'] = self.keys[ki]
                    df0['ind'] = None
                df = df.append(df0, ignore_index=True)
        else:
            df0['value'] = ''
            df0['ind'] = None
            df = df.append(df0, ignore_index=True) 
        df['key'] = df.apply(lambda row: '{}[{}]'.format(row['key'],row['ind']),axis=1)
        return df
        
    def _get_output_info(self):
        if self.batch:
            keys = ['PATH_SNDATA_SIM','GENVERSION','GENPREFIX']
        else:
            keys = ['PATH_SNDATA_SIM','GENVERSION']
        df = pd.DataFrame()
#         print(self.keys)
        for key in keys:
            df0 = {}     
            keystrs = [x.split('[')[0] for x in self.keys.keys()]
            keyarr = [x for x in self.keys.keys() if x.startswith(key)]
            if len(keyarr ) > 0:
                for ki in keyarr:
                    df0['key'] = key
                    if '[' in ki:
                        ind = ki.split('[')[1].split(']')[0]
                        df0['value'] = self.keys[ki]
                        df0['ind'] = ind
                    else:
                        df0['value'] = self.keys[ki]
                        df0['ind'] = None
                    df = df.append(df0, ignore_index=True)
            else:
                df0['value'] = ''
                df0['ind'] = None
                df = df.append(df0, ignore_index=True)               
        return df

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        df = self._get_output_info()
#         print(df)
        df_kcor = pd.DataFrame.from_dict(self._get_kcor_location(), orient='index',columns=['value'])
        df_kcor['key'] = 'KCOR_FILE'
        df_kcor = df_kcor.reset_index().rename(columns={'index':'ind'})
        df = pd.concat([df,df_kcor],ignore_index=True,sort=False)
        df = df.sort_values('ind')
        outdirs = self.get_outdirs(outinfo=df)
        simlibs = self._get_simlibs()
        surveynames = self._get_survey_names(simlibs)
        for i,o in enumerate(outdirs):
            outdirs[i] = os.path.expandvars(outdirs[i])
        #res = os.path.expandvars(outdir)

        # HACK - needs to check SCRATCH_SIMDIR or something else

        # ****This still requires sim to run before glueto***
        # path_to_check = [os.path.join(os.environ['SNDATA_ROOT'],'SIM/'),os.environ['SCRATCH_SIMDIR']]
        # outpath_list = [os.path.join(x,outdir) for x in path_to_check]
        # res = []
        # for outpath in outpath_list:
        #     if os.path.isdir(outpath):
        #         res.append(outpath)
        # if len(res) == 0:
        #     raise ValueError("No sim directory was found in {}".format(('\n').join(outpath_list)))            
        # elif len(res)>1:
        #     raise RuntimeError("More than one directories were found: \n{}".format(('\n').join(res)))
        # else:
        #     res = res[0]

        if pipepro.lower().startswith('train'):
            # if self.batch:
            #     prefix = df.loc[df.key=='GENPREFIX','value'].values[0]
            # else:
            #     prefix = df.loc[df.key=='GENVERSION','value'].values[0]

            # D. Jones - uncomment this line if this doesn't work....
            #prefix = df.loc[df.key=='GENVERSION','value'].values[0]
            ind = df.loc[df.key=='GENVERSION','ind'].values
            if ind[0] is None: ind[0] = 0
            output = ["{}/{}.LIST".format(res,prefix) for res,prefix in zip(outdirs,df.loc[df.key=='GENVERSION','value'].values)]
            kcor = df.loc[df.key=='KCOR_FILE'].set_index('ind').loc[ind,'value'].values
            try:
                survey = [surveynames[str(i)]['SURVEY'] for i in ind]
                subsurvey_list = [surveynames[str(i)]['SUBSURVEY_LIST'] for i in ind]
            except KeyError:
                survey = [surveynames[i]['SURVEY'] for i in ind]
                subsurvey_list = [surveynames[i]['SUBSURVEY_LIST'] for i in ind]
                
            return {'io':output,'kcor':kcor,'ind':ind,'survey':survey,'subsurvey_list':subsurvey_list,'ignore_filters':''}
#             return ["{}/{}.LIST".format(res,prefix) for res,prefix in zip(outdirs,df.loc[df.key=='GENVERSION','value'].values)]
        elif pipepro.lower().startswith('lcfit'):
#             print(df)
            ind = df.loc[df.key=='GENVERSION','ind'].values
            if ind[0] is None: ind[0] = 0
            output = df.loc[df.key=='GENVERSION','value'].values
            kcor = df.loc[df.key=='KCOR_FILE'].set_index('ind').loc[ind,'value'].values
            return {'io':output,'kcor':kcor,'ind':ind}
            # idx = res.find(simpath)
            # if idx !=0:
            #     raise ValueError("photometry must be in $SNDATA_ROOT/SIM")
            # else:
            #     return res[len(simpath):] 
        else:
            raise ValueError("sim can only glue to training or lcfitting")
        
    def get_outdirs(self,outinfo=None):
        if outinfo is None:
            df = self._get_output_info()
        else:
            df = outinfo
        if df.set_index('key').loc['PATH_SNDATA_SIM'].value:
            if isinstance(df.set_index('key').loc['GENVERSION'].value,str):
                outdirs = [os.sep.join(df.set_index('key').loc[['PATH_SNDATA_SIM','GENVERSION'],'value'].values.tolist())]
            else:
                outdirs = []
                for genversion in df.set_index('key').loc['GENVERSION'].value:
                    outdirs += [os.sep.join([df.set_index('key').loc['PATH_SNDATA_SIM'].value,
                                             genversion])]
        else:
            if isinstance(df.set_index('key').loc['GENVERSION'].value,str):
                outdirs = [os.sep.join(['$SNDATA_ROOT/SIM',df.set_index('key').loc['GENVERSION'].value])]
            else:
                outdirs = []
                for genversion in df.set_index('key').loc['GENVERSION'].value:
                    outdirs += [os.sep.join(['$SNDATA_ROOT/SIM',
                                             genversion])]
        return [os.path.expandvars(x) for x in outdirs]
    
    
    def _get_kcor_location(self):
        kcor_dict = {}
        for key,value in self.keys.items():
            if key.startswith('KCOR_FILE'):
                if '[' in key: label = key.split('[')[1].split(']')[0]
                else: label = 0
                kcor_dict[label] = value

        findkey = [key for key,value in self.keys.items() if key.startswith('SIMGEN_INFILE_Ia')]
        n_genversion = len([key for key,value in self.keys.items() if key.startswith('GENVERSION')])
        if len(findkey) > n_genversion:
            findkey = findkey[0:n_genversion]
        for key in findkey:
            label = key.split('[')[1].split(']')[0]
            if label in kcor_dict.keys():
                continue
            else:
                sim_input = finput_abspath(os.path.expandvars(self.keys[key]))
                config,delimiter = _read_simple_config_file(sim_input,sep=':')
                kcorfile = config['KCOR_FILE'].strip()
                if '#' in kcorfile:
                    kcorfile = kcorfile.split('#')[0]
                kcor_dict[label] = kcorfile
        return kcor_dict
    
    
    def _get_simlibs(self):
#         print(self.keys)
        simlib_dict = {}
        for key,value in self.keys.items():
            if key.startswith('SIMLIB_FILE'):
                if '[' in key: label = key.split('[')[1].split(']')[0]
                else: label = 0
                simlib_dict[label] = value

        findkey = [key for key,value in self.keys.items() if key.startswith('SIMGEN_INFILE_Ia')]
        n_genversion = len([key for key,value in self.keys.items() if key.startswith('GENVERSION')])
        if len(findkey) > n_genversion:
            findkey = findkey[0:n_genversion]
        for key in findkey:
            label = key.split('[')[1].split(']')[0]
            if label in simlib_dict.keys():
                continue
            else:
                sim_input = finput_abspath(os.path.expandvars(self.keys[key]))
                config,delimiter = _read_simple_config_file(sim_input,sep=':')
                simlib_file = config['SIMLIB_FILE'].strip()
                simlib_dict[label] = simlib_file
        return simlib_dict
    
    def _append_genmodel_abspath(self):
#         print(self.keys)
        genmodel_dict = {}
        for key,value in self.keys.items():
            if key.startswith('GENMODEL') and "_" not in key:
                if '[' in key: label = key.split('[')[1].split(']')[0]
                else: label = 0
                genmodel_dict[label] = value

        findkey = [key for key,value in self.keys.items() if key.startswith('SIMGEN_INFILE_Ia')]
        n_genversion = len([key for key,value in self.keys.items() if key.startswith('GENVERSION')])
        if len(findkey) > n_genversion:
            findkey = findkey[0:n_genversion]
        for key in findkey:
            label = key.split('[')[1].split(']')[0]
            if label in genmodel_dict.keys():
                continue
            else:
                sim_input = finput_abspath(os.path.expandvars(self.keys[key]))
                config,delimiter = _read_simple_config_file(sim_input,sep=':')
                genmodel_file = config['GENMODEL'].strip()
                genmodel_dict[label] = genmodel_file
        
        genmodel_dict_new = {}
        isbyosed = False
        for label in genmodel_dict.keys():
            genmodel_file = genmodel_dict[label] 
            if 'BYOSED' in genmodel_file and len(genmodel_file.split(' '))>1:
                genmodel_file = genmodel_file.split(' ')[1]
                isbyosed = True
            if (not genmodel_file.startswith('$') and not genmodel_file.startswith('/')) and \
              ('.' not in genmodel_file or os.path.split(genmodel_file)[0] != '' or \
              (not os.path.exists(os.path.expandvars('$SNDATA_ROOT/models/{}/{}'.format(genmodel_file.split('.')[0],genmodel_file))))):
#                 import pdb; pdb.set_trace()
                genmodel_file = '%s/%s'%(cwd,genmodel_file)
                print("GENMODEL changed to:", genmodel_file)
            if isbyosed:
                genmodel_file = 'BYOSED %s'%genmodel_file
            label = 'GENMODEL[{}]'.format(label)
            genmodel_dict_new[label] = genmodel_file
                
        df = pd.DataFrame(genmodel_dict_new,index=['value']).transpose().reset_index().rename(columns={'index':'key'})
        return df
    
    def _get_survey_names(self,simlib_dict):
        result_dict = {}
        for key,simlib_file in simlib_dict.items():
            result_dict[key] = {}
            for findkey in ['SURVEY','SUBSURVEY_LIST']:
                value = _parse_simlib(simlib_file, key=findkey)
                result_dict[key][findkey] = value
        return result_dict


class Training(PyPipeProcedure):

    def configure(self,pro=None,baseinput=None,setkeys=None,proargs=None,
                  prooptions=None,outname="pipeline_train_input.input",
                  labels=None,drop_sim_versions=None,**kwargs):
        self.done_file = None
        self.outname = outname
        self.proargs = proargs
        self.prooptions = prooptions
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          proargs=proargs,prooptions=prooptions,labels=labels,
                          drop_sim_versions=drop_sim_versions)

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('lcfit') or pipepro.lower().startswith('sim'):
            outdir = self._get_output_info().value.values[0]
            ##copy necessary files to a model folder in SNDATA_ROOT
#             modeldir = 'lcfitting/SALT3.test'
            modeldir = outdir
            #self.__transfer_model_files(outdir,modeldir,rename=False)
#             self.__copy_salt2info(modeldir,template_file='lcfitting/SALT2.INFO')
            self._set_output_info(modeldir)
#             os.environ['SNANA_MODELPATH'] = os.path.join(os.getcwd(),'lcfitting') #caused a bug
            return modeldir            
        else:
            raise ValueError("training can only glue to lcfit and sim")

    def gen_input(self,outname=[]):
        self.outname = outname
        self.outname,config_dict = _gen_training_inputs(basefilenames=self.baseinput,setkeys=self.setkeys,
                                                        outnames=self.outname,labels=self.labels)
        self.keys = config_dict['main']
        self.finput = self.outname['main']
        
    def _get_input_info(self):
        section_key_pair = [['iodata','snlists','io']]
        survey_sections = [x for x in self.keys.sections() if x.strip().lower().startswith('survey')]
        for s in survey_sections:
            section_key_pair.append([s,'kcorfile','kcor'])

        dflist = []
        for p in section_key_pair:
            df = {}
            section = p[0]
            key = p[1]
            tag = p[2]
            df['section'] = section
            df['key'] = key
            df['value'] = self.keys[section][key]
            df['tag'] = tag
            df['label'] = 'main'
            dflist.append(df)
        df2 = pd.DataFrame(dflist)
        return df2
        
#         df = {}
#         section = 'iodata'
#         key = 'snlists'
#         df['section'] = section
#         df['key'] = key
#         df['value'] = self.keys[section][key]
#         return pd.DataFrame([df])
    
    def _get_output_info(self):
        df = {}
        section = 'iodata'
        key = 'outputdir'
        df['section'] = section
        df['key'] = key
        df['value'] = self.keys[section][key]
        df['label'] = 'main'
        return pd.DataFrame([df])

    def _set_output_info(self,value):
        df = {}
        section = 'iodata'
        key = 'outputdir'
        df['section'] = section
        df['key'] = key
        df['value'] = value
        self.keys[section][key] = value
        return pd.DataFrame([df])
    
    def __copy_salt2info(self,modeldir,template_file='lcfitting/SALT2.INFO'):
        # temporarily copy SALT2.INFO to model folder, remove when SALT2.INFO can be created by training
        if not os.path.isdir(modeldir):
            os.mkdir(modeldir)
        if not os.path.isfile(os.path.join(modeldir,'SALT2.INFO')):
            shutil.copy(template_file, modeldir)        
            print("SALT2.INFO does not exist. Copying {} to {}".format(template_file,modeldir))
    
    def __transfer_model_files(self,outdir,modeldir,write_info=True,rename=True):
        modelfiles = glob.glob('{}/*.dat'.format(outdir))
        if not modelfiles:
            raise ValueError("[glueto lcfitting] File does not exist. Run training first")
        shellcommand = "cp -p {} {}".format(' '.join(modelfiles),modeldir) 
        shellrun = subprocess.run(list(shellcommand.split()))
        if shellrun.returncode != 0:
            raise RuntimeError(shellrun.stderr)
        else:
            print("salt3 model files copied to {}".format(modeldir))
        
        if write_info:
            fcolor = os.path.join(modeldir,'salt3_color_correction.dat')
            pardict = self.__read_color_law(fcolor)
            finfo = os.path.join(modeldir,'SALT2.INFO')
            if not os.path.exists(finfo):
                subprocess.run(['touch',finfo])
            self.__modify_info_file(finfo,pardict)
                
        if rename:
            files_to_rename = glob.glob('{}/*.dat'.format(modeldir))
            try:
                for f in files_to_rename:
                    shellcommand = "mv {} {}".format(f,f.replace('salt3','salt2'))
                    shellrun = subprocess.run(list(shellcommand.split()))
            except:
                raise ValueError("Can not rename salt3 files")
    
    def __modify_info_file(self,finfo,pardict):
        f = open(finfo,"r")
        lines = f.readlines()
        keys = []
        for i,line in enumerate(lines):
            if line.strip().startswith('#') or ':' not in line:
                continue
            key,value = line.split(':')[0],line.split(':')[1]
            keys.append(key)
            if key == 'RESTLAMBDA_RANGE':
                lines[i] = '{}: {} {}\n'.format(key, pardict['min_lambda'], pardict['max_lambda'])
            elif key == 'COLORLAW_VERSION':
                lines[i] = '{}: {}\n'.format(key, pardict['version'])
            elif key == 'COLORCOR_PARAMS':
                lines[i] = '{}: {} {} {} {}\n'.format(key, pardict['min_lambda'], 
                                                       pardict['max_lambda'],
                                                       pardict['npar'],
                                                       ' '.join(['{:.6f}'.format(x) for x in pardict['pvalues']]))
        outfile = open(finfo,"w")
        for line in lines:
            outfile.write(line)                                    
        if 'RESTLAMBDA_RANGE' not in keys:
            line =  '{}: {} {}\n'.format('RESTLAMBDA_RANGE', pardict['min_lambda'], pardict['max_lambda'])
            outfile.write(line)
        elif 'COLORLAW_VERSION' not in keys:
            line = '{}: {}\n'.format('COLORLAW_VERSION', pardict['version'])
            outfile.write(line)
        elif 'COLORCOR_PARAMS' not in keys:
            lines[i] = '{}: {} {} {} {}\n'.format('COLORCOR_PARAMS', pardict['min_lambda'], 
                                       pardict['max_lambda'],
                                       pardict['npar'],
                                       ' '.join(['{:.2f}'.format(x) for x in pardict['pvalues']]))
            outfile.write(line)           
        
    def __read_color_law(self,fcolor):
        f = open(fcolor,"r")
        lines = f.readlines()
        pardict = {}
        pars = []
        for i,line in enumerate(lines):
            if i == 0:
                pardict['npar'] = int(line)
            elif i in range(1,5):
                pars.append(float(line))
            else:
                pname = line.split('Salt2ExtinctionLaw.')[1].split(' ')[0].strip()
                pvalue = line.split('Salt2ExtinctionLaw.')[1].split(' ')[1].strip()
                pardict[pname] = pvalue
        pardict['pvalues'] = pars
        return pardict    


class LCFitting(PipeProcedure):
            
    def __init__(self,biascor=False):
        self.biascor = biascor
        super().__init__()
        
    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  batch=False,batch_info=None,translate=False,
                  validplots=False,outname="pipeline_lcfit_input.input",
                  done_file='ALL.DONE',plotdir=None,**kwargs):
#         self.done_file = 'ALL.DONE'
#         self.done_file = '%s/%s'%(os.path.dirname(baseinput),os.path.split(done_file)[1])
        self.done_file = done_file
        self.outname = outname
        self.prooptions = prooptions
        self.batch = batch
        self.translate = translate
        self.batch_info = batch_info
        self.validplots = validplots
        self.plotdir = plotdir
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          prooptions=prooptions,batch=batch,batch_info=batch_info,
                          translate=translate,
                          validplots=validplots,plotdir=plotdir)
        
    def gen_input(self,outname="pipeline_lcfit_input.input"):
        self.outname = outname
        self.finput,self.keys,self.done_file = _gen_snana_fit_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                     outname=outname,done_file=self.done_file,
                                                     batch_info=self.batch_info)

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('getmu'):
            output_df = self._get_output_info().set_index('key')
            if self.batch: 
                outprefix = str(output_df.loc['OUTDIR','value'])
                if self.biascor:
                    outprefix += '/{}/FITOPT000.FITRES'.format(str(output_df.loc['VERSION','value']).strip())
                output_abspath = abspath_for_getmu(outprefix) 
                return(str(output_abspath))
            else:
                outprefix = abspath_for_getmu(str(output_df.loc['TEXTFILE_PREFIX','value']).strip())
                return str(outprefix)+'.FITRES.TEXT'
        elif pipepro.lower().startswith('training'):
            return None
        else:
            raise ValueError("lcfitting can only glue to getmu or training")

    def get_outdirs(self):
        return abspath_for_getmu(self._get_output_info().value.values[0])


    def _get_input_info(self):
        section_key_pair = [['SNLCINP','VERSION_PHOTOMETRY','phot','io'],
                            ['FITINP','FITMODEL_NAME','model','io'],
                            ['SNLCINP','KCOR_FILE','phot','kcor']]
        dflist = []
        for p in section_key_pair:
            df = {}
            section = p[0]
            key = p[1]
            t = p[2]
            tag = p[3]
            df['section'] = section
            df['key'] = key
            df['value'] = self.keys[section][key]
            df['type'] = t
            df['tag'] = tag
            dflist.append(df)
        df2 = pd.DataFrame(dflist)
        
#         df = {}       
#         section = 'SNLCINP'
#         key = 'VERSION_PHOTOMETRY'
#         df['section'] = section
#         df['key'] = key
#         df['value'] = self.keys[section][key]
#         df['type'] = 'phot'
        
#         df2 = {}
#         section2 = 'FITINP'
#         key2 = 'FITMODEL_NAME'
#         df2['section'] = section2
#         df2['key'] = key2
#         df2['value'] = self.keys[section2][key2]
#         df2['type'] = 'model'
#         df2 = pd.DataFrame([df,df2])

        if not self.batch:
            return df2.set_index('type')
        else:
            section = 'HEADER'
            key = 'VERSION'
            df['section'] = section
            df['key'] = key
            df['value'] = self.keys[section][key]
            df['type'] = 'phot'
            df['tag'] = 'io'
            df2 = df2.append(df,ignore_index=True)
            return df2.set_index('type')

    def _get_output_info(self):
        if self.batch:
            section = 'HEADER'
            keys = ['OUTDIR']
            if self.biascor:
                keys.append('VERSION')
        else:
            section = 'SNLCINP'
            keys = ['TEXTFILE_PREFIX']

        df_list = []
        for key in keys:
            df = {}
            df['section'] = section
            df['key'] = key
            df['value'] = self.keys[section][key]
            df_list.append(df)
        return pd.DataFrame(df_list)

    def validplot_run(self):
        from saltshaker.pipeline.validplot import lcfitting_validplots
        self.validplot_func = lcfitting_validplots()

        self.get_validplot_inputs()

        for inputfile,inputbase in zip(self.validplot_inputfiles,self.validplot_inputbases):
            self.validplot_func.input(inputfile)
            self.validplot_func.output(outputdir=self.plotdir,prefix='valid_lcfitting_%s'%inputbase)
            self.validplot_func.run()
            
    def get_validplot_inputs(self,outname=None):
        if not self.batch and os.path.exists('%s.FITRES.TEXT'%self.keys['snlcinp']['textfile_prefix'].strip()):
            inputfiles = ['%s.FITRES.TEXT'%self.keys['snlcinp']['textfile_prefix'].strip()]
            inputbases = [self.keys['snlcinp']['textfile_prefix'].strip()]
        elif self.batch and os.path.exists(self.keys['header']['outdir'].strip()):
            inputfiles = glob.glob('%s/*/FITOPT000.FITRES'%self.keys['header']['outdir'].strip())
            inputbases = [inpf.split('/')[-2] for inpf in inputfiles]
        else: raise RuntimeError('Error in validplot_run - could not find the FITRES files created in LCFitting stage')
        if not len(inputfiles): raise RuntimeError('Error in validplot_run - could not find the FITRES files created in LCFitting stage')
            
        self.validplot_inputfiles = inputfiles
        self.validplot_inputbases = inputbases
        
        if isinstance(outname,str):
            with open(outname,'w') as f:
                f.write("INPUTFILES: {}\n".format(','.join(self.validplot_inputfiles)))
                f.write("INPUTBASES: {}\n".format(','.join(self.validplot_inputbases)))
                            
    def extract_gzfitres(self):
        gzfiles = glob.glob('%s/*/FITOPT000.FITRES.gz'%self.keys['header']['outdir'].strip())
        for gzfile in gzfiles:
            os.system('gunzip {}'.format(gzfile))
            
    def get_init_saltpars(self):
        outdir = self.get_outdirs()
        f = glob.glob('%s/*/FITOPT000*.FITRES*'%outdir)[0]
        df = pd.read_csv(f,sep='\s+',comment='#')
        df = df.rename(columns={"zHEL": "zHelio","CID":"SNID"})
        return df[['SNID','zHelio','x0','x1','c','FITPROB','PKMJD']]
            
class GetMu(PipeProcedure):
            
    def __init__(self,bbc=True):
        self.bbc = bbc
        super().__init__()
        
    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  batch=False,batch_info=None,translate=False,
                  validplots=False,plotdir=None,outname="pipeline_getmu_input.input",
                  done_file='ALL.DONE',**kwargs):
#         self.done_file = finput_abspath('%s/%s'%(os.path.dirname(baseinput),os.path.split(done_file)[1]))
        self.done_file = done_file
        self.outname = outname
        self.prooptions = prooptions
        self.batch = batch
        self.translate = translate
        self.batch_info = batch_info
        self.validplots = validplots
        self.plotdir = plotdir
        if self.translate:
            self.outdir_key = 'OUTDIR'
        else:
            self.outdir_key = 'OUTDIR_OVERRIDE'
            
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          prooptions=prooptions,batch=batch,batch_info=batch_info,
                          validplots=validplots,translate=translate,
                          done_file=self.done_file,plotdir=plotdir)

    def gen_input(self,outname="pipeline_getmu_input.input"):
        self.outname = outname
        self.finput,self.keys,self.delimiter,self.done_file = _gen_general_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                                                 outname=outname,sep=['=',': '],done_file=self.done_file,
                                                                                 outdir='Run_GetMu',batch_info=self.batch_info,
                                                                                 outdir_key=self.outdir_key)
        
    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('cosmofit'):
            return self._get_output_info()['value'].values[0]
        else:
            raise ValueError("getmu can only glue to cosmofit")
   
    def _get_input_info(self):
        df = {}
        if not self.batch:
            key = 'file'
            df['key'] = key
            df['value'] = self.keys[key]
            df['tag'] = 'normal'
            return pd.DataFrame([df]).set_index('tag')
        else:
            if 'INPDIR' in self.keys:
                key = 'INPDIR'
                df['key'] = key
                df['value'] = finput_abspath(self.keys[key])
                df['delimiter'] = self.delimiter[key]
            elif 'INPDIR+' in self.keys:
                key = 'INPDIR+'
                df['key'] = key
                df['value'] = self.keys[key]
                df['delimiter'] = self.delimiter[key]
            df['tag'] = 'normal'
            if self.bbc:
                df2 = {}
                key = 'simfile_biascor'
                df2['key'] = key
                df2['value'] = self.keys[key].strip()
                df2['tag'] = 'biascor' 
            else:
                df2 = {}
                
            return pd.DataFrame([df,df2]).set_index('tag')

    def _get_output_info(self):
        if not self.batch:
            df = {}
            key = 'prefix'
            df['key'] = key
            df['value'] = self.keys[key].strip()+'.M0DIF'
            return pd.DataFrame([df])
        else:
#             df = {'key':None,
#                   'value':[self.keys[self.outdir_key],'SALT2mu_FITOPT000_MUOPT000.M0DIF']}
            df = {'key':None,
                  'value':[self.keys[self.outdir_key],'FITOPT000_MUOPT000.M0DIF']}
            return pd.DataFrame([df])          
        
    def validplot_run(self):
        from saltshaker.pipeline.validplot import getmu_validplots
        self.validplot_func = getmu_validplots()
            
        self.get_validplot_inputs()
        for inputfile,inputbase in zip(self.validplot_inputfiles,self.validplot_inputbases):
            self.validplot_func.input(inputfile)
            self.validplot_func.output(outputdir=self.plotdir,prefix='valid_getmu_%s'%self.keys[self.outdir_key])
            self.validplot_func.run()
            
    def get_validplot_inputs(self,outname=None):
        inputfiles = glob.glob('%s/*/FITOPT000_MUOPT000.FITRES'%self.keys[self.outdir_key])
        inputbases = [inputfile.split('/')[-1] for inputfile in inputfiles]
        
        self.validplot_inputfiles = inputfiles
        self.validplot_inputbases = inputbases

        if isinstance(outname,str):
            with open(outname,'w') as f:
                f.write("INPUTFILES: {}\n".format(','.join(self.validplot_inputfiles)))
                f.write("INPUTBASES: {}\n".format(','.join(self.validplot_inputbases)))
                
    def extract_gzfitres(self):
        gzfiles = glob.glob('%s/*/FITOPT000_MUOPT000*.gz'%self.keys[self.outdir_key])
        for gzfile in gzfiles:
            os.system('gunzip {}'.format(gzfile))
                

class CosmoFit(PipeProcedure):
    def configure(self,setkeys=None,pro=None,outname=None,prooptions=None,batch=False,
                  batch_info=None,translate=False,validplots=False,plotdir=None,**kwargs):
        self.done_file = None
        if setkeys is not None:
            outname = setkeys.value.values[0]
        self.prooptions = prooptions
        self.finput = outname
        self.batch = batch
        self.translate = translate
        self.batch_info = batch_info
        self.validplots = validplots
        self.plotdir = plotdir
        super().configure(pro=pro,outname=outname,prooptions=prooptions,batch=batch,
                          batch_info=batch_info,translate=translate,
                          validplots=validplots,plotdir=plotdir)

    def _get_input_info(self):
        df = {}
        df['value'] = 'test'
        return pd.DataFrame([df])

    def validplot_run(self):
        from saltshaker.pipeline.validplot import cosmofit_validplots
        self.validplot_func = cosmofit_validplots()

        self.get_validplot_inputs()
        for inputfile,inputbase in zip(self.validplot_inputfiles,self.validplot_inputbases):
            self.validplot_func.input(inputfile)
            self.validplot_func.output(outputdir=self.plotdir,prefix='valid_cosmofit_%s'%inputbase)
            self.validplot_func.run()

    def get_validplot_inputs(self,outname=None):
        inputfile = '%s.cospar'%self.finput
        #inputbase = inputfile.split('/')[-1].split('.')[0]
        inputbase = inputfile.split('/')[0]      
        
        self.validplot_inputfiles = [inputfile]
        self.validplot_inputbases = [inputbase]
        
        if isinstance(outname,str):
            with open(outname,'w') as f:
                f.write("INPUTFILES: {}\n".format(','.join(self.validplot_inputfiles)))
                f.write("INPUTBASES: {}\n".format(','.join(self.validplot_inputbases)))
                
    def run(self,batch=None,translate=None):
#         arglist = [self.proargs] + [finput_abspath(self.finput)] +[self.prooptions]
        if hasattr(self,'success') and self.success:
            print("Skip a previously successful stage")
            return
        else:
            self.success = False
            
        time_start = time.time()
        arglist = [self.proargs] + [finput_abspath(self.finput)] +[self.prooptions]
        arglist = list(filter(None,arglist))
        args = []
        for arg in arglist:
            if arg is not None:
                for argitem in arg.split(' '):
                    args.append(argitem)
        
        if batch: self.success = _run_batch_pro(self.pro, args, done_file=self.done_file)
        else: self.success = _run_external_pro(self.pro, args)
            
        time_end = time.time()
        time_taken = (time_end - time_start)/60. #in minutes
        print("this took {} minutes".format(time_taken))
                
    
def _run_external_pro(pro,args):

    if isinstance(args, str):
        args = [args]

    print("Running",' '.join([pro] + args))
    if sys.version_info[1] > 6:
        res = subprocess.run(args = list([pro] + args),capture_output=True)
    else:
        res = subprocess.run(args = list([pro] + args))    

    if res.returncode == 0:
        print("{} finished successfully.".format(pro.strip()))
        success = True
    else:
        raise ValueError("Something went wrong..") ##possible to pass the error msg from the program?
        success = False
    return success

def _run_batch_pro(pro,args,done_file=None):
    
    print('looking for DONE_STAMP in %s'%done_file)
    if isinstance(args, str):
        args = [args]

    if done_file:
        # SNANA doesn't remove old done files
        if os.path.exists(done_file): os.system('rm %s'%done_file)

    print("[BATCH] Running",' '.join([pro] + args))
    res = subprocess.run(args = list([pro] + args),capture_output=True)
    stdout = res.stdout.decode('utf-8')

    if 'ERROR MESSAGE' in stdout:
        for line in stdout[stdout.find('ERROR MESSAGE'):].split('\n'):
            print(line)
        raise RuntimeError("Something went wrong...")
    if 'WARNING' in stdout:
        for line in stdout[stdout.find('WARNING'):].split('\n'):
            warnings.warn("The following warning occured:")
            print(line)
    if 'FATAL ERROR' in stdout:
        for line in stdout[stdout.find('FATAL ERROR'):].split('\n'):
            print(line)
        raise RuntimeError("Something went wrong...")

    if not done_file:
        for line in res.stdout.decode('utf-8').split('\n'):
            if 'DONE_STAMP' in line:
                done_file = line.split()[-1]
        # SNANA doesn't remove old done files
        if os.path.exists(done_file): os.system('rm %s'%done_file)

    if not done_file:
        raise RuntimeError('could not find DONE file name in %s output'%pro)

    job_complete=False
    while not job_complete:
        time.sleep(15)
    
        if os.path.exists(done_file): 
            job_complete = True
            # apparently there's a lag between creating the file and writing to it
            while os.stat(done_file).st_size == 0:
                time.sleep(15)

    success = False
    with open(done_file,'r') as fin:
        for line in fin:
            if 'SUCCESS' in line:
                success = True

    if success:
        print("{} finished successfully.".format(pro.strip()))

    return success


def _gen_general_python_input(basefilename=None,setkeys=None,
                              outname=None,delimiter=','):

    config = configparser.ConfigParser()
    if not os.path.isfile(basefilename):
        raise ValueError("File does not exist",basefilename)
    if not os.path.exists(os.path.dirname(outname)):
        os.makedirs(os.path.dirname(outname))
    
    config.read(basefilename)
    if setkeys is None:
        print("No modification on the input file, copying {} to {}".format(basefilename,outname))
        os.system('cp %s %s'%(basefilename,outname))
    else:
        setkeys = pd.DataFrame(setkeys)
        for index, row in setkeys.iterrows():
            sec = row['section']
            key = row['key']
            values = row['value']
            if not sec in config.sections():
                config.add_section(sec)
            if not isinstance(values,list): values = [values]
            config[sec][key] = ''
            for value in values:
                if value is None:
                    continue
                print("Adding/modifying key {}={} in [{}]".format(key,value,sec))
                config[sec][key] = config[sec][key] + '%s%s'%(value,delimiter)
            config[sec][key] = config[sec][key][:-1]
        with _open_shared_file(outname, 'w') as f:
            config.write(f)

        print("input file saved as:",outname)
    return outname,config

def _gen_general_yaml_input(basefilename=None,setkeys=None,
                            outname=None):
    
    with _open_shared_file(basefilename,'r') as f:
        config = yaml.full_load(f)
    
    if not os.path.isfile(basefilename):
        raise ValueError("File does not exist",basefilename)
    if not os.path.exists(os.path.dirname(outname)):
        os.makedirs(os.path.dirname(outname))

    if setkeys is None:
        print("No modification on the input file, copying {} to {}".format(basefilename,outname))
        os.system('cp %s %s'%(basefilename,outname))
    else:
        setkeys = pd.DataFrame(setkeys)
        for index, row in setkeys.iterrows():
            keystr = row['key']
            values = row['value']
            keys = keystr.split(':')
            config_temp = config
            for i,key in enumerate(keys):
                if key in config_temp.keys() and i<len(keys):
                    config_temp = config_temp[key]
                else:
                    raise ValueError("Can only change keys that are already in yaml file")
            config_idx = ("['%s']"*len(keys)) % tuple(keys)
            exec("config{}='{}'".format(config_idx,values))
        with _open_shared_file(outname, 'w') as f:
            yaml.dump(config,f)
        print("input file saved as:",outname)
    return outname,config

def _gen_training_inputs(basefilenames=None,setkeys=None,
                         outnames=None,labels=None):

    outnames_dict = {}
    config_dict = {}
    if isinstance(basefilenames,dict):
        labels = list(basefilenames.keys())
        basefilenames = list(basefilenames.values())
    if isinstance(outnames,dict):
        outnames = [outnames[key] for key in labels]
#     print(basefilenames)
#     print(outnames)
#     print(labels)
    for basefilename,outname,label in zip(basefilenames,outnames,labels):
        if basefilename == '':
            continue
        if not label in ['main','logging','training']:
            raise ValueError("{} is not a valid label, check input file".format(label))
        if '$' in basefilename:
            basefilename = os.path.expandvars(basefilename)
        if not os.path.isfile(basefilename):
            raise ValueError("File does not exist",basefilename)
        if not os.path.exists(os.path.dirname(outname)):
            os.makedirs(os.path.dirname(outname))
   
        if setkeys is None and label in ['main']:
            print("No modification on the input file, copying {} to {}".format(basefilename,outname))
            os.system('cp %s %s'%(basefilename,outname))
        else:
            setkeys_df = pd.DataFrame(setkeys)
#             print(setkeys_df)
            if label in setkeys_df['label'].unique():
                setkeys_l = setkeys_df.loc[setkeys_df['label']==label]
            else:
                setkeys_l = None
#             print(setkeys_l)
#             print(setkeys_df['label'].unique())
            if label in ['main','training']:
                if label == 'main':
                    for l in ['logging','training']:
                        if l not in labels:
                            continue
                        df_add = {}
                        df_add['section'] = 'iodata'
                        df_add['key'] = l+'config'
                        df_add['value'] = outnames[labels.index(l)]
                        setkeys_l = setkeys_l.append(pd.DataFrame([df_add]))
                outnames_dict[label],config_dict[label] = _gen_general_python_input(basefilename=basefilename,setkeys=setkeys_l,outname=outname)
            elif label in ['logging']:
                outnames_dict[label],config_dict[label] = _gen_general_yaml_input(basefilename=basefilename,setkeys=setkeys_l,outname=outname)

    return outnames_dict,config_dict

def _rename_duplicate_keys(keys):
    df = pd.DataFrame(keys,columns=['value'])
    df['count'] = 1
    counts = df.groupby('value').count() 
    try:
        i_genversion_start = keys.index('GENVERSION') 
        i_genversion_end = keys.index('ENDLIST_GENVERSION')
    except:
        raise ValueError("GENVERSION and ENDLIST_GENVERSION must be in the sim put")
    for key in counts.index:  
        ct = counts.loc[key,'count']
        if keys.index(key) >= i_genversion_start and keys.index(key) < i_genversion_end and ct > 0:
            df.loc[df['value']==key,'value'] = ['{}[{}]'.format(key,i) for i in range(ct)]
        else:
            df.loc[df['value']==key] = key
    newkeys = df['value'].values
    return newkeys

def _gen_snana_sim_input(basefilename=None,setkeys=None,
                         outname=None,done_file=None,batch_info=None):

    #TODO:
    #read in kwlist from standard snana kw list
    #determine if the kw is in the list

    #read in a default input file
    #add/edit the kw

    if not os.path.isfile(basefilename):
        raise ValueError("basefilename cannot be None")
    print("Load base sim input file..",basefilename)
    with _open_shared_file(basefilename) as basefile:
        lines = basefile.readlines()
    basekws = []
    basevals = []
    linenum = []

    if setkeys is None:
        config = {}
        for i,line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                kwline = line.split(":",maxsplit=1)
                kw = kwline[0]
                kv = kwline[1]
                if "#" in kwline[1]:
                    kv = kwline[1].split("#")[0].strip()
                if "GENOPT" in line:
                    kw = kwline[1].split()[0]
                    kv = kwline[1].split(maxsplit=1)[1]
                basekws.append(kw.strip())
                basevals.append(kv.strip())
                linenum.append(i)

        basekws_renamed = _rename_duplicate_keys(basekws)
        for i,kw in enumerate(basekws_renamed):
            if "BATCH_INFO" in kw and batch_info is not None:
                print("Changing BATCH_INFO to {}".format(batch_info))
                config[kw] = batch_info
            else:
                config[kw] = basevals[i]  
        if batch_info is None:
            print("No modification on the input file, copying {} to {}".format(basefilename,outname))
            os.system('cp %s %s'%(basefilename,outname))
        else:
            lines_to_write = []
            for line in lines:
                if 'BATCH_INFO' in line:
                    line = batch_info
                lines_to_write.append(line)
            with _open_shared_file(outname,"w") as outfile:
                for line in lines_to_write:
                    outfile.write(line)
            print("Write sim input to file:",outname)
        
    else:
        setkeys = pd.DataFrame(setkeys).dropna(subset=['key'])
        if np.any(setkeys.key.duplicated()):
            print(setkeys)
            raise ValueError("Check for duplicated entries for",setkeys.key[setkeys.key.duplicated()].unique())

        config = {}
        for i,line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                kwline = line.split(":",maxsplit=1)
                kw = kwline[0]
                kv = kwline[1]
                if "#" in kwline[1]:
                    kv = kwline[1].split("#")[0].strip()
                if "GENOPT" in line:
                    kw = kwline[1].split()[0]
                    kv = kwline[1].split(maxsplit=1)[1]
                basekws.append(kw.strip())
                basevals.append(kv.strip())
                linenum.append(i)

        basekws_renamed = _rename_duplicate_keys(basekws)
        # print(basekws_renamed)

        for i,kw in enumerate(basekws_renamed): 
            if kw in setkeys.key.values:
                keyvalue = setkeys[setkeys.key==kw].value.values[0]
                if isinstance(keyvalue,list):
                    val = ' '.join(list(filter(None,keyvalue)))
                else:
                    val = str(keyvalue)
                basevals[i] = val
                keystr = kw.split('[')[0]
                if "[" in kw and 'GENVERSION' not in kw:
                    lines[linenum[i]] = "GENOPT: {} {}\n".format(keystr,val)
                else:
                    lines[linenum[i]] = "{}: {}\n".format(keystr,val)
                print("Setting {} = {}".format(keystr,val.strip()))
                config[kw] = val 
            elif "BATCH_INFO" in kw and batch_info is not None:
                print("Changing BATCH_INFO to {}".format(batch_info))
                lines[linenum[i]] = "BATCH_INFO: {}\n".format(batch_info)
                config[kw] = batch_info
            else:
                config[kw] = basevals[i]          

        for key,value in zip(setkeys.key,setkeys.value):
            if not key in basekws_renamed:
                if isinstance(value,list):
                    valuestr = ' '.join(list(filter(None,value)))
                else:
                    valuestr = str(value)
                if key.startswith('GENVERSION'):
                    raise ValueError("Can not add new GENVERSION at the moment")
                if "[" in key and 'GENVERSION' not in key:
                    keystr = key.split('[')[0]
                    numstr = key.split('[')[1].split(']')[0]
                    newline = "GENOPT: {} {}\n".format(keystr,valuestr)
                    lineloc = [i for i,line in enumerate(lines) if "GENVERSION".format(numstr) in line][int(numstr)]            
                    print("Adding key {} = {} for GENVERSION[{}]".format(keystr,valuestr,numstr))
                else:
                    keystr = key
                    newline = "{}: {}\n".format(keystr,valuestr)
                    lineloc = len(lines)-1
                    print("Adding key {} = {} at the end of file".format(keystr,valuestr))
                lines.insert(lineloc+1,newline)
                config[key] = valuestr.strip()

        with _open_shared_file(outname,"w") as outfile:
            for line in lines:
                outfile.write(line)
        print("Write sim input to file:",outname)

    with _open_shared_file(outname) as fin:
        lines = fin.readlines()

    with _open_shared_file(outname,'w') as fout:
        for line in lines:
            if 'DONE_STAMP' in line:
                continue
            print(line.replace('\n',''),file=fout)
            if 'ENDLIST_GENVERSION' in line:
                print('',file=fout)
                if 'GENPREFIX' in config.keys():
                    done_file = finput_abspath('%s/%s'%('SIMLOGS_%s'%config['GENPREFIX'].split('#')[0].replace(' ',''),done_file.split('/')[-1]))
#                     print('DONE_STAMP: %s'%done_file,file=fout)

                    if os.path.exists('SIMLOGS_%s'%config['GENPREFIX'].split('#')[0].replace(' ','')):
                        print('warning : clobbering old sim dir SIMLOGS_%s so SNANA doesn\'t hang'%config['GENPREFIX'].split('#')[0].replace(' ',''))
                        os.system('rm -r SIMLOGS_%s'%config['GENPREFIX'])
#                 else:
#                     print('DONE_STAMP: %s'%done_file,file=fout)

    return outname,config,done_file


def _gen_snana_fit_input(basefilename=None,setkeys=None,
                         outname=None,done_file=None,batch_info=None):

    import f90nml
    from f90nml.namelist import Namelist
    nml = f90nml.read(basefilename)
    if 'fitinp' in nml.keys() and 'fitmodel_name' in nml['fitinp'] and isinstance(nml['fitinp']['fitmodel_name'],list):
        nml['fitinp']['fitmodel_name'] = ''.join(nml['fitinp']['fitmodel_name'])
    
    # first write the header info
    if not os.path.isfile(basefilename):
        raise ValueError("basefilename cannot be None")
    print("Load base fit input file..",basefilename)
    with _open_shared_file(basefilename) as basefile:
        lines = basefile.readlines()
    basefile.close()
    basekws = []

    #if setkeys is None:
    #    print("No modification on the input file, keeping {} as input".format(basefilename))
    #else:
    nml.__setitem__('header',Namelist())
    nml['header'].__setitem__('version','')
    snlcinp,fitinp = False,False
    for i,line in enumerate(lines):
        if '&snlcinp' in line.lower(): snlcinp = True
        elif '&fitinp' in line.lower(): fitinp = True
        elif '&end' in line.lower(): snlcinp,fitinp = False,False
        if snlcinp or fitinp: continue
        if line.startswith('#'): continue
        if not ':' in line: continue
        key = line.split(':')[0].replace(' ','')
        if key.lower() == 'version':
            value = line.split(':')[1].replace('\n','')
            nml['header'].__setitem__(key,','.join([nml['header']['version'],value]))
        else:
            value = line.split(':')[1].replace('\n','')
            nml['header'].__setitem__(key,value)
#     if not done_file: nml['header'].__setitem__('done_stamp','ALL.DONE')
#     else: nml['header'].__setitem__('done_stamp',done_file)
    
    if batch_info is not None:
        print("Changing BATCH_INFO to {}".format(batch_info))
        nml['header'].__setitem__('batch_info',batch_info)

    if setkeys is not None:
        for index, row in setkeys.iterrows():
            sec = row['section']
            key = row['key']
            v = row['value']
            if not sec.lower() in nml.keys():
                raise ValueError("No section named",sec)
            if key in nml[sec]:
                print("Setting key {}={} in &{}".format(key,v,sec))
            else:
                print("Addding key {}={} in &{}".format(key,v,sec))               
            nml[sec][key] = v
    if nml['header']['version'].lstrip().startswith(','):
        nml['header']['version'] = nml['header']['version'].split(',',maxsplit=1)[1]
    # a bit clumsy, but need to make sure these are the same for now:
    #nml['header'].__setitem__('version',nml['snlcinp']['version_photometry'])
    print("Write fit input to file:",outname)
    _write_nml_to_file(nml,outname,append=True)

    done_file = finput_abspath('%s/ALL.DONE'%(nml['header']['outdir']))
    
    return outname,nml,done_file

def _gen_general_input(basefilename=None,setkeys=None,outname=None,sep='=',
                       batch_info=None,done_file=None,outdir=None,
                       outdir_key='OUTDIR_OVERRIDE'):

    config,delimiter = _read_simple_config_file(basefilename,sep=sep)
    #if setkeys is None:
    #    print("No modification on the input file, keeping {} as input".format(basefilename))
    if setkeys is not None: #else:
        for index, row in setkeys.iterrows():
            key = row['key']
            values = row['value']
            if key not in delimiter.keys():
                delimiter[key] = '='
            if delimiter[key] == '=' and isinstance(values,(list,np.ndarray)):
                values = [','.join(values)]
            else:
                values = [values]
            for i,value in enumerate(values):
                print("Adding/modifying key {}={}".format(key,value))
                config['{}[{}]'.format(key,i)] = value
                delimiter['{}[{}]'.format(key,i)] = delimiter[key]
#     if done_file:
#         key = 'DONE_STAMP'
#         v = done_file
#         config[key] = v
#         if len(delimiter.keys()): delimiter[key] = ': '
    if outdir is not None and outdir_key not in config.keys():
        config[outdir_key] = outdir
        delimiter[outdir_key] = ': '
    if 'BATCH_INFO' in config.keys() and batch_info is not None:
        print("Changing BATCH_INFO to {}".format(batch_info))
        config['BATCH_INFO'] = batch_info
        delimiter['BATCH_INFO'] = ': '
        
    print("input file saved as:",outname)
    _write_simple_config_file(config,outname,delimiter)

    done_file = finput_abspath('%s/ALL.DONE'%(config[outdir_key]))    
    
    if len(sep) == 1: return outname,config,done_file
    else: return outname,config,delimiter,done_file

def _read_simple_config_file(filename,sep='='):
    config,delimiter = {},{}
    with _open_shared_file(filename) as f:
        lines = f.readlines()
    f.close()

    # sighhhh so many SNANA inputs with multiple key/value separators
    if isinstance(sep,str):
        sep = np.array([sep])
    else: sep = np.array(sep)

    for line in lines:
        sep_in_line = []
        for s in sep:
            if s in line and not line.strip().startswith("#"):
                key,value = line.split(s,1)
                if key not in config:
                    config[key] = value.rstrip()
                else:
                    config[key] = np.append([config[key]],[value.rstrip()])
                sep_in_line += [line.find(s)]
            else:
                sep_in_line += [None]
        sep_in_line = np.array(sep_in_line)
        iSepExists = sep_in_line != None
        if len(sep[iSepExists]) == 1: delimiter[key] = sep[iSepExists][0]
        elif len(sep[iSepExists]) > 1: delimiter[key] = sep[iSepExists][sep_in_line[iSepExists] == min(sep_in_line[iSepExists])][0]
        else: continue

    return config,delimiter

def _write_simple_config_file(config,filename,delimiter,sep='='):
    with _open_shared_file(filename,"w") as outfile:
        replace_keys = []
        for key in config.keys():
            if '[' in key:
                replace_keys.append(key.split('[')[0]) 
        for key in config.keys():
            if '[' in key:
                key_to_print = key.split('[')[0]
            else:
                key_to_print = key
            if key in replace_keys:
                continue
            values = config[key]
            if not isinstance(values,list) and not isinstance(values,np.ndarray): values = [values]
            for value in values:
                if not key in delimiter.keys(): outfile.write("{}={}\n".format(key_to_print,value))
                else: outfile.write("{}{}{}\n".format(key_to_print,delimiter[key],value))

    return

def _write_nml_to_file(nml,filename,headerlines=[],append=False):
    lines = []

    for key in nml.keys():
        if key.lower() == 'header':
            headercount = 0
            for key2 in nml[key].keys():
                value = nml[key][key2]
                if isinstance(value,str):
                    value = "{}".format(value)
                elif isinstance(value,list):
                    value = ','.join([str(x) for x in value if x is not None])
                if key2.lower() == 'version':
                    # for version in value.replace(',','').split():
                    for version in value.split(','):
                        # outfile.write("{}: {}".format(key2.upper(),version))
                        # outfile.write("\n")
                        valstr = "{}: {}\n".format(key2.upper(),version)
                        lines.insert(0,valstr)
                        headercount += 1
                else:
                    # outfile.write("{}: {}".format(key2.upper(),value))
                    # outfile.write("\n")
                    valstr = "{}: {}\n".format(key2.upper(),value)
                    lines.insert(0,valstr)
                    headercount += 1
            lines.insert(headercount,'\n\n')

        else:
            # outfile.write('&'+key.upper())
            # outfile.write('\n')
            lines.append('&'+key.upper()+'\n')
            for key2 in nml[key].keys():
                value = nmlval_to_abspath(key2,nml[key][key2])
                if isinstance(value,str) and not value.replace(".","").replace(",","").isdigit() and not value in ['T','F','True','False']:
                    value = "'{}'".format(value.replace("'",""))
                elif isinstance(value,list):
                    vlist_to_join = [x for x in value if x is not None]
                    value = ','.join(["'{}'".format(str(x.replace("'",""))) if isinstance(x,str) and not x.replace(".","").replace(",","").isdigit() else str(x) for x in vlist_to_join])
                # outfile.write("  {} = {}".format(key2.upper(),value))
                # outfile.write("\n")
                valstr = "  {} = {}\n".format(key2.upper(),value)
                lines.append(valstr)
            # outfile.write('&END\n\n')
            lines.append('&END\n\n')

    with _open_shared_file(filename,"w") as outfile:
        for line in lines:
            outfile.write(line)
    outfile.close()

    return

def _parse_simlib(simlib_file, key='SURVEY'):
    if simlib_file.strip().startswith('$'):
        simlib_file = os.path.expandvars(simlib_file)
    
    with _open_shared_file(simlib_file,"r") as f:
        lines = f.readlines()
    for line in lines:
        while len(line.split(':')) > 2:
            keysplit = line.split(':',maxsplit=1)
            keystring = keysplit[0].strip()
            value = keysplit[1].strip().split()[0]
            if key == keystring:
                return value
            else:
                line = line[line.find(value)+len(value):]
        if line.split(':')[0].strip() == key:
            return line.split(':')[1].strip()
    
def _has_handle(fpath):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    print(proc.open_files())
                    return True
        except Exception:
            pass

    return False

def _open_shared_file(filename,flag="r",max_time=600):
    status = False
    total_time = 0
    while status is False and total_time < max_time:
        if not _has_handle(finput_abspath(filename)):        
            f = open(filename, flag)
            status = True
            return f
        else:
            time.sleep(5)
            total_time += 5
    if status is False:
        raise RuntimeError('File %s is opened by another process' %filename)
