import sys
import numpy as np
import pandas as pd
from saltshaker.pipeline import pipeline
from saltshaker.pipeline.pipeline import *
import subprocess
import argparse
import os
import shlex
import random
import f90nml
import logging
import configparser
import time
import datetime
import glob
import copy
import pickle
import atexit
from saltshaker.pipeline.pipeline import SALT3pipe
from saltshaker.pipeline.validplot import ValidPlots,lcfitting_validplots,getmu_validplots,cosmofit_validplots
log=logging.getLogger(__name__)

def _MyPipe(mypipe):
    # Write your own pipeline in a separate file  
    # For example:
    # ./mypipetest.py
    # -------------------------------------------
    # def MyPipe(finput,**kwargs): ##do not change the name of the function [MyPipe]
    #     pipe = SALT3pipe(finput)
    #     pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','train'])
    #     pipe.configure()
    #     pipe.glue(['sim','train'])
    #     return pipe
    # -------------------------------------------
    # and set --mypipe mypipetest

    sys.path.append(os.getcwd())
    import importlib
    mymod = importlib.import_module(mypipe.split('.py')[0])
    print("Using user defined pipeline: {}.py".format(mypipe.split('.py')[0]))
    return mymod.MyPipe


class RunPipe():
    def __init__(self, pipeinput, mypipe=False, batch_mode=False,batch_script=None,start_id=None,
                 randseed=None,fseeds=None,num=None,norun=None,debug=False,timeout=None,
                 make_summary=False,validplots_only=False,success_list=None,load_from_pickle=None,
                 run_background_jobs=False,batch_job_mem_limit=None,onlyrun=None,append_sim_genversion='',
                 update_randseed=False):
        if mypipe is None:
            self.pipedef = self.__DefaultPipe
        else:
            self.pipedef = self.__MyPipe
        self.batch_mode = batch_mode
        self.batch_script = batch_script
        self.pipeinput = pipeinput
        self.mypipe = mypipe
        self.randseed = randseed
        if (not self.batch_mode) and (self.randseed is None):
            self.randseed = random.sample(range(100000),1)[0]
            print("No randseed was given, auto-generated randseed={}".format(self.randseed))
        self.fseeds = fseeds
        self.start_id = start_id
        self.num = num
        if num is not None:
            self.num += start_id
        self.norun = norun
        self.debug = debug
        self.timeout = timeout
        self.make_summary = make_summary
        self.validplots_only = validplots_only
        self.success_list = success_list
        self.load_from_pickle = load_from_pickle
        self.run_background_jobs = run_background_jobs
        self.batch_job_mem_limit = batch_job_mem_limit
        self.onlyrun = onlyrun.split(',') if onlyrun is not None else None
        self.append_sim_genversion = append_sim_genversion
        self.update_randseed = update_randseed
 
    def __DefaultPipe(self):
        pipe = SALT3pipe(finput=self.pipeinput)
        pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','lcfit'])
        pipe.configure()
        pipe.glue(['sim','lcfit'],on='phot')
        return pipe
        
    def __MyPipe(self,**kwargs): 
        MyPipe = _MyPipe(self.mypipe)
        pipe = MyPipe(finput=self.pipeinput,**kwargs)
        return pipe

    def _add_suffix(self,pro,keylist,suffix,section=None,add_label=None):
        df = pd.DataFrame() 
        if str(suffix).isnumeric():
            suffix = int(suffix)
            sformat = '03d'
        else:
            sformat = ''
        for i,key in enumerate(keylist):
            if isinstance(pro.keys, dict) and not isinstance(pro.keys,f90nml.namelist.Namelist):
                keystrlist = [x for x in pro.keys.keys() if x.startswith(key)]
            elif isinstance(pro.keys, (configparser.ConfigParser,f90nml.namelist.Namelist)):
                keystrlist = []
                for sec in section:
                    keystrlist += [x for x in pro.keys[sec] if x.startswith(key)]
            for keystr in keystrlist:
                if section is None:
                    val_old = pro.keys[keystr]
                    sec = None
                else:
                    sec = section[i]
                    val_old = pro.keys[sec][keystr]
                if isinstance(val_old,list):
                    val_new = ['{}_{:{sformat}}'.format(x.strip(),suffix,sformat=sformat) for x in val_old]
                elif isinstance(val_old,dict):
                    val_new = {}
                    for dictkey in val_old:
                        val_new[dictkey] = '{}_{:{sformat}}'.format(val_old.strip(),suffix,sformat=sformat)
                else:
                    val_new = '{}_{:{sformat}}'.format(val_old.strip(),suffix,sformat=sformat)
                df = pd.concat([df,pd.DataFrame([{'section':sec,'key':keystr,'value':val_new,'label':add_label}])])
        return df
    
    def _reconfig_w_suffix(self,proname,df,suffix,done_file=None,byosed=False,**kwargs):
        outname_orig = copy.copy(proname.outname)
        if not byosed:            
            if isinstance(outname_orig,list):
                proname.outname = ['{}_{:03d}'.format(x,self.num) for x in outname_orig]
            elif isinstance(outname_orig,dict):
                for dictkey in outname_orig:
                    proname.outname[dictkey] = '{}_{:03d}'.format(outname_orig[dictkey],self.num)
            else:
                proname.outname = '{}_{:03d}'.format(outname_orig,self.num)
#         else:
#             proname.byosed_dir = '{}_{:03d}'.format(proname.byosed_dir,self.num)

        proname.configure(pro=proname.pro,baseinput=outname_orig,setkeys=df,prooptions=proname.prooptions,
                          batch=proname.batch,translate=proname.translate,validplots=proname.validplots,
                          outname=proname.outname,
                          proargs=proname.proargs,plotdir=proname.plotdir,labels=proname.labels,
                          done_file=done_file,drop_sim_versions=proname.drop_sim_versions,
                          byosed_dir=proname.byosed_dir,**kwargs)  
    
    def make_validplots_sum(self,prostr,inputfile_sum,outputdir,prefix_sum='sum_valid'):
        if prostr.startswith('lcfit'):
            prostr = 'lcfitting'
        if 'lcfit' in prostr:
            validfunc_str = 'lcfitting'
        else:
            validfunc_str = prostr
        validplots_sum = eval(validfunc_str.strip().lower()+'_validplots()')
        prefix_sum = '{}_{}'.format(prefix_sum,prostr)
        validplots_sum.input(inputfile_sum)
        validplots_sum.output(outputdir=outputdir,prefix=prefix_sum)
        validplots_sum.run()        
    
    def _parse_validplot_info(self,infofile):
        with open(infofile,'r') as f:
            lines = f.readlines()
        info_dict = {}
        for line in lines:
            key, value = line.split(':')
            info_dict[key.strip().lower()] = value.strip().split()
        return info_dict
        
    def _combine_fitres(self,files,pro=None,outsuffix=None):
        outname = '{}_{}.FITRES'.format(pro,outsuffix)
        print("combining files {} to {}".format(files,outname))
        dflist = []
        for f in files:
            df = pd.read_csv(f,sep='\s+',comment='#')
            dflist.append(df)
        data = pd.concat(dflist,ignore_index=True,sort=False).fillna('NULL')
        data.to_csv(outname,index=False,sep=' ',float_format='%.5e')
        
    def _combine_cospar(self,files,pro='cosmofit',outsuffix=None):
        all_lines = []
        lnum = 0
        for file in files:
            with open(file,'r') as f:
                lines = f.readlines()
            for line in lines:
                if not (line.startswith('#') and lnum>0):
                    all_lines.append(line)
                    lnum += 1
        outname = '{}_{}.cospar'.format(pro,outsuffix)
        with open(outname,'w') as outf:
            outf.writelines(all_lines)
    
    def _combine_nuisance(self,files,pro=None,outsuffix=None,appendfile=True,outfile=None):
        outlines = []
        for file in files:
            with open(file,'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith('#'):
                        outlines.append(line)
        if not appendfile:
            outname = '{}_{}.nuisance.FITRES'.format(pro,outsuffix)
            with open(outname,'r+') as outf:
                outf.writelines(outlines)                           
        else:
            if outfile is None:
                raise ValueError("outfile can't be None when appendfile=True")
            else:
                outname = outfile
                with open(outname,'r') as fin:
                    lines = fin.readlines()
                comment_count = 0
                for line in lines:
                    if line.strip().startswith('#'):
                        comment_count += 1
                lines = outlines + lines[comment_count:]
                with open(outname,'w') as outf:
                    outf.writelines(lines)                           
        
    def combine_validplot_inputs(self,pros=[],nums=[],outsuffix='combined'):
        for pro in pros:
            if pro.startswith('lcfit'):
                pro = 'lcfitting'
            inputfiles = []
            if len(nums) < 2:
                raise ValueError("No need to combine if numbers < 2")
            for n in nums:
                infofiles = glob.glob('misc/{}_validplot_info_{}*.txt'.format(pro,n))
                for infofile in infofiles:
                    res = self._parse_validplot_info(infofile)
                    if isinstance(res['inputfiles'],list):
                        inputfiles += res['inputfiles']
                    elif isinstance(res['inputfiles'],str):
                        inputfiles.append(res['inputfiles'])
            if pro.startswith('cosmofit'):
                self._combine_cospar(inputfiles,pro,outsuffix)
            else:
                self._combine_fitres(inputfiles,pro,outsuffix)
                if pro.startswith('getmu'):
                    self._combine_nuisance(inputfiles,pro,outsuffix,outfile='{}_{}.FITRES'.format(pro,outsuffix))
    
    def run(self):
        
        if self.validplots_only:
            if self.success_list is None:
                raise RuntimeError("Must provide success list for making validplots only")
            else:
                print("validplots=True. Making Validplots only")
                self.pipeinit = self.pipedef(skip_config=True)
                with open(self.success_list,'r') as f: 
                    success_list = f.readlines()
                self.success_list = [str(x).strip() for x in success_list]
                self.gen_validplots()        
                return
        
        if self.batch_mode == 0:
            if self.load_from_pickle is None:
                self.pipe = self.pipedef()
                
                if self.randseed is not None:
#                     if not any([p.startswith('sim') or p.startswith('biascorsim') for p in self.pipe.pipepros]):
                    if not any(['sim' in p for p in self.pipe.pipepros]):
                        raise RuntimeError("randseed was given but sim/biascorsim is not defined in the pipeline")
                    print('randseed = {}'.format(self.randseed)) 

                    if self.num is not None:
#                         if any([p.startswith('byosed') for p in self.pipe.pipepros]):
#                             self._reconfig_w_suffix(self.pipe.BYOSED,None,self.num,done_file=None,byosed=True)
                        if any([p.startswith('trainsim') for p in self.pipe.pipepros]):
                            suffix = '{}_train_{:03d}'.format(self.append_sim_genversion,self.num)
                            if suffix.startswith('_'):
                                suffix = suffix.replace('_','',1)
                            df_sim_train = self._add_suffix(self.pipe.TrainSim,['GENVERSION','GENPREFIX'],suffix)
                            done_file = "{}_{:03d}/ALL.DONE".format(os.path.dirname(self.pipe.TrainSim.done_file.strip()),self.num)
                            self._reconfig_w_suffix(self.pipe.TrainSim,df_sim_train,self.num,done_file=done_file)
#                             if ['byosed','sim'] in self.pipe.gluepairs: 
#                                 self.pipe.glue(['byosed','sim'])
                        if any([p.startswith('testsim') for p in self.pipe.pipepros]):
                            suffix = '{}_test_{:03d}'.format(self.append_sim_genversion,self.num)
                            if suffix.startswith('_'):
                                suffix = suffix.replace('_','',1)
                            df_sim_test = self._add_suffix(self.pipe.TestSim,['GENVERSION','GENPREFIX'],suffix)
                            done_file = "{}_{:03d}/ALL.DONE".format(os.path.dirname(self.pipe.TestSim.done_file.strip()),self.num)
                            self._reconfig_w_suffix(self.pipe.TestSim,df_sim_test,self.num,done_file=done_file)
#                             if ['byosed','sim'] in self.pipe.gluepairs: 
#                                 self.pipe.glue(['byosed','sim'])
                        if any([p.startswith('biascorsim') for p in self.pipe.pipepros]):
                            suffix = '{}_biascor_{:03d}'.format(self.append_sim_genversion,self.num)
                            if suffix.startswith('_'):
                                suffix = suffix.replace('_','',1)
                            df_sim_biascor = self._add_suffix(self.pipe.BiascorSim,['GENVERSION','GENPREFIX'],suffix)
                            done_file = "{}_{:03d}/ALL.DONE".format(os.path.dirname(self.pipe.BiascorSim.done_file.strip()),self.num)
                            self._reconfig_w_suffix(self.pipe.BiascorSim,df_sim_biascor,self.num,done_file=done_file)
                        if any([((p.startswith('train')) and ('sim' not in p)) for p in self.pipe.pipepros]): 
                            df_train = self._add_suffix(self.pipe.Training,['outputdir'],self.num,section=['iodata'],add_label='main')
                            self._reconfig_w_suffix(self.pipe.Training,df_train,self.num)
                            if ['trainsim','training'] in self.pipe.gluepairs: 
                                self.pipe.glue(['trainsim','training'])
                            if ['training', 'biascorsim'] in self.pipe.gluepairs:
                                self.pipe.glue(['training', 'biascorsim'])
                            if ['initlcfit','training'] in self.pipe.gluepairs:
                                self.pipe.glue(['initlcfit','training'])
                        if any([p.startswith('initlcfit') for p in self.pipe.pipepros]):    
                            for i in range(self.pipe.n_lcfit):
                                df_lcfit = self._add_suffix(self.pipe.InitLCFit[i],['outdir'],self.num,section=['header'])
                                done_file = "{}_{:03d}/ALL.DONE".format(os.path.dirname(self.pipe.InitLCFit[i].done_file.strip()),self.num)
                                self._reconfig_w_suffix(self.pipe.InitLCFit[i],df_lcfit,self.num,done_file=done_file)
                            if ['trainsim','initlcfit'] in self.pipe.gluepairs:
                                self.pipe.glue(['trainsim','initlcfit'],on='phot')
                        if any([p.startswith('lcfit') for p in self.pipe.pipepros]):    
                            for i in range(self.pipe.n_lcfit):
                                df_lcfit = self._add_suffix(self.pipe.LCFitting[i],['outdir'],self.num,section=['header'])
                                done_file = "{}_{:03d}/ALL.DONE".format(os.path.dirname(self.pipe.LCFitting[i].done_file.strip()),self.num)
                                self._reconfig_w_suffix(self.pipe.LCFitting[i],df_lcfit,self.num,done_file=done_file)
                            if ['testsim','lcfit'] in self.pipe.gluepairs:
                                self.pipe.glue(['testsim','lcfit'],on='phot')
                            if ['trainsim','lcfit'] in self.pipe.gluepairs:
                                self.pipe.glue(['trainsim','lcfit'],on='phot')
                            if ['training','lcfit'] in self.pipe.gluepairs:
                                self.pipe.glue(['training','lcfit'],on='model')
    #                             self._reconfig_w_suffix(self.pipe.LCFitting[i],None,self.num)
                        if any([p.startswith('biascorlcfit') for p in self.pipe.pipepros]):       
                            for i in range(self.pipe.n_biascorlcfit):
                                df_lcfit_biascor = self._add_suffix(self.pipe.BiascorLCFit[i],['outdir'],self.num,section=['header'])
                                done_file = "{}_{:03d}/ALL.DONE".format(os.path.dirname(self.pipe.BiascorLCFit[i].done_file.strip()),self.num)
                                self._reconfig_w_suffix(self.pipe.BiascorLCFit[i],df_lcfit_biascor,self.num,done_file=done_file)
    #                             self._reconfig_w_suffix(self.pipe.BiascorLCFit[i],None,self.num)
                            if ['biascorsim','biascorlcfit'] in self.pipe.gluepairs:
                                self.pipe.glue(['biascorsim','biascorlcfit'],on='phot')
                            if ['training','biascorlcfit'] in self.pipe.gluepairs:
                                self.pipe.glue(['training','biascorlcfit'],on='model')
                        if any([p.startswith('getmu') for p in self.pipe.pipepros]): 
                            df_getmu = self._add_suffix(self.pipe.GetMu,[self.pipe.GetMu.outdir_key],self.num)
                            done_file = "{}_{:03d}/ALL.DONE".format(os.path.dirname(self.pipe.GetMu.done_file.strip()),self.num)
                            self._reconfig_w_suffix(self.pipe.GetMu,df_getmu,self.num,done_file=done_file)
                            if ['lcfit','getmu'] in self.pipe.gluepairs:
                                self.pipe.glue(['lcfit','getmu'])
                            if ['biascorlcfit','getmu'] in self.pipe.gluepairs:
                                self.pipe.glue(['biascorlcfit','getmu'])
                            if ['getmu','cosmofit'] in self.pipe.gluepairs:
                                self.pipe.glue(['getmu','cosmofit'])

                    if any([p.startswith('trainsim') for p in self.pipe.pipepros]):
                        sim = self.pipe.TrainSim
                        randseed_old = sim.keys['RANSEED_REPEAT']
                        if 'BATCH_INFO' in sim.keys:
                            nrepeat = sim.keys['BATCH_INFO'].strip().split(' ')[-1]
                        else:
                            nrepeat = randseed_old.split(' ')[0]
                        randseed_new = [nrepeat,str(self.randseed)]
                        df = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                        sim.configure(pro=sim.pro,baseinput=sim.outname,setkeys=df,prooptions=sim.prooptions,
                                      batch=sim.batch,translate=sim.translate,validplots=sim.validplots,
                                      outname=sim.outname)    
                    if any([p.startswith('testsim') for p in self.pipe.pipepros]):
                        sim = self.pipe.TestSim
                        randseed_old = sim.keys['RANSEED_REPEAT']
                        if 'BATCH_INFO' in sim.keys:
                            nrepeat = sim.keys['BATCH_INFO'].strip().split(' ')[-1]
                        else:
                            nrepeat = randseed_old.split(' ')[0]
                        randseed_new = [nrepeat,str(self.randseed+5)]
                        df = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                        sim.configure(pro=sim.pro,baseinput=sim.outname,setkeys=df,prooptions=sim.prooptions,
                                      batch=sim.batch,translate=sim.translate,validplots=sim.validplots,
                                      outname=sim.outname)
                    if any([p.startswith('biascorsim') for p in self.pipe.pipepros]):
                        sim_biascor = self.pipe.BiascorSim
                        randseed_old = sim_biascor.keys['RANSEED_REPEAT']
                        if 'BATCH_INFO' in sim_biascor.keys:
                            nrepeat = sim_biascor.keys['BATCH_INFO'].strip().split(' ')[-1]
                        else:
                            nrepeat = randseed_old.split(' ')[0]
                        randseed_new = [nrepeat,str(self.randseed+10)]
                        df_biascor = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                        sim_biascor.configure(pro=sim_biascor.pro,baseinput=sim_biascor.outname,setkeys=df_biascor,prooptions=sim_biascor.prooptions,
                                              batch=sim_biascor.batch,translate=sim_biascor.translate,validplots=sim_biascor.validplots,
                                              outname=sim_biascor.outname)       
            else:
                print("Load saved pipeline object from a previous run")
                self.pipe = pickle.load(open(self.load_from_pickle, "rb"))
                #redefine randseed
                if self.randseed is not None and self.update_randseed:
                    print("Updating randseed = {}".format(self.randseed))
                    if any([p.startswith('trainsim') for p in self.pipe.pipepros]):
                        sim = self.pipe.TrainSim
                        if hasattr(sim,'success') and not sim.success:
                            randseed_old = sim.keys['RANSEED_REPEAT']
                            if 'BATCH_INFO' in sim.keys:
                                nrepeat = sim.keys['BATCH_INFO'].strip().split(' ')[-1]
                            else:
                                nrepeat = randseed_old.split(' ')[0]
                            randseed_new = [nrepeat,str(self.randseed)]
                            df = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                            baseinput = os.path.join(os.path.split(sim.outname)[0],'LEGACY_'+os.path.split(sim.outname)[1])
                            print("read baseinput from: {}".format(baseinput))
                            sim.configure(pro=sim.pro,baseinput=baseinput,setkeys=df,prooptions=sim.prooptions,
                                          batch=sim.batch,translate=True,validplots=sim.validplots,
                                          outname=sim.outname)    
                    if any([p.startswith('testsim') for p in self.pipe.pipepros]):
                        sim = self.pipe.TestSim
                        if hasattr(sim,'success') and not sim.success:
                            randseed_old = sim.keys['RANSEED_REPEAT']
                            if 'BATCH_INFO' in sim.keys:
                                nrepeat = sim.keys['BATCH_INFO'].strip().split(' ')[-1]
                            else:
                                nrepeat = randseed_old.split(' ')[0]
                            randseed_new = [nrepeat,str(self.randseed+5)]
                            df = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                            baseinput = os.path.join(os.path.split(sim.outname)[0],'LEGACY_'+os.path.split(sim.outname)[1])
                            print("read baseinput from: {}".format(baseinput))
                            sim.configure(pro=sim.pro,baseinput=baseinput,setkeys=df,prooptions=sim.prooptions,
                                          batch=sim.batch,translate=True,validplots=sim.validplots,
                                          outname=sim.outname)
                    if any([p.startswith('biascorsim') for p in self.pipe.pipepros]):
                        sim_biascor = self.pipe.BiascorSim
                        if hasattr(sim_biascor,'success') and not sim_biascor.success:
                            randseed_old = sim_biascor.keys['RANSEED_REPEAT']
                            if 'BATCH_INFO' in sim_biascor.keys:
                                nrepeat = sim_biascor.keys['BATCH_INFO'].strip().split(' ')[-1]
                            else:
                                nrepeat = randseed_old.split(' ')[0]
                            randseed_new = [nrepeat,str(self.randseed+10)]
                            baseinput = os.path.join(os.path.split(sim_biascor.outname)[0],'LEGACY_'+os.path.split(sim_biascor.outname)[1])
                            print("read baseinput from: {}".format(baseinput))
                            df_biascor = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                            sim_biascor.configure(pro=sim_biascor.pro,baseinput=baseinput,setkeys=df_biascor,prooptions=sim_biascor.prooptions,
                                                  batch=sim_biascor.batch,translate=True,validplots=sim_biascor.validplots,
                                                  outname=sim_biascor.outname)   
                
            
            if not self.norun:
                #remove success files from previous runs
                if os.path.exists('PIPELINE_{}.DONE'.format(self.num)):
                    print("Removing old .DONE files")
                    os.system('rm PIPELINE_{}.DONE'.format(self.num))
                    
                self.pipe.run(onlyrun=self.onlyrun)
                
                if self.pipe.success:
                    os.system('echo SUCCESS > PIPELINE_{}.DONE'.format(self.num))
                   
                    for proname in ['lcfit','biascorlcfit','getmu','cosmofit']:
                        if proname in self.pipe.pipepros:
                            pro = self.pipe._get_pipepro_from_string(proname)
                            if not os.path.isdir('misc'):
                                os.makedirs('misc')
                            if isinstance(pro,list):
                                for i in range(len(pro)):
                                    if pro[i].validplots:
                                        if proname.startswith('lcfit'):
                                            proname = 'lcfitting'
                                        pro[i].get_validplot_inputs(outname='misc/{}_validplot_info_{}_{}.txt'.format(proname,self.num,i))
                            else:
                                if pro.validplots:
                                    pro.get_validplot_inputs(outname='misc/{}_validplot_info_{}.txt'.format(proname,self.num))
                else:
                    os.system('echo FAILED > PIPELINE_{}.DONE'.format(self.num))                           
                               
        else:
            if not self.run_background_jobs:
                if self.batch_script is None:
                    raise RuntimeError("batch script is None")
                else:
                    f = open(self.batch_script,'r')
                    lines = f.read()
                    f.close()
            if self.fseeds is None:
                print("No randseed file provided. A random list generated")
                self.randseeds = random.sample(range(100000), self.batch_mode)
                with open('randseeds.out','w') as fs:
                    for s in self.randseeds:
                        fs.write(str(s)+'\n')
            else:
                with open(self.fseeds,'r') as f:
                    seeds = f.readlines()
                    self.randseeds=[int(x) for x in seeds[0:self.batch_mode]]
                    print('randseeds = ',self.randseeds)
            pypro = os.path.expandvars('$MY_SALT3_DIR/SALTShaker/saltshaker/pipeline/runpipe.py')
            if self.debug:
                pycommand_base = 'python {} -c {} --mypipe {} --batch_mode 0'.format(pypro,self.pipeinput,self.mypipe)
            else:
                pycommand_base = 'runpipe -c {} --mypipe {} --batch_mode 0'.format(self.pipeinput,self.mypipe)
            shellrun_list = []
            for i in range(self.batch_mode):
                pycommand = pycommand_base + ' --randseed {} --num {}'.format(self.randseeds[i],i+self.start_id)
                if self.norun:
                    pycommand += ' --norun'
                if self.onlyrun is not None:
                    pycommand += ' --onlyrun {}'.format(','.join(self.onlyrun))
                if self.append_sim_genversion != '':
                    pycommand += ' --append_sim_genversion {}'.format(self.append_sim_genversion)
                if not self.run_background_jobs:
                    cwd = os.getcwd()
                    outfname = os.path.join(cwd,'test_pipeline_batch_script_{:03d}'.format(i+self.start_id))
                    outf = open(outfname,'w')
                    if self.batch_job_mem_limit is not None:
                        lines = lines.split('\n')
                        line_idx = [i for i in range(len(lines)) if '--mem-per-cpu' in lines[i]][0]
                        lines[line_idx] = lines[line_idx].split('=')[0]+"="+str(self.batch_job_mem_limit)
                        lines = "\n".join(lines)
                    outf.write(lines)
                    outf.write('\n')
                    outf.write(pycommand)
                    outf.write('\n')
                    outf.close()
                    print('Submitting batch job...')
                    shellcommand = "sbatch {}".format(outfname) 
                    print(shellcommand)

                    shellrun = subprocess.run(args=shlex.split(shellcommand))
                else:
                    pycommand += ' --timeout {}'.format(self.timeout)
                    logfile = 'log{}_single'.format(i+self.start_id)
                    fstdout = open(logfile.strip()+'.out','w') 
                    fstderr = open(logfile.strip()+'.err','w')
                    print("Running: ", pycommand)
                    shellrun_list.append(subprocess.Popen(args=shlex.split(pycommand),stdout=fstdout, stderr=fstderr))         
            
            if not self.norun:
                #wait for all jobs to finish
                time_start = time.time()
                
                all_finish = False
                job_ids = np.arange(self.batch_mode)+self.start_id
                while all_finish == False and time.time() - time_start < self.timeout:
                    finish_list = []
                    for i in range(self.batch_mode):
                        finish_list.append(os.path.exists('PIPELINE_{}.DONE'.format(i+self.start_id)))
                    if np.all(finish_list):
                        all_finish = True
                    else:
                        if int(time.time() - time_start) % 600 == 0: 
                            print("{}: ".format(time.strftime("%H:%M:%S", time.localtime())))
                            print("wait for all jobs to finish")
                            print("jobs already finished: {}".format(np.array(job_ids)[np.array(finish_list)]))
                        time.sleep(5)
                
                if not all_finish:
                    raise RuntimeError("Timeout after {} seconds".format(self.timeout))
                    
                success_list = []
                for i in range(self.batch_mode):
                    with open('PIPELINE_{}.DONE'.format(i+self.start_id),'r') as f:
                        line = f.readline()
                    if 'SUCCESS' in line:
                        success_list.append(i+self.start_id)
                self.success_list = [str(x) for x in success_list]
                fsuccess = 'success_list_{}-{}.txt'.format(self.start_id,self.batch_mode+self.start_id-1)
                print("Writing out success list in {}".format(fsuccess))
                with open(fsuccess,'w') as fss:
                    print('\n'.join(self.success_list),file=fss)
                
                #combine validplots here
                
                if self.make_summary:
                    self.pipeinit = self.pipedef(skip_config=True)
                    self.gen_validplots()                 

                    
    def gen_validplots(self,validplot_pros = ['lcfit','biascorlcfit','getmu','cosmofit'],plotdir='figs'):
        pipeinit = self.pipeinit
        success_list = self.success_list        
#                 outsuffix = 'combined'
#         for p in pipeinit.pipepros:
#             if isinstance(pipeinit._get_pipepro_from_string(p),list) \
#               and np.any([(hasattr(pi,'validplots') and pi.validplots) for pi in pipeinit._get_pipepro_from_string(p)]):
#                 validplot_pros.append(p)                        
#             elif hasattr(pipeinit._get_pipepro_from_string(p),'validplots') and pipeinit._get_pipepro_from_string(p).validplots:
#                 validplot_pros.append(p)
        print("Making summary plots for successful jobs: {}".format(success_list))
        outsuffix="combined_{}-{}".format(min(success_list),max(success_list))
        if len(success_list)<2:
            print("Number of successful jobs < 2. Ending without making combined plots.")

        else:
            self.combine_validplot_inputs(pros=[x for x in pipeinit.pipepros if x in validplot_pros],
                                          nums=success_list,outsuffix=outsuffix)
            validplot_pros = [x for x in pipeinit.pipepros if x in validplot_pros]
            for p in validplot_pros:
                if p.startswith('lcfit'):
                    p = 'lcfitting'
                if p.startswith('cosmofit'):
                    inputfile_sum = '{}_{}.cospar'.format(p,outsuffix)
                else:
                    inputfile_sum = '{}_{}.FITRES'.format(p,outsuffix)
                print("Making summary plots for {}".format(p))
                if isinstance(pipeinit._get_pipepro_from_string(p),list): 
                    self.make_validplots_sum(p,inputfile_sum,plotdir,
                                             prefix_sum='sum_valid_{}-{}'.format(min(success_list),max(success_list)))  
                else:
                    self.make_validplots_sum(p,inputfile_sum,plotdir,
                                             prefix_sum='sum_valid_{}-{}'.format(min(success_list),max(success_list)))   
                    
def exit_handler(pipe,num):
    picklename = "pipeline_{}.{}.pickle".format(num,pipe.timestamp)
    pickle.dump(pipe, open(picklename, "wb" ))
    print("Wrote pipeline object as {}".format(picklename))      
    print("Exited")
    
    
def main(**kwargs):
    print("Starting pipeline run..")
    
    parser = argparse.ArgumentParser(description='Run SALT3 Pipe.')
    parser.add_argument('-c',dest='pipeinput',default=None,
                        help='pipeline input file')
    parser.add_argument('--mypipe',dest='mypipe',default=None,
                        help='define your own pipe in yourownfilename.py')
    parser.add_argument('--batch_mode',dest='batch_mode',default=0,type=int,
                        help='>0 to specify how many batch jobs to submit')
    parser.add_argument('--batch_script',dest='batch_script',default=None,
                        help='base batch submission script')
    parser.add_argument('--batch_job_mem_limit',dest='batch_job_mem_limit',default=None,
                        help='--mem-per-cpu in batch submission script')
    parser.add_argument('--start_id',dest='start_id',default=0,type=int,
                        help='starting id for naming suffix')
    parser.add_argument('--randseed',dest='randseed',default=None,type=int,
                        help='[internal use] specify randseed for single simulation')
    parser.add_argument('--fseeds',dest='fseeds',default=None,
                        help='provide a list of randseeds for multiple batch jobs')
    parser.add_argument('--num',dest='num',default=None,type=int,
                        help='[internal use] suffix for multiple batch jobs')   
    parser.add_argument('--norun',dest='norun', action='store_true',
                        help='set to only check configurations without launch jobs')   
    parser.add_argument('--debug',dest='debug', action='store_true',
                        help='use $MY_SALT3_DIR instead of installed runpipe for debugging')  
    parser.add_argument('--timeout',dest='timeout', default=36000,type=int,
                        help='running time limit (seconds). only valid if using batch mode')  
    parser.add_argument('--make_summary',dest='make_summary', action='store_true',
                        help='making summary plots for all the runs')  
    parser.add_argument('--validplots_only',dest='validplots_only', action='store_true',
                        help='making summary plots only without running the pipeline (success_list is needed)')  
    parser.add_argument('--success_list',dest='success_list', default=None,
                        help='success list file from a previous run for making validplots')  
    parser.add_argument('--load_from_pickle',dest='load_from_pickle',default=None,
                        help='load a saved pipeline object from a pickle file previously generated. only used for batch_mode=0')  
    parser.add_argument('--run_background_jobs',dest='run_background_jobs',action='store_true',
                        help='use this option to run individual jobs in background instead of submitting a parent slurm job') 
    parser.add_argument('--onlyrun',dest='onlyrun',default=None,
                        help='[comma separated list] only run specific stages') 
    parser.add_argument('--append_sim_genversion',dest='append_sim_genversion',default='',
                        help='append sim GENVERSION') 
    parser.add_argument('--update_randseed',dest='update_randseed',action='store_true',
                        help='update randseed')
    
    p = parser.parse_args()
       
    pipe = RunPipe(p.pipeinput,mypipe=p.mypipe,batch_mode=p.batch_mode,batch_script=p.batch_script,
                   start_id=p.start_id,randseed=p.randseed,fseeds=p.fseeds,num=p.num,norun=p.norun,
                   debug=p.debug,timeout=p.timeout,make_summary=p.make_summary,validplots_only=p.validplots_only,
                   success_list=p.success_list,load_from_pickle=p.load_from_pickle,run_background_jobs=p.run_background_jobs,
                   batch_job_mem_limit=p.batch_job_mem_limit,onlyrun=p.onlyrun,append_sim_genversion=p.append_sim_genversion,
                   update_randseed=p.update_randseed)
    
#     import pdb; pdb.set_trace()
    
    pipe.run()
    
    if hasattr(pipe,'pipe'):
        atexit.register(exit_handler,pipe.pipe,pipe.num)        
    
    
if __name__== "__main__":
    main()    
