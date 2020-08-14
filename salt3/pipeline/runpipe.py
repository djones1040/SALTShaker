import sys
import pandas as pd
from salt3.pipeline import pipeline
from salt3.pipeline.pipeline import *
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
from salt3.pipeline.pipeline import SALT3pipe
from salt3.pipeline.validplot import ValidPlots,lcfitting_validplots,getmu_validplots,cosmofit_validplots
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
                 randseed=None,fseeds=None,num=None,norun=None,debug=False,timeout=None):
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
 
    def __DefaultPipe(self):
        pipe = SALT3pipe(finput=self.pipeinput)
        pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','lcfit'])
        pipe.configure()
        pipe.glue(['sim','lcfit'],on='phot')
        return pipe
        
    def __MyPipe(self): 
        MyPipe = _MyPipe(self.mypipe)
        pipe = MyPipe(finput=self.pipeinput)
        return pipe

    def _add_suffix(self,pro,keylist,suffix,section=None):
        df = pd.DataFrame() 
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
                val_new = '{}_{:03d}'.format(val_old.strip(),suffix)
                df = pd.concat([df,pd.DataFrame([{'section':sec,'key':keystr,'value':val_new}])])
        return df
    
    def _reconfig_w_suffix(self,proname,df,suffix,**kwargs):
        outname_orig = proname.outname
        proname.outname = '{}_{:03d}'.format(proname.outname,self.num)
        proname.configure(pro=proname.pro,baseinput=outname_orig,setkeys=df,prooptions=proname.prooptions,
                          batch=proname.batch,validplots=proname.validplots,outname=proname.outname,
                          proargs=proname.proargs,plotdir=proname.plotdir,**kwargs)  
    
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
        cmd = 'sntable_cat.py -i {} -o {}_{}.FITRES'.format(','.join(files),pro,outsuffix)
        print(cmd)
        cmdrun = subprocess.run(args=shlex.split(cmd),capture_output=True)
        if cmdrun.returncode != 0:
            raise RuntimeError("sntable_cat.py failed")
        
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
        
        if self.batch_mode == 0:
            self.pipe = self.pipedef()
            if self.randseed is not None:
                if not any([p.startswith('sim') or p.startswith('biascorsim') for p in self.pipe.pipepros]):
                    raise RuntimeError("randseed was given but sim/biascorsim is not defined in the pipeline")
                print('randseed = {}'.format(self.randseed)) 
                    
                if self.num is not None:
                    if any([p.startswith('sim') for p in self.pipe.pipepros]):
                        df_sim = self._add_suffix(self.pipe.Simulation,['GENVERSION','GENPREFIX'],self.num)
                        self._reconfig_w_suffix(self.pipe.Simulation,df_sim,self.num)
                    if any([p.startswith('biascorsim') for p in self.pipe.pipepros]):
                        df_sim_biascor = self._add_suffix(self.pipe.BiascorSim,['GENVERSION','GENPREFIX'],self.num)
                        self._reconfig_w_suffix(self.pipe.BiascorSim,df_sim_biascor,self.num)
                    if any([p.startswith('train') for p in self.pipe.pipepros]): 
                        df_train = self._add_suffix(self.pipe.Training,['outputdir'],self.num,section=['iodata'])
                        self._reconfig_w_suffix(self.pipe.Training,df_train,self.num)
                        if ['sim','train'] in self.pipe.gluepairs: 
                            self.pipe.glue(['sim','train'])
                    if any([p.startswith('lcfit') for p in self.pipe.pipepros]):    
                        for i in range(self.pipe.n_lcfit):
                            df_lcfit = self._add_suffix(self.pipe.LCFitting[i],['outdir'],self.num,section=['header'])
                            done_file = "{}_{:03d}".format(self.pipe.LCFitting[i].done_file.strip(),self.num)
                            self._reconfig_w_suffix(self.pipe.LCFitting[i],df_lcfit,self.num,done_file=done_file)
                        if ['sim','lcfit'] in self.pipe.gluepairs:
                            self.pipe.glue(['sim','lcfit'],on='phot')
                        if ['train','lcfit'] in self.pipe.gluepairs:
                            self.pipe.glue(['train','lcfit'],on='model')
#                             self._reconfig_w_suffix(self.pipe.LCFitting[i],None,self.num)
                    if any([p.startswith('biascorlcfit') for p in self.pipe.pipepros]):       
                        for i in range(self.pipe.n_biascorlcfit):
                            df_lcfit_biascor = self._add_suffix(self.pipe.BiascorLCFit[i],['outdir'],self.num,section=['header'])
                            done_file = "{}_{:03d}".format(self.pipe.BiascorLCFit[i].done_file.strip(),self.num)
                            self._reconfig_w_suffix(self.pipe.BiascorLCFit[i],df_lcfit_biascor,self.num,done_file=done_file)
#                             self._reconfig_w_suffix(self.pipe.BiascorLCFit[i],None,self.num)
                        if ['biascorsim','biascorlcfit'] in self.pipe.gluepairs:
                            self.pipe.glue(['biascorsim','biascorlcfit'],on='phot')
                        if ['train','biascorlcfit'] in self.pipe.gluepairs:
                            self.pipe.glue(['train','biascorlcfit'],on='model')
                    if any([p.startswith('getmu') for p in self.pipe.pipepros]): 
                        df_getmu = self._add_suffix(self.pipe.GetMu,['OUTDIR_OVERRIDE'],self.num)
                        done_file = "{}_{:03d}".format(self.pipe.GetMu.done_file.strip(),self.num)
                        self._reconfig_w_suffix(self.pipe.GetMu,df_getmu,self.num,done_file=done_file)
                        if ['lcfit','getmu'] in self.pipe.gluepairs:
                            self.pipe.glue(['lcfit','getmu'])
                        if ['biascorlcfit','getmu'] in self.pipe.gluepairs:
                            self.pipe.glue(['biascorlcfit','getmu'])
                        if ['getmu','cosmofit'] in self.pipe.gluepairs:
                            self.pipe.glue(['getmu','cosmofit'])
            
                if any([p.startswith('sim') for p in self.pipe.pipepros]):
                    sim = self.pipe.Simulation
                    randseed_old = sim.keys['RANSEED_REPEAT']
                    randseed_new = [randseed_old.split(' ')[0],str(self.randseed)]
                    df = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                    sim.configure(pro=sim.pro,baseinput=sim.outname,setkeys=df,prooptions=sim.prooptions,
                                  batch=sim.batch,validplots=sim.validplots,outname=sim.outname)    
                if any([p.startswith('biascorsim') for p in self.pipe.pipepros]):
                    sim_biascor = self.pipe.BiascorSim
                    randseed_old = sim_biascor.keys['RANSEED_REPEAT']
                    randseed_new = [randseed_old.split(' ')[0],str(self.randseed+10)]
                    df_biascor = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                    sim_biascor.configure(pro=sim_biascor.pro,baseinput=sim_biascor.outname,setkeys=df_biascor,prooptions=sim_biascor.prooptions,
                                          batch=sim_biascor.batch,validplots=sim_biascor.validplots,outname=sim_biascor.outname)                
            
            if not self.norun:
                #remove success files from previous runs
                if os.path.exists('PIPELINE_{}.DONE'.format(self.num)):
                    print("Removing old .DONE files")
                    os.system('rm PIPELINE_{}.DONE'.format(self.num))
                    
                self.pipe.run()
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
                                    
          
            """
            below is test
            """      
            
#             os.system('touch PIPELINE_{}.SUCCESS'.format(self.num))

#             for proname in ['lcfit','biascorlcfit','getmu','cosmofit']:
#                 if proname in self.pipe.pipepros:
#                     pro = self.pipe._get_pipepro_from_string(proname)
#                     if not os.path.isdir('misc'):
#                         os.makedirs('misc')
#                     if isinstance(pro,list):
#                         for i in range(len(pro)):
#                             if pro[i].validplots:
#                                 if proname.startswith('lcfit'):
#                                     proname = 'lcfitting'
#                                 pro[i].get_validplot_inputs(outname='misc/{}_validplot_info_{}_{}.txt'.format(proname,self.num,i))
#                     else:
#                         if pro.validplots:
#                             pro.get_validplot_inputs(outname='misc/{}_validplot_info_{}.txt'.format(proname,self.num))
                            
                               
        else:
            if self.batch_script is None:
                raise RuntimeError("batch script is None")
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
            pypro = os.path.expandvars('$MY_SALT3_DIR/SALT3/salt3/pipeline/runpipe.py')
            if self.debug:
                pycommand_base = 'python {} -c {} --mypipe {} --batch_mode 0'.format(pypro,self.pipeinput,self.mypipe)
            else:
                pycommand_base = 'runpipe -c {} --mypipe {} --batch_mode 0'.format(self.pipeinput,self.mypipe)
            for i in range(self.batch_mode):
                pycommand = pycommand_base + ' --randseed {} --num {}'.format(self.randseeds[i],i+self.start_id)
                if self.norun:
                    pycommand += ' --norun'
                cwd = os.getcwd()
                outfname = os.path.join(cwd,'test_pipeline_batch_script_{:03d}'.format(i+self.start_id))
                outf = open(outfname,'w')
                outf.write(lines)
                outf.write('\n')
                outf.write(pycommand)
                outf.write('\n')
                outf.close()
                print('Submitting batch job...')
                shellcommand = "sbatch {}".format(outfname) 
                print(shellcommand)
                
                shellrun = subprocess.run(args=shlex.split(shellcommand))
                
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
                    
                #combine validplots here
                pipe = self.pipedef()
                validplot_pros = []
                outsuffix = 'combined'
                for p in pipe.pipepros:
                    if isinstance(pipe._get_pipepro_from_string(p),list) \
                      and np.any([(hasattr(pi,'validplots') and pi.validplots) for pi in pipe._get_pipepro_from_string(p)]):
                        validplot_pros.append(p)                        
                    elif hasattr(pipe._get_pipepro_from_string(p),'validplots') and pipe._get_pipepro_from_string(p).validplots:
                        validplot_pros.append(p)
                print("Making summary plots for successful jobs: ".format(success_list))
                self.combine_validplot_inputs(pros=[x for x in pipe.pipepros if x in validplot_pros],
                                              nums=success_list,outsuffix="{}_{}-{}".format(outsuffix,min(success_list),max(success_list)))
                for p in validplot_pros:
                    if p.startswith('lcfit'):
                        p = 'lcfitting'
                    if p.startswith('cosmofit'):
                        inputfile_sum = '{}_{}.cospar'.format(p,outsuffix)
                    else:
                        inputfile_sum = '{}_{}.FITRES'.format(p,outsuffix)
                    print("Making summary plots for {}".format(p))
                    if isinstance(pipe._get_pipepro_from_string(p),list): 
                        self.make_validplots_sum(p,inputfile_sum,pipe._get_pipepro_from_string(p)[0].plotdir,
                                                 prefix_sum='sum_valid_{}-{}'.format(min(success_list),max(success_list)))  
                    else:
                        self.make_validplots_sum(p,inputfile_sum,pipe._get_pipepro_from_string(p).plotdir,
                                                 prefix_sum='sum_valid_{}-{}'.format(min(success_list),max(success_list)))  
                    
            """
            below is test
            """
#             #wait for all jobs to finish
#             all_finish = False
#             while all_finish == False:
#                 finish_list = []
#                 for i in range(self.batch_mode):
#                     finish_list.append(os.path.exists('PIPELINE_{}.SUCCESS'.format(i+self.start_id)))
#                 if np.all(finish_list):
#                     all_finish = True
#                 else:
#                     time.sleep(5)
#             #combine validplots here
#             pipe = self.pipedef()
#             validplot_pros = []
#             outsuffix = 'combined'
#             for p in pipe.pipepros:
#                 if isinstance(pipe._get_pipepro_from_string(p),list) \
#                   and np.any([(hasattr(pi,'validplots') and pi.validplots) for pi in pipe._get_pipepro_from_string(p)]):
#                     validplot_pros.append(p)                        
#                 elif hasattr(pipe._get_pipepro_from_string(p),'validplots') and pipe._get_pipepro_from_string(p).validplots:
#                     validplot_pros.append(p)
#             self.combine_validplot_inputs(pros=[x for x in pipe.pipepros if x in validplot_pros],
#                                           nums=np.arange(self.batch_mode)+self.start_id,outsuffix=outsuffix)
#             for p in validplot_pros:
#                 if p.startswith('lcfit'):
#                     p = 'lcfitting'
#                 if p.startswith('cosmofit'):
#                     inputfile_sum = '{}_{}.cospar'.format(p,outsuffix)
#                 else:
#                     inputfile_sum = '{}_{}.FITRES'.format(p,outsuffix)
#                 print("Making summary plots for {}".format(p))
#                 if isinstance(pipe._get_pipepro_from_string(p),list): 
#                     self.make_validplots_sum(p,inputfile_sum,pipe._get_pipepro_from_string(p)[0].plotdir,'sum_valid_{}'.format(p))  
#                 else:
#                     self.make_validplots_sum(p,inputfile_sum,pipe._get_pipepro_from_string(p).plotdir,'sum_valid_{}'.format(p))  
                    
        
def main(**kwargs):
    parser = argparse.ArgumentParser(description='Run SALT3 Pipe.')
    parser.add_argument('-c',dest='pipeinput',default=None,
                        help='pipeline input file')
    parser.add_argument('--mypipe',dest='mypipe',default=None,
                        help='define your own pipe in yourownfilename.py')
    parser.add_argument('--batch_mode',dest='batch_mode',default=0,type=int,
                        help='>0 to specify how many batch jobs to submit')
    parser.add_argument('--batch_script',dest='batch_script',default=None,
                        help='base batch submission script')
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
    
    p = parser.parse_args()
    
    pipe = RunPipe(p.pipeinput,mypipe=p.mypipe,batch_mode=p.batch_mode,batch_script=p.batch_script,
                   start_id=p.start_id,randseed=p.randseed,fseeds=p.fseeds,num=p.num,norun=p.norun,
                   debug=p.debug,timeout=p.timeout)
    pipe.run()
    
    
if __name__== "__main__":
    main()    
