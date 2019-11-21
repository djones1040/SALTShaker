import sys
import pandas as pd
import pipeline
from pipeline import *
import subprocess
import argparse
import os
import shlex
import random

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
    import importlib
    mymod = importlib.import_module(mypipe.split('.py')[0])
    print("Using user defined pipeline: {}.py".format(mypipe.split('.py')[0]))
    return mymod.MyPipe


class RunPipe():
    def __init__(self, pipeinput, mypipe=False, batch_mode=False,batch_script=None,randseed=None,
                 fseeds=None,num=None,norun=None):
        if mypipe is None:
            self.pipedef = self.__DefaultPipe
        else:
            self.pipedef = self.__MyPipe
        self.batch_mode = batch_mode
        self.batch_script = batch_script
        self.pipeinput = pipeinput
        self.mypipe = mypipe
        self.randseed = randseed
        self.fseeds = fseeds
        self.num = num
        self.norun = norun
 
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
            if section is None:
                val_old = pro.keys[key]
                sec = None
            else:
                sec = section[i]
                val_old = pro.keys[sec][key]
            val_new = '{}_{:03d}'.format(val_old.strip(),suffix)
            df = pd.concat([df,pd.DataFrame([{'section':sec,'key':key,'value':val_new}])])
        return df
    
    def _reconfig_w_suffix(self,proname,df,suffix):
        proname.outname = '{}_{:03d}'.format(proname.outname,self.num)
        proname.configure(pro=proname.pro,baseinput=proname.baseinput,setkeys=df,prooptions=proname.prooptions,
                        batch=proname.batch,validplots=proname.validplots,outname=proname.outname,proargs=proname.proargs)  
    
    def run(self):
        if self.batch_mode == 0:
            self.pipe = self.pipedef()
            if self.randseed is not None:
                if not any([p.startswith('sim') for p in self.pipe.pipepros]):
                    raise RuntimeError("randseed was given but sim is not defined in the pipeline")
                print('randseed = {}'.format(self.randseed))                   
                sim = self.pipe.Simulation
                randseed_old = sim.keys['RANSEED_REPEAT']
                randseed_new = [randseed_old.split(' ')[0],str(self.randseed)]
                df = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                sim.configure(pro=sim.pro,baseinput=sim.baseinput,setkeys=df,prooptions=sim.prooptions,
                              batch=sim.batch,validplots=sim.validplots,outname=sim.outname)    
                if self.num is not None:
                    df_sim = self._add_suffix(sim,['GENVERSION','GENPREFIX'],self.num)
                    self._reconfig_w_suffix(sim,df_sim,self.num)
                    if any([p.startswith('train') for p in self.pipe.pipepros]):    
                        self.pipe.glue(['sim','train'])
                    if any([p.startswith('lcfit') for p in self.pipe.pipepros]):                           
                        self.pipe.glue(['sim','lcfit'],on='phot')
     
            if not self.norun:
                self.pipe.run()
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
            pycommand_base = 'python {} -c {} --mypipe {} --batch_mode 0'.format(pypro,self.pipeinput,self.mypipe)
            for i in range(self.batch_mode):
                pycommand = pycommand_base + ' --randseed {} --num {}'.format(self.randseeds[i],i)
                cwd = os.getcwd()
                outfname = os.path.join(cwd,'test_pipeline_batch_script_{:03d}'.format(i))
                outf = open(outfname,'w')
                outf.write(lines)
                outf.write('\n')
                outf.write(pycommand)
                outf.write('\n')
                outf.close()
                print('Submitting batch job...')
                shellcommand = "sbatch {}".format(outfname) 
                print(shellcommand)
                if not self.norun:
                    shellrun = subprocess.run(args=shlex.split(shellcommand))
        
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
    parser.add_argument('--randseed',dest='randseed',default=None,type=int,
                        help='[internal use] specify randseed for single simulation')
    parser.add_argument('--fseeds',dest='fseeds',default=None,
                        help='provide a list of randseeds for multiple batch jobs')
    parser.add_argument('--num',dest='num',default=None,type=int,
                        help='[internal use] suffix for multiple batch jobs')   
    parser.add_argument('--norun',dest='norun', action='store_true',
                        help='set to only check configurations without launch jobs')   
    
    p = parser.parse_args()
    
    pipe = RunPipe(p.pipeinput,mypipe=p.mypipe,batch_mode=p.batch_mode,batch_script=p.batch_script,
                   randseed=p.randseed,fseeds=p.fseeds,num=p.num,norun=p.norun)
    pipe.run()
    
    
if __name__== "__main__":
    main()    
