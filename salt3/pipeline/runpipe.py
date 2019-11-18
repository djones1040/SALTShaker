import sys
import pandas as pd
import pipeline
from pipeline import *
import subprocess
import argparse
import os
import shlex
import random

def MyPipe(finput):
    # write your own pipeline here        
    pipe = SALT3pipe(finput)
    pipe.build(data=False,mode='customize',onlyrun=['byosed','sim'])
    pipe.configure()
    return pipe

class RunPipe():
    def __init__(self, pipeinput, mypipe=False, batch_mode=False,batch_script=None,randseed=None,
                 fseeds=None,num=None,norun=None):
        if not mypipe:
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
    
    def run(self):
        if self.batch_mode == 0:
            self.pipe = self.pipedef()
            if self.randseed is not None:
                print('randseed = {}'.format(self.randseed))
                sim = self.pipe.Simulation
                randseed_old = sim.keys['RANSEED_REPEAT']
                randseed_new = [randseed_old.split(' ')[0],str(self.randseed)]
                df = pd.DataFrame([{'key':'RANSEED_REPEAT','value':randseed_new}])
                if self.num is not None:
                    genversion_old = sim.keys['GENVERSION']
                    genversion_new = '{}_{:03d}'.format(genversion_old.strip(),self.num)
                    df = pd.concat([df,pd.DataFrame([{'key':'GENVERSION','value':genversion_new}])])
                    genprefix_old = sim.keys['GENPREFIX']
                    genprefix_new = '{}_{:03d}'.format(genprefix_old.strip(),self.num)
                    df = pd.concat([df,pd.DataFrame([{'key':'GENPREFIX','value':genprefix_new}])])
                    sim.outname = '{}_{:03d}'.format(sim.outname,self.num)
                sim.configure(pro=sim.pro,baseinput=sim.baseinput,setkeys=df,prooptions=sim.prooptions,
                              batch=sim.batch,validplots=sim.validplots,outname=sim.outname)      
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

    def __DefaultPipe(self):
        pipe = SALT3pipe(finput=self.pipeinput)
        pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','lcfit'])
        pipe.configure()
        pipe.glue(['sim','lcfit'],on='phot')
        return pipe
        
    def __MyPipe(self):       
        pipe = MyPipe(finput=self.pipeinput)
        return pipe
        
def main(**kwargs):
    parser = argparse.ArgumentParser(description='Run SALT3 Pipe.')
    parser.add_argument('-c',dest='pipeinput',default=None)
    parser.add_argument('--mypipe',dest='mypipe',default=0,type=int)
    parser.add_argument('--batch_mode',dest='batch_mode',default=0,type=int)
    parser.add_argument('--batch_script',dest='batch_script',default=None)
    parser.add_argument('--randseed',dest='randseed',default=None,type=int)
    parser.add_argument('--fseeds',dest='fseeds',default=None)
    parser.add_argument('--num',dest='num',default=None,type=int)   
    parser.add_argument('--norun',dest='norun', action='store_true')   
    
    p = parser.parse_args()
    
    pipe = RunPipe(p.pipeinput,mypipe=p.mypipe,batch_mode=p.batch_mode,batch_script=p.batch_script,
                   randseed=p.randseed,fseeds=p.fseeds,num=p.num,norun=p.norun)
    pipe.run()
    
    
if __name__== "__main__":
    main()    