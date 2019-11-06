import sys
import pandas as pd
import pipeline
from pipeline import *
import subprocess
import argparse
import os
import shlex

def MyPipe():
    # write your own pipeline here
    return

class RunPipe():
    def __init__(self, pipeinput, mypipe=False, batch_mode=False,batch_script=None):
        if not mypipe:
            self.pipe = self.__DefaultPipe
        else:
            self.pipe = self.__MyPipe
        self.batch_mode = batch_mode
        self.batch_script = batch_script
        self.pipeinput = pipeinput
        self.mypipe = mypipe
    
    def run(self):
        if not self.batch_mode:
            self.pipe()
        else:
            if self.batch_script is None:
                raise RuntimeError("batch script is None")
            f = open(self.batch_script,'r')
            lines = f.read()
            f.close()
            pypro = os.path.expandvars('$WFIRST_ROOT/SALT3/salt3/pipeline/runpipe.py')
            pycommand = 'python {} -c {} --mypipe {} --batch_mode 0'.format(pypro,self.pipeinput,self.mypipe)
            cwd = os.getcwd()
            outfname = os.path.join(cwd,'test_pipeline_batch_script')
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

    def __DefaultPipe(self):
        pipe = SALT3pipe(finput=self.pipeinput)
        pipe.build(data=False,mode='customize',onlyrun=['byosed','sim','lcfit'])
        pipe.configure()
        pipe.glue(['sim','lcfit'],on='phot')
        pipe.run()

    def __MyPipe(self):       
        MyPipe()
        
def main(**kwargs):
    parser = argparse.ArgumentParser(description='Run SALT3 Pipe.')
    parser.add_argument('-c',dest='pipeinput',default=None)
    parser.add_argument('--mypipe',dest='mypipe',default=0,type=int)
    parser.add_argument('--batch_mode',dest='batch_mode',default=0,type=int)
    parser.add_argument('--batch_script',dest='batch_script',default=None)
    
    p = parser.parse_args()
    
    pipe = RunPipe(p.pipeinput,mypipe=p.mypipe,batch_mode=p.batch_mode,batch_script=p.batch_script)
    pipe.run()
    
    
if __name__== "__main__":
    main()    