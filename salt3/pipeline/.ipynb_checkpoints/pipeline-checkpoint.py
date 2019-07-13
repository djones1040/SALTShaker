##SALT3 pipeline
##sim -> training -> fitting

import subprocess

class SALT3pipe():
    def __init__(self,finput=None):
        self.finput = finput

    def modify_input():
        pass

    def configure():
        self.Simulation = Simulation()
        self.Training = Training()
        self.Fitting = Fitting()

    def run():
        self.Simulation.run()
        self.Training.run()
        self.Fitting.run()


class PipeProcedure():
    def __init__(self):
        pass

    def configure(self,pro=None,baseinput=None):        
        self.pro = pro
        self.baseinput = baseinput

    def modify_input(self):
        self.finput = self.baseinput

    def run(self):
        _run_external_pro(self.pro,self.finput)


class Simulation(PipeProcedure):
    def __init__(self):
        pass

    def modify_input(self):
        self.finput = gen_snana_sim_input(self.baseinput)

class Training(PipeProcedure):
    def __init__(self):
        pass

class LCFitting(PipeProcedure):
    def __init__(self):
        pass
    
def _run_external_pro(pro,args):

    if isinstance(args, str):
        args = [args]

    res = subprocess.run([pro] + args)

    if res.returncode == 0:
        print("{} finished successfully.".format(pro.strip()))
    else:
        raise ValueError("Something went wrong..") ##possible to pass the error msg from the program?

    return

def gen_snana_sim_input(basefilename="siminputs/sim_PS1_IA.INPUT",setkeys=None,**kwargs):

    #TODO:
    #read in kwlist from standard snana kw list
    #determine if the kw is in the list

    #read in a default input file
    #add/edit the kw

    if type(setkeys) 

    print("Load base sim input file..")
    basefile = open(basefilename,"r")
    lines = basefile.readlines()
    basekws = []

    for i,line in enumerate(lines):
        kwline = line.split(":")
        kw = kwline[0]
        basekws.append(kw)
        if kw in setkeys.keys():
            print("Setting {} = {}".format(kw,kwargs[kw]))
            kwline[1] = setkeys[kw]
        lines[i] = ": ".join(kwline)+"\n"

    outname = "sim_input_test.input"
    outfile = open(outname,"w")
    for line in lines:
        outfile.write(line)

    for key,value in kwargs.items():
        if not key in basekws:
            print("Adding key {}={}".format(key,value))
            newline = key+": "+value+"\n"
            outfile.write(newline)

    print("Write sim input to file:",outname)

    return outname