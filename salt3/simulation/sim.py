##generate SNANA input file with given parameters
##run SNANA simulations and wait for the SNANA job to finish
##format the output to be integrated in the training pipeline

import subprocess

def gen_snana_input(basefilename="siminputs/sim_PS1_IA.INPUT",**kwargs):

    #TODO:
    #read in kwlist from standard snana kw list
    #determine if the kw is in the list

    #read in a default input file
    #add/edit the kw

    print("Load base sim input file..")
    basefile = open(basefilename,"r")
    lines = basefile.readlines()
    basekws = []

    for i,line in enumerate(lines):
        kwline = line.split(":")
        kw = kwline[0]
        basekws.append(kw)
        if kw in list(kwargs):
            print("Setting {} = {}".format(kw,kwargs[kw]))
            kwline[1] = kwargs[kw]
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

    return


def run_snana_sim(simpro="snlc_sim.exe",siminput="siminputs/sim_PS1_IA.INPUT"):

    ## run the simulation from shell
    res = subprocess.run([simpro, siminput])

    if res.returncode == 0:
        print("SNANA sim finished successfully.")
    else:
        raise ValueError("Something went wrong..") ##possible to pass the error msg from the program?

    return

def format_output():

    return




