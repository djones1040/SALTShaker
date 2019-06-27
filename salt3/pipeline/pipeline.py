##SALT3 pipeline
##sim -> training -> lcfitting -> salt2mu -> wfit

import subprocess
import configparser
import pandas as pd
import os
import numpy as np
import time
import glob

class SALT3pipe():
    def __init__(self,finput=None):
        self.finput = finput
        self.BYOSED = BYOSED()
        self.Simulation = Simulation()
        self.Training = Training()
        self.LCFitting = LCFitting()
        self.GetMu = GetMu()
        self.CosmoFit = CosmoFit()

    def gen_input(self):
        pass

    def configure(self):
        config = configparser.ConfigParser()
        config.read(self.finput)
        m2df = self._multivalues_to_df
        self.BYOSED.setkeys = m2df(config.get('byosed','set_key'),
                                   colnames=['section','key','value'])
        self.BYOSED.configure(baseinput=config.get('byosed','baseinput'),
                              setkeys=self.BYOSED.setkeys,
                              outname=config.get('byosed','outinput'))
        self.Simulation.setkeys = m2df(config.get('simulation','set_key'),
                                       colnames=['key','value'],
                                       stackvalues=True)
        self.Simulation.configure(baseinput=config.get('simulation','baseinput'),
                                  setkeys=self.Simulation.setkeys,
                                  outname=config.get('simulation','outinput'),
                                  pro=config.get('simulation','pro'),
                                  prooptions=config.get('simulation','prooptions'))
        self.Training.setkeys = m2df(config.get('training','set_key'),
                                     colnames=['section','key','value'])
        self.Training.configure(baseinput=config.get('training','baseinput'),
                                setkeys=self.Training.setkeys,
                                outname=config.get('training','outinput'),
                                pro=config.get('training','pro'),
                                proargs=config.get('training','proargs'),
                                prooptions=config.get('training','prooptions'))
        self.LCFitting.setkeys = m2df(config.get('lcfitting','set_key'),colnames=['section','key','value'])
        self.LCFitting.configure(baseinput=config.get('lcfitting','baseinput'),
                                 setkeys=self.LCFitting.setkeys,
                                 outname=config.get('lcfitting','outinput'),
                                 pro=config.get('lcfitting','pro'))
        self.GetMu.setkeys = m2df(config.get('getmu','set_key'),colnames=['key','value'])
        self.GetMu.configure(baseinput=config.get('getmu','baseinput'),
                             setkeys=self.GetMu.setkeys,
                             outname=config.get('getmu','outinput'),
                             pro=config.get('getmu','pro'),
                             prooptions=config.get('getmu','prooptions'))

        self.CosmoFit.configure(outname=config.get('cosmofit','outinput'),
                                pro=config.get('cosmofit','pro'),
                                prooptions=config.get('getmu','prooptions'))


    def run(self):
        self.Simulation.run()
        self.Training.run()
        self.LCFitting.run()
        self.GetMu.run()
        self.CosmoFit.run()
        

    def glue(self,pipepros=None,on='phot'):
        if pipepros is None:
            return
        elif not isinstance(pipepros,list) or len(pipepros) !=2:
            raise ValueError("pipepros must be list of length 2, {} of {} was given".format(type(pipepros),len(pipepros)))       
        
        pro1 = self._get_pipepro_from_string(pipepros[0])
        pro2 = self._get_pipepro_from_string(pipepros[1])
        pro1_out = pro1.glueto(pro2)
        if pipepros[1].lower().startswith('lcfit'):
            pro2_in = pro2._get_input_info().loc[on]
        else:
            pro2_in = pro2._get_input_info().loc[0]
        pro2_in['value'] = pro1_out
        setkeys = pd.DataFrame([pro2_in])
        print("Connecting ",pipepros)
        if not pipepros[1].lower().startswith('cosmofit'):
            pro2.configure(setkeys = setkeys,
                           pro=pro2.pro,
                           proargs=pro2.proargs,
                           baseinput=pro2.outname,
                           prooptions=pro2.prooptions,
                           outname=pro2.outname)
        else:
            pro2.configure(pro=pro2.pro,
                           prooptions=pro2.prooptions,
                           outname=setkeys['value'].values[0])            

    def _get_pipepro_from_string(self,pipepro_str):
        if pipepro_str.lower().startswith("sim"):
            pipepro = self.Simulation
        elif pipepro_str.lower().startswith("train"):
            pipepro = self.Training
        elif pipepro_str.lower().startswith("lcfit"):
            pipepro = self.LCFitting
        elif pipepro_str.lower().startswith("getmu"):
            pipepro = self.GetMu
        elif pipepro_str.lower().startswith("cosmofit"):
            pipepro = self.CosmoFit
        else:
            raise ValueError("Unknow pipeline procedure:",pipepro.strip())
        return pipepro

    def _multivalues_to_df(self,values,colnames=None,stackvalues=False):
        df = pd.DataFrame([s.split() for s in values.split('\n')])
        if df.empty:
            return None
        if stackvalues and df.shape[1] > len(colnames):
            numbercol = [colnames[-1]+'.'+str(i) for i in range(df.shape[1]-len(colnames)+1)]
            df.columns = colnames[0:-1] + numbercol
            lastcol = colnames[-1]
            df[lastcol] = df[[col for col in df.columns if col.startswith(lastcol)]].values.tolist()
            df = df.drop(numbercol,axis=1)
        else:
            df.columns = colnames
        return df


class PipeProcedure():
    def __init__(self):
        self.pro = None
        self.baseinput = None
        self.setkeys = None
        self.proargs = None
        self.outname = None

    def configure(self,pro=None,baseinput=None,setkeys=None,
                  proargs=None,prooptions=None,**kwargs):  
        self.pro = pro
        self.baseinput = baseinput
        self.setkeys = setkeys
        self.proargs = proargs
        self.prooptions = prooptions

        self.gen_input(outname=self.outname)

    def gen_input(self,outname=None):
        pass

    def run(self):
        arglist = [self.proargs] + [self.finput] +[self.prooptions]
        arglist = list(filter(None,arglist))
        args = []
        for arg in arglist:
            if arg is not None:
                for argitem in arg.split(' '):
                    args.append(argitem)
        _run_external_pro(self.pro, args)

    def _get_input_info(self):
        pass
    
    def _get_output_info(self):
        pass


class PyPipeProcedure(PipeProcedure):

    def gen_input(self,outname="pipeline_generalpy_input.input"):
        self.outname = outname
        self.finput,self.keys = _gen_general_python_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                outname=outname)


class BYOSED(PyPipeProcedure):

    def configure(self,baseinput=None,setkeys=None,
                  outname="pipeline_byosed_input.input",
                  bkp_orig_param=False,**kwargs):   
        self.outname = outname
        super().configure(pro=None,baseinput=baseinput,setkeys=setkeys)
        #rename current byosed param
        byosed_default = "BYOSED/BYOSED.params"
        if bkp_orig_param:
            byosed_rename = "{}.{}".format(byosed_default,int(time.time()))
            if os.path.isfile(byosed_default):
                shellcommand = "cp -p {} {}".format(byosed_default,byosed_rename) 
                shellrun = subprocess.run(list(shellcommand.split()))
                if shellrun.returncode != 0:
                    raise RuntimeError(shellrun.stdout)
                else:
                    print("{} copied as {}".format(byosed_default,byosed_rename))
        #copy new byosed input to BYOSED folder
        shellcommand = "cp -p {} {}".format(outname,byosed_default)
        shellrun = subprocess.run(list(shellcommand.split()))
        if shellrun.returncode != 0:
            raise RuntimeError(shellrun.stdout)
        else:
            print("{} is copied to {}".format(outname,byosed_default))

class Simulation(PipeProcedure):

    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  outname="pipeline_byosed_input.input",**kwargs):
        self.outname = outname
        self.prooptions = prooptions
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,prooptions=prooptions)

    def gen_input(self,outname="pipeline_sim_input.input"):
        self.outname = outname
        self.finput,self.keys = _gen_snana_sim_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                           outname=outname)
    
    def _get_output_info(self):
        df = {}
        key = 'GENVERSION'
        df['key'] = key
        df['value'] = self.keys[key]
        return pd.DataFrame([df])

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        outdir = self._get_output_info().value.values[0]
        outpath = os.path.join(os.environ['SNDATA_ROOT'],'SIM/',outdir)
        if os.path.isdir(outdir):
            res = outdir
        elif os.path.exists(outpath):
            res = outpath
        else:
            raise ValueError("Path does not exists",outdir,outpath)             
        if pipepro.lower().startswith('train'):
            return glob.glob(os.path.join(res,'*.LIST'))[0]
        elif pipepro.lower().startswith('lcfit'):
            simpath = os.path.join(os.environ['SNDATA_ROOT'],'SIM/')
            idx = res.find(simpath)
            if idx !=0:
                raise ValueError("photometry must be in $SNDATA_ROOT/SIM")
            else:
                return res[len(simpath):] 
        else:
            raise ValueError("sim can only glue to training or lcfitting")
        

class Training(PyPipeProcedure):

    def configure(self,pro=None,baseinput=None,setkeys=None,proargs=None,
                  prooptions=None,outname="pipeline_train_input.input",**kwargs):
        self.outname = outname
        self.proargs = proargs
        self.prooptions = prooptions
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          proargs=proargs,prooptions=prooptions)

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('lcfit'):
            outdir = self._get_output_info()
            ##copy necessary files to a model folder in SNDATA_ROOT
        else:
            raise ValueError("training can only glue to lcfit")

    def _get_input_info(self):
        df = {}
        section = 'iodata'
        key = 'snlist'
        df['section'] = section
        df['key'] = key
        df['value'] = self.keys[section][key]
        return pd.DataFrame([df])
    
    def _get_output_info(self):
        df = {}
        section = 'iodata'
        key = 'outputdir'
        df['section'] = section
        df['key'] = key
        df['value'] = self.keys[section][key]
        return pd.DataFrame([df])

class LCFitting(PipeProcedure):

    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  outname="pipeline_lcfit_input.input",**kwargs):
        self.outname = outname
        self.prooptions = prooptions
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,prooptions=prooptions)

    def gen_input(self,outname="pipeline_lcfit_input.input"):
        self.outname = outname
        self.finput,self.keys = _gen_snana_fit_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                     outname=outname)


    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('getmu'):
            outprefix = self._get_output_info().value.values[0]
            return glob.glob((str(outprefix)+'*.FITRES.TEXT'))[0]
        else:
            raise ValueError("lcfitting can only glue to getmu")

    def _get_input_info(self):
        df = {}
        section = 'SNLCINP'
        key = 'VERSION_PHOTOMETRY'
        df['section'] = section
        df['key'] = key
        df['value'] = self.keys[section][key]
        df['type'] = 'phot'
        df2 = {}
        section2 = 'FITINP'
        key2 = 'FITMODEL_NAME'
        df2['section'] = section2
        df2['key'] = key2
        df2['value'] = self.keys[section2][key2]
        df2['type'] = 'model'
        return pd.DataFrame([df,df2]).set_index('type')

    def _get_output_info(self):
        df = {}
        section = 'SNLCINP'
        key = 'TEXTFILE_PREFIX'
        df['section'] = section
        df['key'] = key
        df['value'] = self.keys[section][key]
        return pd.DataFrame([df])
 
class GetMu(PipeProcedure):
    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  outname="pipeline_getmu_input.input",**kwargs):
        self.outname = outname
        self.prooptions = prooptions
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,prooptions=prooptions)

    def gen_input(self,outname="pipeline_getmu_input.input"):
        self.outname = outname
        self.finput,self.keys = _gen_general_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                          outname=outname)        
    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('cosmofit'):
            return self._get_output_info()['value'].values[0]
        else:
            raise ValueError("getmu can only glue to cosmofit")
   
    def _get_input_info(self):
        df = {}
        key = 'file'
        df['key'] = key
        df['value'] = self.keys[key]
        return pd.DataFrame([df])

    def _get_output_info(self):
        df = {}
        key = 'prefix'
        df['key'] = key
        df['value'] = self.keys[key].strip()+'.M0DIF'
        return pd.DataFrame([df])

class CosmoFit(PipeProcedure):
    def configure(self,setkeys=None,pro=None,outname=None,prooptions=None,**kwargs):
        if setkeys is not None:
            outname = setkeys.value.values[0]
        self.prooptions = prooptions
        self.finput = outname

        super().configure(pro=pro,outname=outname,prooptions=prooptions)

    def _get_input_info(self):
        df = {}
        df['value'] = 'test'
        return pd.DataFrame([df])

def _run_external_pro(pro,args):

    if isinstance(args, str):
        args = [args]

    print("Running",' '.join([pro] + args))
    res = subprocess.run(args = list([pro] + args))
    
    if res.returncode == 0:
        print("{} finished successfully.".format(pro.strip()))
    else:
        raise ValueError("Something went wrong..") ##possible to pass the error msg from the program?

    return

def _gen_general_python_input(basefilename=None,setkeys=None,
                              outname=None):

    config = configparser.ConfigParser()
    if not os.path.isfile(basefilename):
        raise ValueError("File does not exist",basefilename)
    if not os.path.exists(os.path.dirname(outname)):
        os.makedirs(os.path.dirname(outname))
    
    config.read(basefilename)
    if setkeys is None:
        print("No modification on the input file, keeping {} as input".format(basefilename))
    else:
        setkeys = pd.DataFrame(setkeys)
        for index, row in setkeys.iterrows():
            sec = row['section']
            key = row['key']
            v = row['value']
            if not sec in config.sections():
                config.add_section(sec)
            print("Adding/modifying key {}={} in [{}]".format(key,v,sec))
            config[sec][key] = v
        with open(outname, 'w') as f:
            config.write(f)

        print("input file saved as:",outname)
    return outname,config

def _gen_snana_sim_input(basefilename=None,setkeys=None,
                         outname=None):

    #TODO:
    #read in kwlist from standard snana kw list
    #determine if the kw is in the list

    #read in a default input file
    #add/edit the kw

    if not os.path.isfile(basefilename):
        raise ValueError("basefilename cannot be None")
    print("Load base sim input file..",basefilename)
    basefile = open(basefilename,"r")
    lines = basefile.readlines()
    basekws = []
    
    if setkeys is None:
        print("No modification on the input file, keeping {} as input".format(basefilename))
        config = {}
        for i,line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                kwline = line.split(":")
                kw = kwline[0]
                config[kw] = kwline[1].strip()
    else:
        setkeys = pd.DataFrame(setkeys)
        if np.any(setkeys.key.duplicated()):
            raise ValueError("Check for duplicated entries for",setkeys.keys[setkeys.keys.duplicated()].unique())

        config = {}
        for i,line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                kwline = line.split(":")
                kw = kwline[0]
                basekws.append(kw)
                if kw in setkeys.key.values:
                    keyvalue = setkeys[setkeys.key==kw].value.values[0]
                    kwline[1] = ' '.join(list(filter(None,keyvalue)))+'\n'
                    print("Setting {} = {}".format(kw,kwline[1].strip()))
                lines[i] = ": ".join(kwline)
                config[kw] = kwline[1].strip()

        outfile = open(outname,"w")
        for line in lines:
            outfile.write(line)

        for key,value in zip(setkeys.key,setkeys.value):
            if not key in basekws:
                print("Adding key {}={}".format(key,value))
                newline = key+": "+' '.join(list(filter(None,value)))
                outfile.write(newline)
                config[key] = value

        print("Write sim input to file:",outname)

    return outname,config


def _gen_snana_fit_input(basefilename=None,setkeys=None,
                              outname=None):
    import f90nml
    nml = f90nml.read(basefilename)

    if setkeys is None:
        print("No modification on the input file, keeping {} as input".format(basefilename))
    else:
        for index, row in setkeys.iterrows():
            sec = row['section']
            key = row['key']
            v = row['value']
            if not sec.lower() in nml.keys():
                raise ValueError("No section named",sec)
            print("Adding/modifying key {}={} in &{}".format(key,v,sec))
            nml[sec][key] = v

        print("Write fit input to file:",outname)
        _write_nml_to_file(nml,outname)
    return outname,nml

def _gen_general_input(basefilename=None,setkeys=None,outname=None):
    config = _read_simple_config_file(basefilename)
    if setkeys is None:
        print("No modification on the input file, keeping {} as input".format(basefilename))
    else:
        for index, row in setkeys.iterrows():
            key = row['key']
            v = row['value']
            print("Adding/modifying key {}={}".format(key,v))
            config[key] = v

        print("input file saved as:",outname)
        _write_simple_config_file(config,outname)

    return outname,config

def _read_simple_config_file(filename,sep='='):
    config = {}
    f = open(filename,"r")
    lines = f.readlines()
    for line in lines:    
        if sep in line and not line.strip().startswith("#"):
            key,value = line.split(sep)
            config[key] = value.rstrip()
    return config

def _write_simple_config_file(config,filename,sep='='):
    outfile = open(filename,"w")
    for key in config.keys():
        value = config[key]
        outfile.write("{}={}\n".format(key,value))
    return

def _write_nml_to_file(nml,filename):
    outfile = open(filename,"w")
    for key in nml.keys():
        outfile.write('&'+key.upper())
        outfile.write('\n')
        for key2 in nml[key].keys():
            value = nml[key][key2]
            if isinstance(value,str):
                value = "'{}'".format(value)
            elif isinstance(value,list):
                value = ','.join([str(x) for x in value if not isinstance(x,str)])
            outfile.write("  {} = {}".format(key2.upper(),value))
            outfile.write("\n")
        outfile.write('&END\n\n')
    return
