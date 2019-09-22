##SALT3 pipeline
##sim -> training -> lcfitting -> salt2mu -> wfit

import subprocess
import configparser
import pandas as pd
import os
import numpy as np
import time
import glob
import warnings
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
    if not finput.replace(' ','').startswith('/') and not finput.startswith('$') and \
       '/' in finput: finput = '%s/%s'%(cwd,finput)
    return finput

def abspath_for_getmu(finput):
    if not finput.replace(' ','').startswith('/') and not finput.startswith('$'): finput = '%s/%s'%(cwd,finput.replace(' ',''))
    return finput

def nmlval_to_abspath(key,value):
    if key.lower() in ['kcor_file','vpec_file'] and not value.startswith('/') and not value.startswith('$') and '/' in value:
        if key.lower() == 'kcor_file' and os.path.exists(os.path.expandvars('$SNDATA_ROOT/kcor/%s'%value)):
            return value
        else:
            value = '%s/%s'%(cwd,value)
    return value

class SALT3pipe():
    def __init__(self,finput=None):
        self.finput = finput
        self.BYOSED = BYOSED()
        self.Simulation = Simulation()
        self.Training = Training()
        self.LCFitting = LCFitting()
        self.GetMu = GetMu()
        self.CosmoFit = CosmoFit()
        self.Data = Data()
        self.BiascorSim = Simulation()
        self.BiascorLCFit = LCFitting()

        self.build_flag = False
        self.config_flag = False
        self.glue_flag = False

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
        
        for prostr in self.pipepros:
            sectionname = [x for x in config.sections() if x.startswith(prostr)]
            if len(sectionname) == 1:
                prostr = sectionname[0]
            pipepro = self._get_pipepro_from_string(prostr)
            setkeys = self._get_config_option(config,prostr,'set_key')
            if setkeys is not None:
                pipepro.setkeys = m2df(setkeys)
            else:
                pipepro.setkeys = None
            baseinput = self._get_config_option(config,prostr,'baseinput')
            outname = self._get_config_option(config,prostr,'outinput')
            pro = self._get_config_option(config,prostr,'pro')
            batch = self._get_config_option(config,prostr,'batch',dtype=boolean_string)
            validplots = self._get_config_option(config,prostr,'validplots',dtype=boolean_string)
            proargs = self._get_config_option(config,prostr,'proargs')
            prooptions = self._get_config_option(config,prostr,'prooptions')
            snlist = self._get_config_option(config,prostr,'snlist')
            pipepro.configure(baseinput=baseinput,
                              setkeys=pipepro.setkeys,
                              outname=outname,
                              pro=pro,
                              proargs=proargs,
                              prooptions=prooptions,
                              snlist=snlist,
                              batch=batch,
                              validplots=validplots)

    def run(self,onlyrun=None):
        if not self.build_flag: build_error()
        if not self.config_flag: config_error()

        if onlyrun is not None:
            if isinstance(onlyrun,str):
                onlyrun = [onlyrun]
        
        for prostr in self.pipepros:
            if onlyrun is not None and prostr not in onlyrun:
                continue
            
            pipepro = self._get_pipepro_from_string(prostr)
            pipepro.run(batch=pipepro.batch)

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
        pro2 = self._get_pipepro_from_string(pipepros[1])
        pro1_out = pro1.glueto(pro2)
        if 'lcfit' in pipepros[1].lower():
            pro2_in = pro2._get_input_info().loc[on]
            pro2_in['value'] = ', '.join(pro1_out)
        else:
            pro2_in = pro2._get_input_info().loc[0]
            pro2_in['value'] = pro1_out

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
                           validplots=pro1.validplots)
        
        if not pipepros[1].lower().startswith('cosmofit'):
            pro2.configure(setkeys = setkeys,
                           pro=pro2.pro,
                           proargs=pro2.proargs,
                           baseinput=pro2.outname,
                           prooptions=pro2.prooptions,
                           outname=pro2.outname,
                           batch=pro2.batch,
                           validplots=pro2.validplots)
        else:
            pro2.configure(pro=pro2.pro,
                           prooptions=pro2.prooptions,
                           outname=setkeys['value'].values[0],
                           batch=pro2.batch,
                           validplots=pro2.validplots)


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
        elif pipepro_str.lower().startswith("train"):
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
        else:
            raise ValueError("Unknow pipeline procedure:",pipepro.strip())
        return pipepro

    def _multivalues_to_df(self,values,colnames=None,stackvalues=False):
        df = pd.DataFrame([s.split() for s in values.split('\n')])
        if df.empty:
            return None
        if colnames is None:
            if df.shape[1] == 2:
                colnames = ['key','value']
            elif df.shape[1] == 3:
                if np.any(df.isna()):
                    colnames = ['key','value']
                    stackvalues = True
                else:
                    colnames=['section','key','value']
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
                  proargs=None,prooptions=None,batch=False,
                  validplots=False,**kwargs):  
        if pro is not None and "$" in pro:
            self.pro = os.path.expandvars(pro)
        else:
            self.pro = pro
        self.baseinput = baseinput
        self.setkeys = setkeys
        self.proargs = proargs
        self.prooptions = prooptions
        self.batch = batch
        self.validplots = validplots

        self.gen_input(outname=self.outname)

    def gen_input(self,outname=None):
        pass

    def run(self,batch=None):
        arglist = [self.proargs] + [finput_abspath(self.finput)] +[self.prooptions]
        arglist = list(filter(None,arglist))
        args = []
        for arg in arglist:
            if arg is not None:
                for argitem in arg.split(' '):
                    args.append(argitem)

        if batch: _run_batch_pro(self.pro, args, done_file=self.done_file)
        else: _run_external_pro(self.pro, args)

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

    def configure(self,snlist=None,**kwargs):
        self.keys = {'snlist':snlist}

    def _get_output_info(self):
        df = {}
        key = 'snlist'
        df['key'] = key
        df['value'] = self.keys[key]
        return pd.DataFrame([df])

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        snlist = self._get_output_info().value.values[0]
        if not os.path.exists(snlist):
            raise ValueError("Path does not exists",snlist)             
        if pipepro.lower().startswith('train'):
            return snlist
        elif pipepro.lower().startswith('lcfit'):
            simpath = os.path.join(os.environ['SNDATA_ROOT'],'SIM/')
            idx = snlist.find(simpath)
            if idx !=0:
                raise ValueError("photometry must be in $SNDATA_ROOT/SIM")
            else:
                return os.path.dirname(snlist[len(simpath):]) 
        else:
            raise ValueError("data can only glue to training or lcfitting")
    
    def run(self,**kwargs):
        pass


class BYOSED(PyPipeProcedure):

    def configure(self,baseinput=None,setkeys=None,
                  outname="pipeline_byosed_input.input",byosed_default="BYOSED/BYOSED.params",
                  bkp_orig_param=False,**kwargs):   
        self.done_file = None
        self.outname = outname
        super().configure(pro=None,baseinput=baseinput,setkeys=setkeys)
        #rename current byosed param
        if os.path.exists(os.path.dirname(byosed_default)):
            byosed_default = byosed_default
        elif os.path.exists(os.path.dirname(byosed_default.lower())):
            byosed_default = byosed_default.lower()
        else:
            raise ValueError("Directory {} does not exists".format(os.path.dirname(byosed_default)))

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

    def run(self,**kwargs):
        pass

class Simulation(PipeProcedure):

    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  batch=False,validplots=False,
                  outname="pipeline_byosed_input.input",**kwargs):
        self.done_file = finput_abspath('%s/Sim.DONE'%os.path.dirname(baseinput))
        self.outname = outname
        self.prooptions = prooptions
        self.batch = batch
        self.validplots = validplots
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          prooptions=prooptions,batch=batch,validplots=validplots)

    def gen_input(self,outname="pipeline_sim_input.input"):
        self.outname = outname
        self.finput,self.keys,self.done_file = _gen_snana_sim_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                                    outname=outname,done_file=self.done_file)
    
    def _get_output_info(self):
        if self.batch:
            keys = ['PATH_SNDATA_SIM','GENVERSION','GENPREFIX']
        else:
            keys = ['PATH_SNDATA_SIM','GENVERSION']
        df = pd.DataFrame()
        for key in keys:
            df0 = {}     
            df0['key'] = key
            if key in self.keys.keys():
                df0['value'] = self.keys[key]
            else:
                df0['value'] = ''
            df = df.append(df0, ignore_index=True)

        return df

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        df = self._get_output_info()
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
                outdir = os.sep.join(['$SNDATA_ROOT/SIM',df.set_index('key').loc['GENVERSION'].value])
            else:
                outdirs = []
                for genversion in df.set_index('key').loc['GENVERSION'].value:
                    outdirs += [os.sep.join(['$SNDATA_ROOT/SIM',
                                             genversion])]
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
            prefix = df.loc[df.key=='GENVERSION','value'].values[0]
            return "{}/{}.LIST".format(res,prefix)
        elif pipepro.lower().startswith('lcfit'):
            return df.loc[df.key=='GENVERSION','value'].values[0]
            # idx = res.find(simpath)
            # if idx !=0:
            #     raise ValueError("photometry must be in $SNDATA_ROOT/SIM")
            # else:
            #     return res[len(simpath):] 
        else:
            raise ValueError("sim can only glue to training or lcfitting")
        
class Training(PyPipeProcedure):

    def configure(self,pro=None,baseinput=None,setkeys=None,proargs=None,
                  prooptions=None,outname="pipeline_train_input.input",**kwargs):
        self.done_file = None
        self.outname = outname
        self.proargs = proargs
        self.prooptions = prooptions
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          proargs=proargs,prooptions=prooptions)

    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('lcfit'):
            outdir = self._get_output_info().value.values[0]
            ##copy necessary files to a model folder in SNDATA_ROOT
            modeldir = 'lcfitting/SALT3.test'
            #self.__transfer_model_files(outdir,modeldir,rename=False)
            self._set_output_info(modeldir)
            return modeldir
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

    def _set_output_info(self,value):
        df = {}
        section = 'iodata'
        key = 'outputdir'
        df['section'] = section
        df['key'] = key
        df['value'] = value
        self.keys[section][key] = value
        return pd.DataFrame([df])
    
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

    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  batch=False,validplots=False,outname="pipeline_lcfit_input.input",**kwargs):
        self.done_file = None
        self.outname = outname
        self.prooptions = prooptions
        self.batch = batch
        self.validplots = validplots
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          prooptions=prooptions,batch=batch,validplots=validplots)

    def gen_input(self,outname="pipeline_lcfit_input.input"):
        self.outname = outname
        self.finput,self.keys = _gen_snana_fit_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                     outname=outname)


    def glueto(self,pipepro):
        if not isinstance(pipepro,str):
            pipepro = type(pipepro).__name__
        if pipepro.lower().startswith('getmu'):
            outprefix = abspath_for_getmu(self._get_output_info().value.values[0])
            if self.batch: return str(outprefix)
            else: return str(outprefix)+'.FITRES.TEXT'
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
        df2 = pd.DataFrame([df,df2])

        if not self.batch:
            return df2.set_index('type')
        else:
            section = 'HEADER'
            key = 'VERSION'
            df['section'] = section
            df['key'] = key
            df['value'] = self.keys[section][key]
            df['type'] = 'phot'
            df2 = df2.append(df,ignore_index=True)
            return df2.set_index('type')

    def _get_output_info(self):
        df = {}
        if self.batch:
            section = 'HEADER'
            key = 'OUTDIR'
        else:
            section = 'SNLCINP'
            key = 'TEXTFILE_PREFIX'
        df['section'] = section
        df['key'] = key
        df['value'] = self.keys[section][key]
        return pd.DataFrame([df])
 
class GetMu(PipeProcedure):
    def configure(self,pro=None,baseinput=None,setkeys=None,prooptions=None,
                  batch=False,validplots=False,outname="pipeline_getmu_input.input",**kwargs):
        self.done_file = finput_abspath('%s/GetMu.DONE'%os.path.dirname(baseinput))
        self.outname = outname
        self.prooptions = prooptions
        self.batch = batch
        self.validplots = validplots
        super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,
                          prooptions=prooptions,batch=batch,validplots=validplots,done_file=self.done_file)

    def gen_input(self,outname="pipeline_getmu_input.input"):
        self.outname = outname
        self.finput,self.keys,self.delimiter = _gen_general_input(basefilename=self.baseinput,setkeys=self.setkeys,
                                                                  outname=outname,sep=['=',': '],done_file=self.done_file)
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

        return pd.DataFrame([df])

    def _get_output_info(self):
        df = {}
        key = 'prefix'
        df['key'] = key
        df['value'] = self.keys[key].strip()+'.M0DIF'
        return pd.DataFrame([df])

class CosmoFit(PipeProcedure):
    def configure(self,setkeys=None,pro=None,outname=None,prooptions=None,batch=False,
                  validplots=False,**kwargs):
        self.done_file = None
        if setkeys is not None:
            outname = setkeys.value.values[0]
        self.prooptions = prooptions
        self.finput = outname
        self.batch = batch
        self.validplots = validplots
        super().configure(pro=pro,outname=outname,prooptions=prooptions,batch=batch,validplots=validplots)

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

def _run_batch_pro(pro,args,done_file=None):

    if isinstance(args, str):
        args = [args]

    if done_file:
        # SNANA doesn't remove old done files
        if os.path.exists(done_file): os.system('rm %s'%done_file)

    print("Running",' '.join([pro] + args))
    res = subprocess.run(args = list([pro] + args),capture_output=True)
    stdout = res.stdout.decode('utf-8')
    if 'ERROR MESSAGE' in stdout:
        for line in stdout[stdout.find('ERROR MESSAGE'):].split('\n'):
            print(line)
        raise RuntimeError("Something went wrong...")
    if 'WARNING' in stdout:
        for line in stdout[stdout.find('WARNING'):].split('\n'):
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
            time.sleep(15)

    success = False
    with open(done_file,'r') as fin:
        for line in fin:
            if 'SUCCESS' in line:
                success = True


    if success:
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
        print("No modification on the input file, copying {} to {}".format(basefilename,outname))
        os.system('cp %s %s'%(basefilename,outname))
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
                         outname=None,done_file=None):

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
        print("No modification on the input file, copying {} to {}".format(basefilename,outname))
        os.system('cp %s %s'%(basefilename,outname))
        config = {}
        for i,line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                kwline = line.split(":",maxsplit=1)
                kw = kwline[0]
                if kw not in config:
                    config[kw] = kwline[1].strip()
                else:
                    config[kw] = np.append([config[kw]],[kwline[1].strip()])
    else:
        setkeys = pd.DataFrame(setkeys)
        if np.any(setkeys.key.duplicated()):
            raise ValueError("Check for duplicated entries for",setkeys.keys[setkeys.keys.duplicated()].unique())

        config = {}
        for i,line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                kwline = line.split(":",maxsplit=1)
                kw = kwline[0]
                if "#" in kwline[1]:
                    kwline[1] = kwline[1].split("#")[0].strip()+'\n'
                basekws.append(kw)
                if kw in setkeys.key.values:
                    keyvalue = setkeys[setkeys.key==kw].value.values[0]
                    if isinstance(keyvalue,list):
                        kwline[1] = ' '.join(list(filter(None,keyvalue)))+'\n'
                    else:
                        kwline[1] = str(keyvalue)+'\n'
                    print("Setting {} = {}".format(kw,kwline[1].strip()))
                lines[i] = ": ".join(kwline)
                config[kw] = kwline[1].strip()

        outfile = open(outname,"w")
        for line in lines:
            outfile.write(line)

        for key,value in zip(setkeys.key,setkeys.value):
            if not key in basekws:
                if isinstance(value,list):
                    valuestr = ' '.join(list(filter(None,value)))
                else:
                    valuestr = str(value)
                newline = key+": "+valuestr+'\n'
                print("Adding key {} = {}".format(key,valuestr))
                outfile.write(newline)
                config[key] = valuestr.strip()

        print("Write sim input to file:",outname)

    with open(outname,'a') as fout:
        if 'GENPREFIX' in config.keys():
            done_file = finput_abspath('%s/%s'%('SIMLOGS_%s'%config['GENPREFIX'],done_file.split('/')[-1]))
            print('DONE_STAMP: %s'%done_file,file=fout)
        else:
            print('DONE_STAMP: %s'%done_file,file=fout)

    return outname,config,done_file


def _gen_snana_fit_input(basefilename=None,setkeys=None,
                         outname=None):

    import f90nml
    from f90nml.namelist import Namelist
    nml = f90nml.read(basefilename)

    # first write the header info
    if not os.path.isfile(basefilename):
        raise ValueError("basefilename cannot be None")
    print("Load base fit input file..",basefilename)
    basefile = open(basefilename,"r")
    lines = basefile.readlines()
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
    nml['header'].__setitem__('done_stamp','ALL.DONE')

    if setkeys is not None:
        for index, row in setkeys.iterrows():
            sec = row['section']
            key = row['key']
            v = row['value']
            if not sec.lower() in nml.keys():
                raise ValueError("No section named",sec)
            print("Adding/modifying key {}={} in &{}".format(key,v,sec))
            nml[sec][key] = v

    # a bit clumsy, but need to make sure these are the same for now:
    #nml['header'].__setitem__('version',nml['snlcinp']['version_photometry'])
    print("Write fit input to file:",outname)
    _write_nml_to_file(nml,outname,append=True)

    return outname,nml

def _gen_general_input(basefilename=None,setkeys=None,outname=None,sep='=',done_file=None):

    config,delimiter = _read_simple_config_file(basefilename,sep=sep)
    #if setkeys is None:
    #    print("No modification on the input file, keeping {} as input".format(basefilename))
    if setkeys is not None: #else:
        for index, row in setkeys.iterrows():
            key = row['key']
            values = row['value']
            if not isinstance(values,list) and not isinstance(values,np.ndarray): values = [values]
            for value in values:
                print("Adding/modifying key {}={}".format(key,value))
                config[key] = value
    if done_file:
        key = 'DONE_STAMP'
        v = done_file
        config[key] = v

    print("input file saved as:",outname)
    _write_simple_config_file(config,outname,delimiter)

    if len(sep) == 1: return outname,config
    else: return outname,config,delimiter

def _read_simple_config_file(filename,sep='='):
    config,delimiter = {},{}
    f = open(filename,"r")
    lines = f.readlines()

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
    outfile = open(filename,"w")
    for key in config.keys():
        values = config[key]
        if not isinstance(values,list) and not isinstance(values,np.ndarray): values = [values]
        for value in values:
            if not key in delimiter.keys(): outfile.write("{}={}\n".format(key,value))
            else: outfile.write("{}{}{}\n".format(key,delimiter[key],value))
    outfile.close()

    return

def _write_nml_to_file(nml,filename,headerlines=[],append=False):
    outfile = open(filename,"w")

    for key in nml.keys():
        if key.lower() == 'header':
            for key2 in nml[key].keys():
                value = nml[key][key2]
                if isinstance(value,str):
                    value = "{}".format(value)
                elif isinstance(value,list):
                    value = ','.join([str(x) for x in value if not isinstance(x,str)])
                if key2.lower() == 'version':
                    for version in value.replace(',','').split():
                        outfile.write("{}: {}".format(key2.upper(),version))
                        outfile.write("\n")
                else:
                    outfile.write("{}: {}".format(key2.upper(),value))
                    outfile.write("\n")

        else:
            outfile.write('&'+key.upper())
            outfile.write('\n')
            for key2 in nml[key].keys():
                value = nmlval_to_abspath(key2,nml[key][key2])
                if isinstance(value,str):
                    value = "'{}'".format(value)
                elif isinstance(value,list):
                    value = ','.join([str(x) for x in value if not isinstance(x,str)])
                outfile.write("  {} = {}".format(key2.upper(),value))
                outfile.write("\n")
            outfile.write('&END\n\n')
    return
