##SALT3 pipeline
##sim -> training -> fitting

import subprocess
import configparser
import pandas as pd
import os

class SALT3pipe():
	def __init__(self,finput=None):
		self.finput = finput
		self.BYOSED = BYOSED()
		self.Simulation = Simulation()
		self.Training = Training()
		self.LCFitting = LCFitting()

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
									   colnames=['key','value'])
		self.Simulation.configure(baseinput=config.get('simulation','baseinput'),
								  setkeys=self.Simulation.setkeys,
								  outname=config.get('simulation','outinput'),
								  pro=config.get('simulation','pro'))
		self.Training.setkeys = m2df(config.get('training','set_key'),
									 colnames=['section','key','value'])
		self.Training.configure(baseinput=config.get('training','baseinput'),
								setkeys=self.Training.setkeys,
								outname=config.get('training','outinput'),
								pro=config.get('training','pro'),
								proargs=config.get('training','proargs'))
		self.LCFitting.setkeys = m2df(config.get('lcfitting','set_key'),colnames=['key','value'])


	def run(self):
		self.Simulation.run()
		self.Training.run()
		self.LCFitting.run()


	def _multivalues_to_df(self,values,colnames=None):
		df = pd.DataFrame([s.split() for s in values.split('\n')], columns=colnames)
		return df


class PipeProcedure():
	def __init__(self):
		self.pro = None
		self.baseinput = None
		self.setkeys = None
		self.proargs = None
		self.outname = None

	def configure(self,pro=None,baseinput=None,setkeys=None,
				  proargs=None):  
		self.pro = pro
		self.baseinput = baseinput
		self.setkeys = setkeys
		self.proargs = proargs

		if setkeys is not None:
			self.gen_input(outname=self.outname)
		else:
			self.finput = baseinput

	def gen_input(self):
		pass

	def run(self):
		if self.proargs is None:
			args = self.finput
		else:
			args = [self.proargs] + [self.finput]
		_run_external_pro(self.pro, args)


class PyPipeProcedure(PipeProcedure):

	def gen_input(self,outname="pipeline_generalpy_input.input"):
		self.outname = outname
		self.finput = _gen_general_python_input(basefilename=self.baseinput,setkeys=self.setkeys,
												outname=outname)


class BYOSED(PyPipeProcedure):

	def configure(self,baseinput=None,setkeys=None,
				  outname="pipeline_byosed_input.input"):
		self.outname = outname
		super().configure(pro=None,baseinput=baseinput,setkeys=setkeys)


class Simulation(PipeProcedure):

	def configure(self,pro=None,baseinput=None,setkeys=None,
				  outname="pipeline_byosed_input.input"):
		self.outname = outname
		super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys)

	def gen_input(self,outname="pipeline_sim_input.input"):
		self.outname = outname
		self.finput = _gen_snana_sim_input(basefilename=self.baseinput,setkeys=self.setkeys,
										   outname=outname)

class Training(PyPipeProcedure):

	def configure(self,pro=None,baseinput=None,setkeys=None,proargs=None,
				  outname="pipeline_train_input.input"):
		self.outname = outname
		self.proargs = proargs
		super().configure(pro=pro,baseinput=baseinput,setkeys=setkeys,proargs=proargs)

class LCFitting(PipeProcedure):

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

def _gen_general_python_input(basefilename=None,setkeys=None,
								 outname=None):

	config = configparser.ConfigParser()
	if not os.path.isfile(basefilename):
		raise ValueError("File does not exist",basefilename)
	if not os.path.exists(os.path.dirname(outname)):
		os.makedirs(os.path.dirname(outname))
	
	config.read(basefilename)
	setkeys = pd.DataFrame(setkeys)
	for index, row in setkeys.iterrows():
		sec = row['section']
		key = row['key']
		v = row['value']
		if not sec in config.sections():
			config.add_section(sec)
		print("Adding key {}={} in [{}]".format(key,v,sec))
		config[sec][key] = v
	with open(outname, 'w') as f:
		config.write(f)

	print("input file saved as:",outname)
	return outname


def _gen_snana_sim_input(basefilename=None,setkeys=None,
						 outname=None):

	#TODO:
	#read in kwlist from standard snana kw list
	#determine if the kw is in the list

	#read in a default input file
	#add/edit the kw

	print("Load base sim input file..",basefilename)
	basefile = open(basefilename,"r")
	lines = basefile.readlines()
	basekws = []
	setkeys = pd.DataFrame(setkeys)

	for i,line in enumerate(lines):
		kwline = line.split(":")
		kw = kwline[0]
		basekws.append(kw)
		if kw in setkeys.key:
			print("Setting {} = {}".format(kw,setkeys[kw].value))
			kwline[1] = setkeys[kw].value
		lines[i] = ": ".join(kwline)

	outfile = open(outname,"w")
	for line in lines:
		outfile.write(line)

	for key,value in zip(setkeys.key,setkeys.value):
		if not key in basekws:
			print("Adding key {}={}".format(key,value))
			newline = key+": "+value
			outfile.write(newline)

	print("Write sim input to file:",outname)

	return outname
