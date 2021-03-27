import yaml
from logging import config
import logging
import os.path

def dictconfigfromYAML(filename,outputdir):
	if not os.path.exists(filename):
		print(f"warning: logger config file {filename} doesn't exist, default logger configuration will be used (everything above debug to console)")
		logging.basicConfig(format='%(message)s',level=logging.INFO)
		return False
	else:
		if not os.path.exists(outputdir):
			os.makedirs(outputdir)
		with open(filename,'r') as logConfigFile:
			configdictionary=yaml.safe_load(logConfigFile.read())
			for handler in configdictionary['handlers']:
				if 'filename' in configdictionary['handlers'][handler]:
					configdictionary['handlers'][handler]['filename']=configdictionary['handlers'][handler]['filename'].replace('OUTPUTDIR',outputdir)
			config.dictConfig(configdictionary)
		return True
