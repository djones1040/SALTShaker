import yaml
from logging import config
import os.path

def dictconfigfromYAML(filename,outputdir):
	if not os.path.exists(filename):
		print(f"warning: logger config file {user_options.loggingconfig} doesn't exist, default logger configuration will be used (everything above debug to console)")
		return False
	else:
		with open(filename,'r') as logConfigFile:
			configdictionary=yaml.safe_load(logConfigFile.read())
			for handler in configdictionary['handlers']:
				if 'filename' in configdictionary['handlers'][handler]:
					configdictionary['handlers'][handler]['filename']=configdictionary['handlers'][handler]['filename'].replace('OUTPUTDIR',outputdir)
			config.dictConfig(configdictionary)
		return True
