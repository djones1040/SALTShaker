import argparse
import configparser
from os import path 

import logging
log=logging.getLogger(__name__)

__all__= ['expandvariablesandhomecommaseparated','FullPaths','EnvAwareArgumentParser' ,
       'ConfigWithCommandLineOverrideParser','boolean_string','nonetype_or_int','generateerrortolerantaddmethod']
                
def expandvariablesandhomecommaseparated(paths):
        return ','.join([path.expanduser(path.expandvars(x)) for x in paths.split(',')])

class FullPaths(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest,expandvariablesandhomecommaseparated(values))
                
def boolean_string(s):
        if s not in {'False', 'True', 'false', 'true', '1', '0'}:
                raise ValueError('Not a valid boolean string')
        return (s == 'True') | (s == '1') | (s == 'true')

def nonetype_or_int(s):
        if s == 'None': return None
        else: return int(s)


class EnvAwareArgumentParser(argparse.ArgumentParser):

        def add_argument(self,*args,**kwargs):
                if 'action' in kwargs:
                        action=kwargs['action']
                        
                        if (action==FullPaths or action=='FullPaths') and 'default' in kwargs:
                                kwargs['default']=expandvariablesandhomecommaseparated(kwargs['default'])
                return super().add_argument(*args,**kwargs)
                
class ConfigWithCommandLineOverrideParser(EnvAwareArgumentParser):
                
        def addhelp(self):
                default_prefix='-'
                self.add_argument(
                                default_prefix+'h', default_prefix*2+'help',
                                action='help', default=argparse.SUPPRESS,
                                help=('show this help message and exit'))

        def add_argument_with_config_default(self,config,section,*keys,**kwargs):
                """Given a ConfigParser and a section header, scans the section for matching keys and sets them as the default value for a command line argument added to self. If a default is provided, it is used if there is no matching key in the config, otherwise this method will raise a KeyError"""
                if 'clargformat' in kwargs:
                        if kwargs['clargformat'] =='prependsection':
                                kwargs['clargformat']='--{section}_{key}'
                else:
                        kwargs['clargformat']='--{key}'
                
                clargformat=kwargs.pop('clargformat')

                clargs=[clargformat.format(section=section,key=key) for key in keys]
                def checkforflagsinconfig():
                        for key in keys:
                                if key in config[section]:
                                        return key,config[section][key]
                        raise KeyError(f'key {key} not found in section {section} of config file')

                try:
                        includedkey,kwargs['default']=checkforflagsinconfig()
                except KeyError:
                        if 'default' in kwargs:
                                pass
                        else:
                                message=f"Key {keys[0]} not found in section {section}; valid keys include: {', '.join(keys)}"
                                if 'help' in kwargs:
                                        message+=f"\nHelp string: {kwargs['help'].format(**kwargs)}"
                                raise KeyError(message)
                if 'nargs' in kwargs and ((type(kwargs['nargs']) is int and kwargs['nargs']>1) or (type(kwargs['nargs'] is str and (kwargs['nargs'] in ['+','*'])))):
                        if kwargs['default'] != argparse.SUPPRESS:
                            if not 'type' in kwargs:
                                    kwargs['default']=kwargs['default'].split(',')
                            else:
                                    kwargs['default']=list(map(kwargs['type'],kwargs['default'].split(',')))
                            if type(kwargs['nargs']) is int:
                                    try:
                                            assert(len(kwargs['default'])==kwargs['nargs'])
                                    except:
                                            nargs=kwargs['nargs']
                                            numfound=len(kwargs['default'])
                                            raise ValueError(f"Incorrect number of arguments in {(config)}, section {section}, key {includedkey}, {nargs} arguments required while {numfound} were found")
                return super().add_argument(*clargs,**kwargs)
                
    
def generateerrortolerantaddmethod(parser):
        def wraparg(*args,**kwargs):
            #Wrap this method to catch exceptions, providing a true if no exception was raised, False otherwise.
            try:
                    parser.add_argument_with_config_default(*args,**kwargs)
                    return True
            except Exception as e:
                    log.error('\n'.join(e.args))
                    return False
        return wraparg

