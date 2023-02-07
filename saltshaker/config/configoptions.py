
import configargparse
import sys
from os import path

import logging
log=logging.getLogger(__name__)



def expandvariablesandhomecommaseparated(paths):
        return ','.join([os.path.expanduser(os.path.expandvars(x)) for x in paths.split(',')])

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
                
class DefaultRequiredParser(configargparse.ArgParser):
    """ Extension to configparse.ArgParser that defaults all non-positional arguments to be a required argument"""
    def add_argument(self,*args,**kwargs):
    
        if 'action' in kwargs  and (kwargs['action']==FullPaths or kwargs['action']=='FullPaths') and 'default' in kwargs:
                kwargs['default']=expandvariablesandhomecommaseparated(kwargs['default'])

        if 'required' not in kwargs and args[0].startswith(parser.prefix_chars):
                kwargs['required']=not 'default' in kwargs
                
        super().add_argument(*args,**kwargs)


