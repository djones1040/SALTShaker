from abc import ABC,abstractmethod
from collections import namedtuple
import logging
log=logging.getLogger(__name__)
import numpy as np

from functools import reduce

salttrainingresult= namedtuple('salttrainingresult',
            ['num_lightcurves' , 'num_spectra' ,'num_sne' ,
            'parlist' , 'params' , 'params_raw' ,'phase' ,'wave' , 'componentnames', 
            
            'componentsurfaces' ,  'modelerrsurfaces', 'dataerrsurfaces' ,
            'modelcovsurfaces','datacovsurfaces','clpars','clerr','clscat','snparams','stepsizes'])        
        
__registry__=dict()

class salttrainingoptimizer(ABC):

    

    def __init_subclass__(cls):
#         super().__init_subclass__()
        __registry__[cls.__name__]=cls
    
    @property
    @abstractmethod
    def configoptionnames(self):
        pass
    
    @abstractmethod
    def __init__(self, guess,saltresids,outputdir,options):
        self.ifixedparams= reduce( np.union1d, [saltresids.__dict__['i'+x] for x in options.fixedparams.replace(' ','').split(',') if len(x)] ,[])
        self.ifixedparams= np.isin(np.arange(guess.size), self.ifixedparams)

        
    
    @classmethod
    @abstractmethod
    def add_training_options(cls,parser,config):
        """ Specifies to the parser all required configuration options"""
        pass
    
    @abstractmethod
    def optimize(self,initialparams):
        """ Given an initial set of SALT model parameters, returns an optimized set of parameters"""
        pass

#     @abstractmethod
#     def estimateparametererrors(self,initialparams):
#         """ Given an initial set of optimized SALT model parameters, estimate errors in the parameters"""
#         pass


def getoptimizer(name):
    """ Takes as input a string, returns a registered valid `salttrainingoptimizer` class"""
    return __registry__[name]
    
from . import gaussnewton,gradientdescent
