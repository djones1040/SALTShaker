from abc import ABC,abstractmethod
from collections import namedtuple
import logging
log=logging.getLogger(__name__)

         
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
        self.modelobj=saltresids
        self.outputdir=outputdir
        for key in self.configoptionnames:
            self.__dict__[key]=options.__dict__[key]
    
    @classmethod
    @abstractmethod
    def add_training_options(cls,parser,config):
        """ Specifies to the parser all required configuration options"""
        pass
    
    @abstractmethod
    def optimize(self,initialparams):
        """ Given an initial set of SALT model parameters, returns an optimized set of parameters"""
        pass



def getoptimizer(name):
    """ Takes as input a string, returns a registered valid `salttrainingoptimizer` class"""
    return __registry__[name]
    
from . import gaussnewton,gradientdescent
