import numpy as np

@colorlawclass
class colorlaw_default:

    def colorexp(c,colorLaw):

        colorexp= 10. ** (colorLaw * c)
        
        return colorexp

    def colorderiv(modulatedFlux,colorlaw):

        return np.sum((modulatedFlux)*np.log(10)*colorlaw[np.newaxis,idx], axis=1)[np.newaxis].transpose()

    def colorlawderiv(modulatedFlux,colorLawDeriv,varyParams,iCL,c):

        clderiv = (np.sum((modulatedFlux)[:,:,np.newaxis]*\
                          colorLawDeriv[:,varyParams[iCL]][np.newaxis,idx,:], axis=1))*-0.4*np.log(10)*c
    
        return clderiv
