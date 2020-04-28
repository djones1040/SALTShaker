from astropy.io import ascii
import numpy as np

bias=ascii.read('../biascor/FITOPT000.FITRES')
print(len(bias))
