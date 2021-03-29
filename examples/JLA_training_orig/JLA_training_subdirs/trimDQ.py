#!/usr/bin/env python
# D. Jones - 5/30/20

import os
import glob
import numpy as np
from saltshaker.util import snana

# snana.exe snfit_Hamuy1996.nml HEADER_OVERRIDE_FILE Hamuy1996_LC.FITRES SIMLIB_OUT simlib/Hamuy96.SIMLIB
# snana.exe snfit_Hicken2009.nml HEADER_OVERRIDE_FILE Hicken2009_LC.FITRES SIMLIB_OUT simlib/Hicken2009.SIMLIB
# snana.exe snfit_Jha2006.nml HEADER_OVERRIDE_FILE Jha2006_LC.FITRES SIMLIB_OUT simlib/Jha2006.SIMLIB
# snana.exe snfit_OTHER_LOWZ.nml HEADER_OVERRIDE_FILE OTHER_LOWZ_LC.FITRES SIMLIB_OUT simlib/OTHER_LOWZ.SIMLIB
# snana.exe snfit_Riess1999.nml HEADER_OVERRIDE_FILE Riess1999_LC.FITRES SIMLIB_OUT simlib/Riess1999.SIMLIB
# snana.exe snfit_SDSS.nml HEADER_OVERRIDE_FILE SDSS.FITRES SIMLIB_OUT simlib/SDSS.SIMLIB
# snana.exe snfit_SNLS3.nml HEADER_OVERRIDE_FILE SNLS3_LC.FITRES SIMLIB_OUT simlib/SNLS3.SIMLIB
# ------------
# snana.exe snfit_Foundation_DR1.nml HEADER_OVERRIDE_FILE Foundation_DR1.FITRES.TEXT SIMLIB_OUT simlib/Foundation_DR1.SIMLIB

# snlc_sim.exe JLA_TRAINING_Hamuy96.INPUT
# snlc_sim.exe JLA_TRAINING_Hicken09.INPUT
# snlc_sim.exe JLA_TRAINING_Jha06.INPUT
# snlc_sim.exe JLA_TRAINING_OTHER_LOWZ.INPUT
# snlc_sim.exe JLA_TRAINING_Riess99.INPUT
# snlc_sim.exe JLA_TRAINING_SDSS.INPUT
# snlc_sim.exe JLA_TRAINING_SNLS3.INPUT

def main():

	for lcdir in ['Hamuy1996_LC','Hicken2009_LC','Jha2006_LC',
				  'OTHER_LOWZ_LC','Riess1999_LC','SDSS',
				  'SNLS3_LC']:
		files = glob.glob(f"{lcdir}/*dat")
		if not len(files):
			files = glob.glob(f"{lcdir}/*DAT")

		for f in files:
			speclines = []
			with open(f) as fin, open(f.replace(lcdir,f'lcdata/{lcdir}'),'w') as fout:
				for line in fin:
					line = line.replace('\n','')
					if line.startswith('SPEC:') and int(line.split()[-1]) == 1:
						speclines += [line]
					elif line.startswith("REDSHIFT_HELIO:"):
						zHel = float(line.split()[1])
						if zHel < 0.01:
							print(f"REDSHIFT_HELIO: {zHel+0.01:.5f} +- 0.001    (HELIO)",file=fout)
						else:
							print(line,file=fout)
					elif line.startswith('SPECTRUM_NLAM:'):
						pass
					elif line.startswith('SPECTRUM_END:'):
						print(f"SPECTRUM_NLAM:     {len(speclines)}",file=fout)
						for specline in speclines:
							print(specline,file=fout)
						print("SPECTRUM_END:",file=fout)
						print("",file=fout)
						speclines = []
					elif not line.startswith('SPEC:'):
						print(line,file=fout)

						
if __name__ == "__main__":
	main()
