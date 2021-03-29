#!/usr/bin/env python
# D. Jones - compile pkmjds for Foundation, DES, PS1, original JLA
from txtobj import txtobj

def main():
    fitres_files = ['fitres_init/Foundation_DR1.FITRES.TEXT',
                    'fitres_init/ps1mds.fitres.txt',
                    'fitres_init/SALT2mu_DES_G10.FITRES',
                    'fitres_init/CfA4.fitres',
                    'fitres_init/CSP.fitres']

    with open('SALT3_PKMJD_INIT.LIST','w') as fout1,open('SALT3_PARS_INIT.LIST','w') as fout2:
        print("",file=fout1)
        print("# SNID zHelio x0 x1 c FITPROB",file=fout2)
        
        for f in fitres_files:
            fr = txtobj(f,fitresheader=True)
            print(f)
            if 'ps1mds' in f: fr.zHEL = fr.zCMB
            for j,i in enumerate(fr.CID):
                if 'zHEL' in fr.__dict__.keys():
                    print(f"{i} {fr.zHEL[j]} {fr.x0[j]} {fr.x1[j]} {fr.c[j]} {fr.FITPROB[j]}",file=fout2)
                else:
                    print(f"{i} {fr.zCMB[j]} {fr.x0[j]} {fr.x1[j]} {fr.c[j]} {fr.FITPROB[j]}",file=fout2)
                print(f"{i} {fr.PKMJD[j]}",file=fout1)

if __name__ == "__main__":
    main()
