import sys,os
import numpy as np
import logging
log=logging.getLogger(__name__)

try:
    from .txtobj import txtobj
except:
    from txtobj import txtobj


def _getDataFormat(filename,):
    fin = open(filename,'r')
    lines = fin.readlines()
    header=''
    next_break=False
    for l in lines:
        if l.startswith('VARNAMES:'):
            l = l.replace('\n','')
            coldefs = l.split()
            next_break=True
            header+=l
        elif next_break:
            l=l.replace('\n','')
            rowprfx=l.split()[0]
            break
        else:
            header+=l
    fin.close()

    with open(filename) as f:
        reader = [x.replace('\n','').split() for x in f if x.startswith('%s'%rowprfx)]
    
    fmt={k:None for k in coldefs}
    i=0
    for column in zip(*reader):
        try:
            temp=[int(x) for x in column]
            fmt[coldefs[i]]='%i'
        except:
            try:
                temp=[float(x) for x in column]
                temp=[len(str(x)[str(x).find('.'):str(x).find('e')]) if 'e' in str(x) else False for x in column]
                if np.any(temp):
                    fmt[coldefs[i]]="%."+'%ie'%int(max(temp))
                else:
                    temp=[len(str(x)[str(x).find('.'):]) if '.' in str(x) else 0 for x in column]
                    fmt[coldefs[i]]="%."+'%if'%int(max(temp))
            except:
                fmt[coldefs[i]]='%s'
                
        i += 1
    return(' '.join([fmt[x] for x in coldefs]),coldefs,header)

def readfitres(fitresfile, fitresheader = True, version=None):
    fr = txtobj(fitresfile,fitresheader = fitresheader)
    fr.filename=os.path.splitext(os.path.basename(fitresfile))[0]
    fr.version=version
    return(fr)

def writefitres(fitresobj,outfile,wrtfitresfmt,wrtfitresvars,
                wrtfitresheader,append=False):

    if not append:
        fout = open(outfile,'w')
        print(wrtfitresheader,file=fout)
    else:
        fout = open(outfile,'a')

    for c in range(len(fitresobj.CID)):
        outvars = ()
        for v in wrtfitresvars:
            outvars += (fitresobj.__dict__[v][c],)
        print(wrtfitresfmt%outvars,file=fout)
              

    fout.close()

def cutFitRes(fitresfile,clobber=False,cuts=[],field=None):
    fitresfmt,fitresvars,fitresheader=_getDataFormat(fitresfile)
    fr=readfitres(fitresfile)

    for cut in cuts:
        if '<' in cut[1]:
            cond=fr.__dict__[cut[0]]<float(cut[1][1:])
        elif '>' in cut[1]:
            cond=fr.__dict__[cut[0]]>float(cut[1][1:])
        else:
            try:
                cond=fr.__dict__[cut[0]]==float(cut[1][1:])
            except:
                cond=fr.__dict__[cut[0]]==cut[1][1:]
        if field is not None:
            cut_ind=np.logical_or(np.logical_and(fr.FIELD==field,cond),fr.FIELD!=field)
        else:
            cut_ind=cond
        for c in fitresvars:
            if c=='VARNAMES:':
                continue
            fr.__dict__[c]=fr.__dict__[c][cut_ind]

    if not clobber:
        os.rename(fitresfile,fitresfile.replace(os.path.basename(fitresfile),'ORIGINAL.FITRES'))
    writefitres(fr,fitresfile,fitresfmt,fitresvars,fitresheader)


def main():
    cutFitRes('/Users/jpierel/rodney/salt3_testing/FITOPT000.FITRES',clobber=False,cuts=[['zCMB','>0.8']],field='MEDIUM')
    #fr=readfitres('test.fitres')
    #print(np.min(fr.zCMB[fr.FIELD=='MEDIUM']))

if __name__=='__main__':
    main()