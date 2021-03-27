#!/usr/bin/env python
# D. Jones - 10/30/20
# try figure out which Foundation spectra in the training set
# are public and where they all come from

# 76 appear to be public
# ||||||| - 7 from our group potentially
# the remaining 61.......
# 

import numpy as np
from saltshaker.util import snana
import dateutil.parser
from astropy.time import Time
import glob
from astrodateutils import date_to_mjd, mjd_to_date
from collections import OrderedDict
import json
import requests
from coordutils import GetSexigesimalString

tnsapi = 'https://wis-tns.weizmann.ac.il/api/get'
tnsapikey = 'ecd2dec8cee4ed72a39fe8467ddd405fec4eef14'

def get(url,json_list,api_key):
    try:
        # url for get obj
        get_url=url+'/object'
        # change json_list to json format
        json_file=OrderedDict(json_list)
        # construct the list of (key,value) pairs
        get_data=[('api_key',(None, api_key)),
                  ('data',(None,json.dumps(json_file)))]
        # get obj using request module
        response=requests.post(get_url, files=get_data)
        return response
    except Exception as e:
        return [None,'Error message : \n'+str(e)]

def search(url,json_list,api_key):
  try:
    search_url=url+'/search'
    json_file=OrderedDict(json_list)
    search_data=[('api_key',(None, api_key)),
                 ('data',(None,json.dumps(json_file)))]
    response=requests.post(search_url, files=search_data)

    return response
  except Exception as e:
    return [None,'Error message : \n'+str(e)]
    
def format_to_json(source):
    # change data to json format and return
    parsed=json.loads(source,object_pairs_hook=OrderedDict)
    #result=json.dumps(parsed,indent=4)
    return parsed #result

def quicklist():
    snidlist = np.loadtxt('public_spec.txt',unpack=True,usecols=[0],dtype=str)

    snfiles = glob.glob('SALT3_training_data/Foundation*spec*txt')
    public_speccount = 0
    for s in snfiles:
        sn = snana.SuperNova(s)
        if len(sn.SPECTRA.keys()) and sn.SNID not in snidlist: print(sn.SNID)

def comp_to_kyle():
    snidlist = np.loadtxt('public_spec.txt',unpack=True,usecols=[0],dtype=str)
    snidlist_kyle = np.loadtxt('kyle_spectra.txt',unpack=True,usecols=[0],dtype=str)
    #for s in snidlist:
    #    if s in snidlist_kyle: print(s)
    import os
    snfiles = glob.glob('SALT3_training_data/Foundation*spec*txt')
    public_speccount = 0
    for s in snfiles:
        sn = snana.SuperNova(s)
        if len(sn.SPECTRA.keys()) and sn.SNID not in snidlist and sn.SNID not in snidlist_kyle:
            print(sn.SNID)
            #os.system(f'ls /Users/David/Dropbox/research/SALT3/salt3/data/trainingdata/FoundationSpeccopy/*{sn.SNID}*')
        
def main():
    snfiles = glob.glob('SALT3_training_data/Foundation*spec*txt')
    public_speccount = 0
    for s in snfiles:
        sn = snana.SuperNova(s)
        #if sn.SNID != 'ASASSN-17aj': continue
        if not len(sn.SPECTRA.keys()): continue
        #print(sn.SNID)

        object_in_TNS = False
        if sn.SNID.startswith('SN2') or sn.SNID.startswith('AT2'):
            TNSGetSingle = [("objname",sn.SNID.replace('AT','').replace('SN','')),
                            ("photometry","0"),
                            ("spectra","1")]

            response=get(tnsapi, TNSGetSingle, tnsapikey)
            json_data = format_to_json(response.text)
        else:
            rasex,decsex = GetSexigesimalString(sn.RA,sn.DECL)
            search_obj=[("ra",rasex), ("dec",decsex), ("radius","2"), ("units","arcsec"),
                        ("objname",""), ("internal_name","")]            
            response=search(tnsapi, search_obj, tnsapikey)
            json_data = format_to_json(response.text)
            if len(json_data['data']['reply']):
                tns_snid = json_data['data']['reply'][0]['objname']
                
                TNSGetSingle = [("objname",tns_snid),
                                ("photometry","0"),
                                ("spectra","1")]

                response=get(tnsapi, TNSGetSingle, tnsapikey)
                json_data = format_to_json(response.text)
            else: continue
        
        if len(json_data['data']['reply']['spectra']):
            spec_is_public = False
            for tnsspec in json_data['data']['reply']['spectra']:
                if np.abs(date_to_mjd(tnsspec['obsdate']) - sn.SPECTRA[0]['SPECTRUM_MJD']) < 1:
                    spec_is_public = True
                    if spec_is_public: print(sn.SNID,tnsspec['observer'])
            if spec_is_public: public_speccount += 1

    print(public_speccount)

    
if __name__ == "__main__":
    #main()
    #quicklist()
    comp_to_kyle()
