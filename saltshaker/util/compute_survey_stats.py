import numpy as np
import logging

def sn_numbers(datadict):

    surveys = []
    n_spectra = []
    for k in datadict.keys():
        surveys += [datadict[k].survey]
        n_spectra += [len(datadict[k].specdata.keys())]
    surveys,n_spectra = np.array(surveys),np.array(n_spectra)

    unique_surveys,counts = np.unique(surveys,return_counts=True)

    survey_stats_dict = {}
    for u,c in zip(unique_surveys,counts):
        ns = np.sum(n_spectra[surveys == u])
        logging.info(f'survey: {u}, N_SNE: {c}, N_spectra: {ns}')
        survey_stats_dict[u] = (c,ns)

    return survey_stats_dict
