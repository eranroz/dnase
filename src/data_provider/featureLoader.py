"""
This module is helper module to load features other than DNase.
All public functions returns dictionary object where the keys are chromosomes and values are according to specific data
"""
import logging
import re

__author__ = 'eranroz'

_chr_mapper = re.compile('^chr([0-9]+)$')
_other_chromosomes = dict()
_other_chromosomes_inv = dict()
_other_index = 26


def _chr_index_to_str(i):
    """
    maps chromosome index to str
    @param i: index of chromosome
    @return: string such as chrX, chr1
    """
    if i < 24:
        return 'chr%i' % i
    elif i == 24:
        return 'chrY'
    elif i == 25:
        return 'chrX'
    elif i == 26:
        return 'chrM'
    else:
        return _other_chromosomes_inv[i]


def _chr_str_to_index(chr_str):
    """
    mapps chromosome string to internal index
    @param chr_str: chromosome string such as chr1
    @return: internal index
    """
    global _other_index
    m = _chr_mapper.match(chr_str.decode('ascii'))
    if m is not None:
        return float(m.group(1))
    elif chr_str == b'chrY':
        return 24
    elif chr_str == b'chrX':
        return 25
    elif chr_str == b'chrM':
        return 26
    else:
        try:
            return _other_chromosomes[chr_str]
        except KeyError:
            _other_index += 1
            _other_chromosomes[chr_str] = _other_index
            _other_chromosomes_inv[_other_index] = chr_str.decode('ascii')
            return _other_index
            #print(chr_str)
            #raise Exception


def load_hmec_ctcf_peaks():
    """
    loads CTCF signal peaks from HMEC brest tissue
    @return: dict - keys chromosomes and values are signal peaks (values array of start col | end col | value col)
    """
    import os
    import numpy as np
    from config import OTHER_DATA

    ctcf_path = os.path.join(OTHER_DATA, "CTCF", "wgEncodeBroadHistoneHmecCtcfStdPk.broadPeak.gz")

    #chromosome, start, end, name, score, orientation, signalValue, pValue, qValue
    ctct_matrix = np.loadtxt(ctcf_path, delimiter='\t', usecols=(0, 1, 2, 7),
                             converters={0: _chr_str_to_index})  #0, 1, 2, 4, 6, 7

    dict_vals = dict((_chr_index_to_str(i), ctct_matrix[ctct_matrix[:, 0] == i, 1:]) for i in set(ctct_matrix[:, 0]))

    return dict_vals


def _download_known_genes(known_genes_path):
    """
    Downloads known gene table locally
    @param known_genes_path: path to place known gene table
    """
    import os

    try:
        import urllib.request

        urlret = urllib.request.urlretrieve
    except:
        import urllib.urlretrieve
        urlret = urllib.urlretrieve
    logging.info('known_genes table not found, download from UCSC')
    if not os.path.exists(os.path.dirname(known_genes_path)):
        os.makedirs(os.path.dirname(known_genes_path))

    urlret('http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/knownGene.txt.gz', known_genes_path)


def load_known_genes():
    """
    Loads knownGenes table

    @rtype dict
    @return:  keys- chromosomes, values - known genes array (tx start | tx end)
    """
    import os
    from config import OTHER_DATA
    import numpy as np

    known_genes_path = os.path.join(OTHER_DATA, "knownGenes", "knownGenes.txt.gz")
    if not os.path.exists(known_genes_path):
        _download_known_genes(known_genes_path)
    #columns:
    #name	chrom	strand	txStart	txEnd	cdsStart	cdsEnd	exonCount	exonStarts	exonEnds	proteinID	alignID
    # we skip chromosome such as chr6_apd_hap1
    known_genes_matrix = np.loadtxt(known_genes_path, delimiter='\t', dtype='int', usecols=(1, 3, 4),
                                    converters={1: _chr_str_to_index}, comments='#')

    dict_vals = dict((_chr_index_to_str(i), known_genes_matrix[known_genes_matrix[:, 0] == i, 1:]) for i in
                     set(known_genes_matrix[:, 0]))
    return dict_vals
