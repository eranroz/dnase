"""
This module is helper module to load features other than DNase.
All public functions returns dictionary object where the keys are chromosomes and values are according to specific data
"""
import logging
import re
import os

try:
    import urllib.request
    urlret = urllib.request.urlretrieve
except:
    import urllib.urlretrieve
    urlret = urllib.urlretrieve

from config import DATA_DIR

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
    from config import GENOME

    logging.info('known_genes table not found, download from UCSC')
    if not os.path.exists(os.path.dirname(known_genes_path)):
        os.makedirs(os.path.dirname(known_genes_path))

    urlret('http://hgdownload.cse.ucsc.edu/goldenPath/{}/database/knownGene.txt.gz'.format(GENOME), known_genes_path)


def load_mapability(kmer=50):
    from config import GENOME
    from data_provider import SeqLoader


    bigwig_file = 'wgEncodeCrgMapabilityAlign{}mer.bigWig'.format(kmer)
    cached_path = os.path.join(DATA_DIR, "ucscFiles", bigwig_file)
    npz_mapability = cached_path.replace('.bigWig', '')

    if not os.path.exists(cached_path):
        urlret('http://hgdownload.cse.ucsc.edu/gbdb/{}/bbi/{}'.format(GENOME, bigwig_file), cached_path)
        with tempfile.NamedTemporaryFile('w+', encoding='ascii') as tmp_file:
            subprocess.call([BIG_WIG_TO_BED_GRAPH, cached_path, tmp_file.name])
            seq = SeqLoader.load_bg(tmp_file.name)
            SeqLoader.save_result_dict(npz_mapability, seq)
    return SeqLoader.load_result_dict(npz_mapability+'.npz')

def load_known_genes(columns=['txStart', 'txEnd'], dtype='int'):
    """
    Loads knownGenes table

    @rtype dict
    @return:  keys- chromosomes, values - known genes array (tx start | tx end)
    """
    from config import DATA_DIR
    import numpy as np

    known_genes_path = os.path.join(DATA_DIR, "knownGenes", "knownGenes.txt.gz")
    if not os.path.exists(known_genes_path):
        _download_known_genes(known_genes_path)
    #columns:
    columns_schema = ['name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonCount', 'exonStarts', 'exonEnds', 'proteinID', 'alignID']
    #
    # we skip chromosome such as chr6_apd_hap1
    cols_indics = [columns_schema.index(c) for c in columns]
    required_converters = {1: _chr_str_to_index}
    if 'strand' in columns:
        required_converters[2] = lambda x: 1 if x == b'+' else -1
    known_genes_matrix = np.loadtxt(known_genes_path, delimiter='\t', dtype=dtype, usecols=[1] + cols_indics,
                                    converters=required_converters, comments='#')

    call_mapping = lambda i: _chr_index_to_str(i)
    if dtype != 'int' :
        call_mapping = lambda i: _chr_index_to_str(float(i))
    dict_vals = dict((call_mapping(i), known_genes_matrix[known_genes_matrix[:, 0] == i, 1:]) for i in
                     set(known_genes_matrix[:, 0]))
    return dict_vals