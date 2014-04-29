"""
Use this file to define directories that various scripts use.

"""
__author__ = 'eran'
import os

# directory with all dnase data as files of { 'chr1': array, 'chr2': array } pickled and zip
DATA_DIR = os.path.abspath('../data/')
RAW_DATA_OTHER = os.path.abspath('../raw_other_data')  # backup directory to save original data files
OTHER_DATA = os.path.abspath('../other_data/')  # directory for other biological signal data
RES_DIR = os.path.abspath('../results/')  # directory for results
BIN_DIR = os.path.abspath('../bin/')  # directory for executable of transformers
PUBLISH_DIR = os.path.abspath('../results/hub/')  # directory accessible to web for publish results
BED_GRAPH_RESULTS_DIR = os.path.join(RES_DIR, 'open-closed', 'data')  # directory for segmentation
NCBI_DIR = '...'

# dependencies - use dataDownloader.setup_environment or install.sh in bin directory
WIG_TO_BIG_WIG = os.path.join(BIN_DIR, 'wigToBigWig')
BIG_WIG_TO_BED_GRAPH = os.path.join(BIN_DIR, 'bigWigToBedGraph')
BED_GRAPH_TO_BIG_WIG = os.path.join(BIN_DIR, 'bedGraphToBigWig')
CHROM_SIZES = os.path.join(BIN_DIR, 'hg19.chrom.sizes')


# can be created by createMeanMarkers
MEAN_MARKERS = os.path.join(OTHER_DATA, 'bedMarkersNpz')  # directory for storing mean between samples of specific cell type
#MEAN_MARKERS_NEW = os.path.join(OTHER_DATA, 'bedMarkersNpz')  # directory for storing mean between samples with npz


# models
MODELS_DIR = os.path.join(RES_DIR, "models")

PUBLISH_URL_PATH = 'http://...'  # accessible url to expose results to genome browser