"""
Use this file to define directories that various scripts use.

"""
__author__ = 'eran'
import os

GENOME = 'hg19'

# directory with all dnase data as files of { 'chr1': array, 'chr2': array } pickled and zip
DATA_DIR = os.path.abspath('../data/')
RAW_DATA_OTHER = os.path.abspath('../raw_other_data')  # backup directory to save original data files
OTHER_DATA = os.path.abspath('../other_data/')  # directory for other biological signal data
RES_DIR = os.path.abspath('../results/')  # directory for results
BIN_DIR = os.path.abspath('../bin/')  # directory for executable of transformers
PUBLISH_DIR = os.path.abspath('../results/hub/')  # directory accessible to web for publish results
BED_GRAPH_RESULTS_DIR = os.path.join(RES_DIR, 'open-closed', 'data')  # directory for segmentation
SIGNAL_DIR = os.path.join(DATA_DIR, 'data')  # directory with serialized data (npz format)

# dependencies - use dataDownloader.setup_environment or install.sh in bin directory
WIG_TO_BIG_WIG = os.path.join(BIN_DIR, 'wigToBigWig')
BIG_WIG_TO_BED_GRAPH = os.path.join(BIN_DIR, 'bigWigToBedGraph')
BED_GRAPH_TO_BIG_WIG = os.path.join(BIN_DIR, 'bedGraphToBigWig')
BED_TO_BIG_BED = os.path.join(BIN_DIR, 'bedToBigBed')
CHROM_SIZES = os.path.join(BIN_DIR, '{}.chrom.sizes'.format(GENOME))


# can be created by createMeanMarkers
MEAN_MARKERS = os.path.join(DATA_DIR, 'markers')  # directory for storing mean between samples of specific cell type
MEAN_DNASE_DIR = os.path.join(DATA_DIR, "represent", "mean_dnaseNpz")

# models
MODELS_DIR = os.path.join(RES_DIR, "models")

# publish
PUBLISH_URL_PATH_HUB = 'http://...'  # accessible url to expose results to genome browser
TRACK_DESCRIPTION_TEMPALTE = os.path.abspath(os.path.join("data_provider", "trackDescriptionTemplate.html"))

# other configurations
BIO_DATA_CREDITS = ''  # credit for the data (or the source of the biological data)