"""
Classifier of genome based on DNase experiments.

This class also serve as a model manager, so you can persist a trained classifier in hard drive,
and store related files in same directory
"""
import logging
import os
import numpy as np
from config import MODELS_DIR, DATA_DIR, MEAN_DNASE_DIR
from data_provider import SeqLoader
from data_provider.LazyLoader import LazyChromosomeLoader
from data_provider.data_publisher import publish_dic

__author__ = 'eranroz'
import pickle

_model_file_name = "model.pkl"


def model_dir(model_name):
    """
    Get the directory associated with model
    @param model_name: name of model
    @return: directory of the model
    """
    return os.path.join(MODELS_DIR, model_name)


def load(model_name):
    """
    Loads a model
    @rtype : DNaseClassifier
    @param model_name: name of model to be loaded
    @return: a segmentation model
    """
    model_path = os.path.join(model_dir(model_name), _model_file_name)
    if not os.path.exists(model_path):
        raise IOError("Model doesn't exist (%s)" % model_path)
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def model_exist(model_name):
    """
    Check whether a model exist
    @param model_name: model name
    @return: true is such model exist otherwise false
    """
    return os.path.exists(os.path.join(model_dir(model_name), _model_file_name))


class DNaseMetaClassifier(object):
    """
    Meta class for classifiers: mono classifier, multichannel classifier etc.
    """

    def __init__(self, strategy, resolution, name="unnamed"):
        """
        Initializes a new instance of C{Classifier}

        @type resolution: int
        @param strategy: strategy for classification
        @type strategy: C{ClassifierStrategy}
        """
        self.strategy = strategy
        self.name = name
        self.resolution = resolution
        if not isinstance(resolution, int):
            raise Exception('Resolution must be int')

    def fit(self, training_sequences):
        """
        Fits the model before actually running it
        @param training_sequences: training sequences for the Baum-Welch (EM)
        """
        self.strategy.fit(training_sequences)

    def classify(self, sequence_dicts):
        """
        Classifies each sequence
        @param sequence_dicts: sequences to classify
        @return: generator for classified sequences
        """
        for seq in sequence_dicts:
            print("classifying sequence")
            yield self.strategy.classify(seq)

    def __str__(self):
        strategy_str = str(self.strategy)
        return strategy_str

    def save(self, warn=True):
        """
        Save the classifier model in models directory

        @param warn: whether to warn when override a model
        """
        path = os.path.join(MODELS_DIR, self.name)
        if os.path.exists(path):
            if warn:
                logging.warning("Model already exist - overriding it")
                confirm = input("Do you want to override model \"%s\"? (Y/N)" % path).upper().strip()
                while confirm not in ['Y', 'N']:
                    confirm = input("Do you want to override model \"%s\"? (Y/N)" % path).upper().strip()

                if confirm == 'N':
                    return

        else:
            # create directory
            os.makedirs(path)

        with open(os.path.join(path, _model_file_name), 'wb') as model_file:
            pickle.dump(self, model_file)

    def model_dir(self, join=""):
        """
        Get the directory associated with this model

        @param join: name of file within model directory
        @return: directory name associated with this model
        """
        return os.path.join(model_dir(self.name), join)

    def segmentation_file_path(self):
        """
        Get the associated segmentation file path associated with the model.
        """
        return self.model_dir("segmentation.npz")


class DNaseClassifier(DNaseMetaClassifier):
    """
    Classifier for active and non-active areas in chromatin based on DNase data for a single sequence
    """

    def fit_file(self, infile):
        """
        fit model based on input file
        @param infile: input file
        """
        transformer = self.strategy.data_transform()
        data = SeqLoader.load_dict(os.path.basename(infile), resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(infile) or DATA_DIR)  # TODO: can load only partial
        self.fit([data])

    def classify_file(self, file_name, chromosome=None):
        transformer = self.strategy.data_transform()

        data = SeqLoader.load_dict(file_name, resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(file_name) or DATA_DIR, chromosome=chromosome)
        return self.classify([data])

    def save_classify_file(self, file_name, out_file, save_raw=True, save_npz=True, save_bg=True):
        """
        Classifies file and saves it and related files to directory of the model
        @param file_name: name of file to classify
        @param out_file: name of output file without extension
        @param save_raw: whether to save raw data after transformation
        @param save_npz:  whether to save classified sequence as npz for later use
        @param save_bg: whether to save classified sequence as bg file (for UCSC)
        """
        transformer = self.strategy.data_transform()

        data = SeqLoader.load_dict(os.path.basename(file_name), resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(file_name) or DATA_DIR)
        path_to_save = self.model_dir()
        if save_raw:
            print('Writing raw file')
            #SeqLoader.build_bedgraph(data, resolution=self.resolution,
            #                         output_file=os.path.join(path_to_save, '%s.raw.bg' % out_file))
            publish_dic(data,
                        self.resolution,
                        '%s.%s.raw' % (self.name, out_file),
                        short_label="Raw %s" % out_file,
                        long_label="Raw file after transformation")
        segmentation = self.classify([data])
        for classified_seq in segmentation:
            if save_bg:
                print('Writing result file (bg format)')
                publish_dic(classified_seq, self.resolution, '%s.%s' % (self.name, out_file),
                            short_label="%s-%s" % (out_file, self.name),
                            long_label="HMM classification.  %s" % (str(self)))
            if save_npz:
                print('Writing result file (npz format)')
                SeqLoader.save_result_dict(os.path.join(path_to_save, '%s.npz' % out_file), classified_seq)
        return segmentation

    def load_data(self, infile, chromosomes=None):
        """
        loads the data of file
        """
        transformer = self.strategy.data_transform()
        data = SeqLoader.load_dict(infile, resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(infile) or DATA_DIR, chromosome=chromosomes)
        return data


class DNaseMultiChannelClassifier(DNaseMetaClassifier):
    """
    Classifier for multichannel

    strategy - should be C{GMMClassifier}
    """

    def __init__(self, strategy, resolution=1000, name=None):
        # assign default name for the model
        if name is None:
            name = 'multichannel%i' % resolution

        # whether to use sparse matrix representation during calculations
        # it seems that sparse matrix aren't required as num zeros is about 3% (chromosome 8, all cell types)
        # in resolution of 1000bp
        self.sparse = False
        super().__init__(strategy, resolution, name)

    def load_multichannel_data(self, directory=MEAN_DNASE_DIR, chromosomes=None):
        """
        Loads multichannel data. return a sparse representation
        @param directory: directory of files to analyze (directory with npz files)
        @param chromosomes: chromosomes to load
        @type directory: str
        @return: dict where keys are chromosomes and values are matrix: cell_types X genome position

        @remarks:
        """

        if self.sparse:
            import scipy.sparse
            compress = lambda x: scipy.sparse.coo_matrix(x)
            vstack = scipy.sparse.vstack
        else:
            compress = lambda x: x
            vstack = np.vstack
        all_cells = []
        # for debug: can be cut to 4 cells
        cell_types_paths = [os.path.join(directory, cell_type) for cell_type in os.listdir(MEAN_DNASE_DIR)]

        for cell_i, cell_type_path in enumerate(cell_types_paths):
            logging.info('Loading %s (%i/%i)' % (cell_type_path, cell_i+1, len(cell_types_paths)))
            cell_data = SeqLoader.load_result_dict(cell_type_path)
            cell_data_new = dict()  # chromosome to down-sampled sparse matrix

            for k in (cell_data.keys() if chromosomes is None else chromosomes):
                cell_data_new[k] = compress(SeqLoader.down_sample(cell_data[k], self.resolution / 20))
            all_cells.append(cell_data_new)

        # organize be chromosomes
        if chromosomes is None:
            chromosomes = all_cells[0].keys()
        chromosomes_dic = dict()
        len_dim = 1 if self.sparse else 0
        for chromosome in chromosomes:

            max_length = max([cell[chromosome].shape[len_dim] for cell in all_cells])
            rows_matrices = []
            for cell in all_cells:
                if max_length > cell[chromosome].shape[len_dim]:
                    tmp = np.zeros(max_length)
                    if self.sparse:
                        tmp[0:cell[chromosome].shape[len_dim]] = cell[chromosome].todense()
                    else:
                        tmp[0:cell[chromosome].shape[len_dim]] = cell[chromosome]
                    rows_matrices.append(compress(tmp))
                else:
                    rows_matrices.append(cell[chromosome])
            # each row represent different cell, each column different position
            chromosome_matrix = vstack(rows_matrices)

            chromosomes_dic[chromosome] = chromosome_matrix
        return chromosomes_dic

    def load_data(self, directory=MEAN_DNASE_DIR):
        """
        Lazy loads data from directory
        @param directory: directory of files to analyze (directory with npz files)
        @return: dict where keys are chromosomes and values are matrix: cell_types X genome position
        """
        transform = self.strategy.data_transform()
        loader = LazyChromosomeLoader(lambda x: transform(self.load_multichannel_data(chromosomes=[x],
                                                                                      directory=directory)[x]))
        return loader

    def classify_data(self, data):
        """
        Classifies each sequence
        @param data: sequences to classify
        @return: chromosome dictionary with classified data
        """
        return self.strategy.classify(data)