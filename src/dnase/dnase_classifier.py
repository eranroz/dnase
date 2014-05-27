"""
Classifier of genome based on DNase experiments.

This class also serve as a model manager, so you can persist a trained classifier in hard drive,
and store related files in same directory
"""
import logging
import os
from config import MODELS_DIR, DATA_DIR
from data_provider import SeqLoader
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


class DNaseClassifier(object):
    """
    Classifier for active and non-active areas in chromatin based on DNase data for a single sequence
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

    def load_data(self, infile, chromosomes=None):
        transformer = self.strategy.data_transform()
        data = SeqLoader.load_dict(infile, resolution=self.resolution, transform=transformer,
                                   directory=os.path.dirname(infile) or DATA_DIR, chromosome=chromosomes)
        return data

    def segmentation_file_path(self):
        """
        Get the associated segmentation file path associated with the model.
        """
        return self.model_dir("segmentation.npz")

