"""
Classifier of genome based on DNase experiments.

This class also serve as a model manager, so you can persist a trained classifier in hard drive,
and store related files in same directory
"""
import logging
import os
from config import MODELS_DIR

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
    @param model_name:
    @return:
    """
    model_path = os.path.join(model_dir(model_name), _model_file_name)
    if not os.path.exists(model_path):
        raise IOError("Model doesn't exist (%s)" % model_path)
    return pickle.load(model_path)


class DNaseClassifier(object):
    """
    Classifier for active and non-active areas in chromatin based on DNase data for a single sequence
    """

    def __init__(self, strategy, name="unnamed"):
        """
        Initializes a new instance of C{Classifier}

        @param strategy: strategy for classification
        @type strategy: C{ClassifierStrategy}
        """
        self.strategy = strategy
        self.name = name

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

    def save(self, warn=True):
        """
        Save the classifier model in models directory

        @param warn: whether to warn when override a model
        """
        path = os.path.join(MODELS_DIR, self.name)
        if not os.path.exists(path):
            if warn:
                logging.warning("Model already exist - overriding it")
                confirm = input("Do you want to override model \"%s\"? (Y/N)").upper().strip()
                while confirm not in ['Y', 'N']:
                    confirm = input("Do you want to override model \"%s\"? (Y/N)").upper().strip()

                if confirm == 'N':
                    return

        else:
            # create directory
            os.makedirs(path)

        with open(os.path.join(path, _model_file_name)) as model_file:
            pickle.dump(self, model_file)

    def model_dir(self):
        """
        Get the directory associated with this model

        @return: directory name associated with this model
        """
        return model_dir(self.name)

