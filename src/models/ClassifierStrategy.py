"""
Base class for different strategies for genome classification
"""
__author__ = 'eranroz'
from abc import ABCMeta, abstractmethod


class ClassifierStrategy:
    """
    Strategy for segmentation
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, training_sequences):
        """
        Trains/Fits the classifier before actual classification
        @param training_sequences: training sequences for the Baum-Welch (EM)
        @return tuple (p-likelihood, fit_params)
        """
        pass

    @abstractmethod
    def classify(self, sequence):
        """
        Classifies sequence to open and closed area
        @param sequence: sequence to classify
        """
        pass

    @abstractmethod
    def data_transform(self):
        """
        Get data transformer for transforming raw data before classify or fit
        """
        pass
