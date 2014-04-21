"""
Base class for score functions
"""
from abc import ABCMeta, abstractmethod

__author__ = 'eranroz'


class BaseScoreModel:
    """
    Base class for score functions
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def score(self, model=None, resolution=None, training_set=None):
        """
        Score segmentation according to the specific score function
        @param model: model used for creating segmentation
        @param resolution: resolution in bp of bins used for segmentation
        @param training_set: array of segmentations (true for opened, false for closed domain)
        """
        pass