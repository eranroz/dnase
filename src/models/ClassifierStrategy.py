"""
Base class for different strategies for genome classification
"""
__author__ = 'eranroz'
from abc import ABCMeta, abstractmethod
import numpy as np

class ClassifierStrategy(object):
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

    def name(self):
        """
        A friendly name for the strategy. If the strategy includes important details override it to include them
        @return: friendly name for the strategy
        """
        return self.__class__.__name__

    def preprocessing_html(self, data, input_labels, model_name):
        """
        html description of the preprocessing strategy of the classifier. (optional)
        @param model_name: associated model name
        @param input_labels: labels for the input data (features)
        @param data: data or sample of the data, which can be used to create fancy histograms
        @return: html code that describes the preprocessing strategy
        """
        return ""


class HmmClassifierStrategy(ClassifierStrategy):
    """
    A classification strategy that use HMM model to assign states

    model - some HMMModel
    """
    def __init__(self, model):
        self.model = model

    def num_states(self):
        """
        Number of states that can be assigned (excluding pseudo state of beginning)
        @return: number of states that can be assigned
        """
        return self.model.num_states()-1

    def states_names(self):
        """
        Gives the state by friendly name
        """
        state_names = [str(state) for state in np.arange(self.num_states())]
        # TODO: think of a nice way to assign names for the states? ior should it be model dependent
        return state_names

    def states_colormap(self):
        """
        Colormap between state and color
        """
        import matplotlib as mpl
        import matplotlib.cm as cm

        n_states = self.num_states()  # expect the begin state
        norm = mpl.colors.Normalize(vmin=0, vmax=n_states)
        cmap = cm.spectral
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        color_schema = dict()
        for i in range(0, n_states + 1):
            rgb = list(m.to_rgba(i)[:3])
            for j in range(0, 3):
                rgb[j] = int(255 * rgb[j])
            color_schema[i] = '#{:02x}{:02x}{:02x}'.format(*rgb)
        return color_schema

    @abstractmethod
    def fit(self, training_sequences):
        pass

    @abstractmethod
    def classify(self, sequence):
        pass

    @abstractmethod
    def data_transform(self):
        pass