"""
Class for discretization.
"""
import numpy as np

__author__ = 'Eran'


class DiscreteTransformer(object):
    """
    A class to transform sequences to discrete alphabet
    @param percentiles: percentiles according to which assign letter (0-100).
                        Values between 0 to value of percentiles[0] assign letter 0. etc

    @note For consistence transform the class use the values of percentiles of the first sequence,
          for latter called sequences
    """

    def __init__(self, percentiles=[60, 75, 90]):
        self.percentiles = percentiles
        self.percentiles_values = None

    def __call__(self, *args, **kwargs):
        """
        Transforms the sequence to discrete alphabet
        """
        lg_data_n_zero = np.log(np.asfarray(args[0]) + 1)  # we don't want to get 0 for log
        #print('finding percentiles')
        if self.percentiles_values is None:
            self.percentiles_values = np.percentile(lg_data_n_zero, q=self.percentiles)

        mapped2 = np.zeros(len(lg_data_n_zero), dtype=int)# * len(self.percentiles_values)
        for i, bounds in enumerate(self.percentiles_values):
            mapped2[lg_data_n_zero > bounds] = i + 1
        return mapped2