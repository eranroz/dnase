"""
Pca transformer for reducing number of dimensions
"""
import logging
import numpy as np

__author__ = 'eranroz'


class PcaTransformer(object):
    """
    Reduce number of dimensions using PCA
    """
    def __init__(self, w=None):
        self.w = w

    def fit(self, data, ndim=None, min_energy=0.9):
        data_centered = data - np.mean(data, 1)[:, None]
        co_var = np.cov(data_centered)
        eig_vals, eig_vecs = np.linalg.eig(co_var)
        eig_order = eig_vals.argsort()[::-1]  # just to be sure we have eigenvalues ordered
        eig_vals = eig_vals[eig_order] / np.sum(eig_vals)
        eig_vecs = eig_vecs[eig_order]
        if ndim is None:
            explains = np.cumsum(eig_vals)
            ndim = np.where(explains > min_energy)[0][0]
            logging.info('PcaTransformer selected %i dims (original: %i), which explains %i%% of the data' % (ndim, data_centered.shape[0], np.round(explains[ndim]*100)))

        self.w = eig_vecs[0:ndim, :]
        return self.w

    def __call__(self, *args, **kwargs):
        data_centered = args[0] - np.mean(args[0], 1)[:, None]
        pca_matrix = np.dot(self.w, data_centered)
        return pca_matrix