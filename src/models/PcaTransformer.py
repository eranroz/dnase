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
    def __init__(self, w=None, mu=0):
        self.w = w
        self.mu = mu  # used for recovery

    def fit(self, data, ndim=None, min_energy=0.9):
        """
        Transforms the data using linear transformation (w)
        @param data: data to transform
        @param ndim: number of dimensions (based on principle analysis)
        @param min_energy:
        @return:
        """
        self.mu = np.mean(data, 1)
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

    def recover(self, reduced):
        """
        Inverse for pca, reconstruction of the original data
        @param reduced: reduced data
        @return: reconstruction of original data
        """
        return np.dot(reduced, self.w)+self.mu

    def __call__(self, *args, **kwargs):
        data_centered = args[0] - self.mu[:, np.newaxis]  # np.mean(args[0], 1)[:, None]
        pca_matrix = np.dot(self.w, data_centered)
        return pca_matrix

    @staticmethod
    def empty(ndims):
        """
        Creates an identity linear transformation.
        @param ndims: number of dimensions
        @return: identity matrix transformation
        """
        transform = PcaTransformer(w=np.eye(ndims), mu=np.zeros(ndims))
        return transform
