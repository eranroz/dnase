"""
Handles multidimensional huge data.

since it requires huge size memory:
 - we use the mean from different cell types instead of just using samples.
 - we use PCA to reduce the number of cell types

There are two approaches:
1. Discrete - discretization for words for each sequence, and then building words by combining them
2. Continuous -  Use the real values of the channel with multi dimensional gaussian and covariance matrix to evaluate


Assumptions:
    Just as an estimation for the size: 242 cells x 2492506 chromosome 1 size (bins of size 100)
    requires 4.5Gb

see also:
    multichannel_classify - script for multichannel classifications
"""
import numpy as np
from dnase.ClassifierStrategy import ClassifierStrategy
from dnase.PcaTransformer import PcaTransformer
from hmm.HMMModel import GaussianHMM, DiscreteHMM
from hmm.bwiter import bw_iter, IteratorCondition, DiffCondition

__author__ = 'eranroz'


class GMMClassifier(ClassifierStrategy):
    """
    multivariate version of HMMClassifier for multichannel data

    * It uses PCA to reduce number of learned channels
    * It adds some functions for smart selection of the initial state
    """

    def __init__(self, model=None, pca_reduction=None, train_chromosome='chr1'):
        """
        @type model: GaussianHMM
        @param model: GaussianHMM to model the multichannel data
        """
        self.model = model
        self.pca_reduction = pca_reduction
        self.train_chromosome = train_chromosome

    def training_chr(self, chromosome):
        """
        Specifies on which chromosome we want to train or fit the model
        @param chromosome: chromosome name for training
        @return: None
        """
        self.train_chromosome = chromosome

    def fit(self, data, iterations=None, energy=0.9, pca_components=None):
        """
        fits the classifiers to training sequences and returns the log likelihood after fitting
        @param pca_components: number of dimensions to use for PCA (set energy to None)
        @param energy: cumulative energy to use for pca (set pca_components to None)
        @param data: data to use for PCA reduction matrix selection
        @param iterations: number of iterations number of iteration
        @return: likelihood for the model based on the model
        """
        old_model = self.model
        print("Starting fitting")

        training_seqs = data[self.train_chromosome]

        if self.pca_reduction is None:
            transformer = PcaTransformer()
            self.pca_reduction = transformer.fit(training_seqs[0], min_energy=energy, ndim=pca_components)
        else:
            transformer = PcaTransformer(w=self.pca_reduction)
        training_seqs = transformer(training_seqs)

        #TODO: use different sequences?
        self.model, p = bw_iter(training_seqs, self.model,
                                IteratorCondition(iterations) if iterations is not None else DiffCondition())

        print("Model fitting finished. likelihood", p)
        print("Old model")
        print(old_model)
        print("New model")
        print(self.model)
        return p

    def classify(self, sequence_dict):
        """
        Classifies chromosomes across samples (such as different tissues)
        @param sequence_dict: dict like object with keys as chromosomes and values as matrix
        @return: viterbi state assignment for the genome
        """
        classified = dict()
        transformer = PcaTransformer(w=self.pca_reduction)
        for chromosome, sequence in sequence_dict.items():
            print('Classifying chromosome', chromosome)
            # reduce dimensions
            sequence = transformer(sequence)
            # fit
            classified[chromosome] = self.model.viterbi(sequence)

        return classified

    def data_transform(self):
        """
        get associated data transformation pre-processing
        @return: log(x+1)
        """
        return lambda x: np.log(np.array(x) + 1)

    def default(self, data, train_chromosome='chr8', num_states=10):
        """
        Default initialization for GMM classifier with:
        * "training" for PCA (based on train chromosome covar
        * heuristic selection of number of state and their emission (soft k means)
        * state transition - random initialization with some prior assumptions
        @type train_chromosome: str
        @type num_states: int
        @param data: data (or partial data) to use for selection of pca transformation, and k-means for states
                    (initial guess). dictionary like object
        @param train_chromosome: chromosome to use for training (must be in data. eg data[train_chromosome]
        @param num_states: number of states in HMM
        """
        chrom_data = data[train_chromosome]
        transformer = PcaTransformer()
        pca_reduction = transformer.fit(chrom_data)
        chrom_data = transformer(chrom_data)

        emission = GMMClassifier._continuous_state_selection(chrom_data, num_states=num_states)
        n_states = len(emission) + 1  # number of states plus begin state
        print('Number of states selected %i' % (n_states-1))
        state_transition = np.random.random((n_states, n_states))
        # fill diagonal with higher values
        np.fill_diagonal(state_transition, np.sum(state_transition, 1))
        state_transition[:, 0] = 0  # set transition to begin state to zero
        # normalize
        state_transition /= np.sum(state_transition, 1)[:, np.newaxis]
        # initial guess
        initial_model = GaussianHMM(state_transition, emission)
        self.model = initial_model
        self.pca_reduction = pca_reduction
        self.train_chromosome = train_chromosome

    @staticmethod
    def default_strategy(data, train_chromosome='chr8', num_states=10):
        """
        Creates a default GMM classifier with heuristic guess (see default)
         @type train_chromosome: str
        @type num_states: int
        @param data: data (or partial data) to use for selection of pca transformation, and k-means for states
                    (initial guess). dictionary like object
        @param train_chromosome: chromosome to use for training (must be in data. eg data[train_chromosome]
        @param num_states: number of states in HMM
        @return: a GMM classifier
        """
        classifier = GMMClassifier()
        classifier.default(data, train_chromosome, num_states)
        return classifier

    @staticmethod
    def _continuous_state_selection(data, num_states):
        """
        Heuristic creation of emission for states in continuous multidimensional model.
        Instead of random selection of the emission matrix we find clusters of co-occurring values,
        and use those clusters as means for states and the close values as estimation for covariance matrix

        @param visualize: whether to visualize the state selection
        @param num_states: number of states in model
        @param data: dense data for specific chromosome
        @return: initial emission for gaussian mixture model HMM (array of (mean, covariance)
        """
        def soft_k_means_step(data, clusters):
            """
            Soft k means
            @param data: data to cluster
            @param clusters: number of clusters
            @return: new clusters means
            """
            w = np.array([np.sum(np.power(data - c, 2), axis=1) for c in clusters])
            w /= ((np.max(w)+np.mean(w)) / 1000)  # scale w
            w = np.minimum(w, 500)  # 500 is enough (to eliminate underflow)
            w = np.exp(-w)
            w = w / np.sum(w, 0)  # normalize for each point
            w = w / np.sum(w, 1)[:, None]  # normalize for all cluster
            new_clusters = np.dot(w, data)
            return new_clusters
        data = data.T
        num_sub_samples = 2
        sub_indics = np.random.permutation(np.arange(data.shape[0] - data.shape[0] % num_sub_samples))
        n_clusters = num_states or data.shape[1] * 2  # number of clustering will be subject to pruning

        clusters = np.random.random((n_clusters, data.shape[1])) * np.max(data, 0)

        # once we have assumption for clusters work with real sub batches of the data
        sub_indics = sub_indics.reshape(num_sub_samples, -1)

        different_clusters = False
        step = 0
        while not different_clusters:
            diff = np.ones(1)
            iter_count = 0
            while np.any(diff > 1e-1) and iter_count < 10:
                sub_data = data[sub_indics[step % num_sub_samples], :]
                new_clusters = soft_k_means_step(sub_data, clusters)
                diff = np.sum((new_clusters - clusters) ** 2, axis=1)
                clusters = new_clusters
                iter_count += 1
            step += 1

            if num_states:
                different_clusters = True
            else:
                dist_matrix = np.array([np.sum(np.power(clusters - c, 2), axis=1) for c in clusters])
                np.fill_diagonal(dist_matrix, 1000)
                closest_cluster = np.min(dist_matrix)
                threshold = 2*np.mean(dist_matrix)/np.var(dist_matrix)  # or to just assign 0.1?
                if closest_cluster < threshold:
                    # pruning the closest point and add random to close points
                    subject_to_next_prune = list(set(np.where(dist_matrix < threshold)[0]))
                    clusters[subject_to_next_prune, :] += 0.5 * clusters[subject_to_next_prune, :] * np.random.random(
                        (len(subject_to_next_prune), data.shape[1]))
                    clusters = clusters[np.arange(n_clusters) != np.where(dist_matrix == closest_cluster)[0][0], :]
                    n_clusters -= 1
                else:
                    different_clusters = True

        # now assign points to clusters
        clusters = clusters[np.argsort(np.sum(clusters ** 2, 1))]  # to give some meaning
        W = np.array([np.sum(np.power(data - c, 2), axis=1) for c in clusters])
        W /= (np.mean(W) / 500)  # scale w
        W = np.minimum(W, 500)
        W = np.exp(-W)
        W /= np.sum(W, 0)  # normalize for each point
        W /= np.sum(W, 1)[:, None]  # normalize for all cluster
        means = np.dot(W, data)
        # and add some randomality
        means += means*np.random.random(means.shape)*0.1
        covs = []
        min_std = np.finfo(float).eps
        for mu, p in zip(means, W):
            seq_min_mean = data - mu
            new_cov = np.dot((seq_min_mean.T * p), seq_min_mean)
            new_cov = np.maximum(new_cov, min_std)
            covs.append(new_cov)
        means_covs = list(zip(means, covs))
        return means_covs

    def __str__(self):
        return str(self.model)


class DiscreteMultichannelHMM(ClassifierStrategy):
    """
    A model for discrete multichannel HMM:
    data [position x tissue] =(PCA)> data [position x tissue combination] => discretization => word encoding => HMM
    """

    def __init__(self):
        self.model = None
        self.pca_reduction = None

    def classify(self, sequence):
        raise NotImplementedError

    def fit(self, data):
        # TODO: only partially implemented here not tested...
        raise NotImplementedError
        from scipy.stats import norm as gaussian
        min_alpha = 0
        n_words = np.max(data)
        # init hmm model
        n_states = 5
        state_transition = np.zeros(n_states + 1)
        # begin state
        state_transition[0, 1:] = np.random.rand(n_states)
        # real states - random with some constraints. state 1 is most closed, and n is most opened
        real_states = np.random.rand((n_states, n_states))
        # set strong diagonal
        diagonal_selector = np.eye(n_states, dtype='bool')
        real_states[diagonal_selector] = np.sum(real_states, 1) * 9
        real_states /= np.sum(real_states, 1)[:, None]
        state_transition[1:, 1:] = real_states
        # normalize

        # emission
        emission = np.zeros((n_states + 1, n_words))
        real_emission = np.random.random((n_states, n_words))
        for i in np.arange(0, n_states):
            mean = i * (n_words / n_states)
            variance = (n_words / n_states)
            real_emission[i, :] = gaussian(mean, variance).pdf(np.arange(n_words))
        real_emission /= np.sum(real_emission, 1)[:, None]
        emission[1:, 1:] = real_emission

        # init hmm
        print('Creating model')

        self.model = DiscreteHMM(state_transition, emission, min_alpha=min_alpha)
        print('Training model')


    def data_transform(self):
        """
        get associated data transformation prepossessing
        """
        if self.pca_reduction is None:
            return lambda x: x
        else:
            return lambda x: DiscreteMultichannelHMM.preprocess(self.pca_reduction(x))

    @staticmethod
    def preprocess(data):
        discrete = DiscreteMultichannelHMM.multichannel_discrete_transform(data)
        multichannel_data = DiscreteMultichannelHMM.encode_discrete_words(discrete)
        return multichannel_data

    @staticmethod
    def encode_discrete_words(data):
        """
        Transforms a discrete matrix to one dimensional words
        @param data: discrete matrix
        @return: words array
        """
        new_data = np.zeros(data.shape[1])
        alphbet = np.power(2, np.arange(data.shape[0] * np.max(data)))
        alphbet_assign = enumerate(alphbet)
        # transform to powers of 2
        for i in np.arange(0, np.max(data) + 1):
            for j in np.arange(0, new_data.shape[0]):
                selector = (data[j, :] == i)
                data[j, selector] = next(alphbet_assign)

        for cell in data:
            # bitwise or
            new_data |= cell

        return new_data

    @staticmethod
    def multichannel_discrete_transform(data, percentiles=[60, 75, 90]):
        """
        Transforms a matrix from continuous values to discrete values
        @param percentiles: percentiles used for discretization
        @param data: continuous values matrix
        @return: discrete values
        """
        data = np.log(data + 1)
        prec_values = np.percentile(data, q=percentiles)
        max_val = np.max(data) + 1
        min_val = np.min(data) - 1
        new_chrom_data = np.zeros_like(data)
        for i, vals in enumerate(zip([min_val] + prec_values, prec_values + [max_val])):
            new_chrom_data[(data >= vals[0]) & (data < vals[1])] = i
        return new_chrom_data