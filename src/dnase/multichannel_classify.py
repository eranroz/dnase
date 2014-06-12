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


TODO:
- Make HMM methods work on multidimensional data
    - for discrete (should it just work?)
    - for continuous - multivariate gaussian mixture requires fixing mainly the maximization step and
       to use {hmm.multivariatenormal}
- Dump multi-cell data to sparse matrix for each chromosome
- many of the samples have zeros, we may use scipy.sparse to keep it in memory efficient.
   - scipy.sparse.lil_matrix to build a HUGE matrix of data from all cells.
   - scipy.sparse.csc to go over columns during HMM manipulations

"""
#TODO: implement it properly
import argparse
import os
import pickle
import datetime

import numpy as np

from config import OTHER_DATA, NCBI_DIR, BED_GRAPH_RESULTS_DIR, DATA_DIR, MEAN_DNASE_DIR, MODELS_DIR, RES_DIR
from data_provider import SeqLoader
from data_provider.data_publisher import publish_dic
from dnase.dnase_classifier import DNaseClassifier
from dnase.HMMClassifier import HMMClassifier
from hmm import bwiter
from hmm.HMMModel import DiscreteHMM, GaussianHMM
from hmm.multivariatenormal import MultivariateNormal
from data_provider import data_publisher


__author__ = 'eranroz'


def load_multichannel(resolution=100, chromosomes=None):
    """
    Loads multichannel data. return a sparse representation
    @type chromosomes: list
    @param chromosomes: chromosomes to load
    @param resolution: resolution of the required data
    @return: dict where keys are chromosomes and values are matrix: cell_types X genome position
    """
    import scipy.sparse

    all_cells = []
    cell_types_paths = [os.path.join(MEAN_DNASE_DIR, cell_type) for cell_type in os.listdir(MEAN_DNASE_DIR)]
    #cell_types_paths = [os.path.join(MEAN_DNASE_DIR, cell_type) for cell_type in os.listdir(MEAN_DNASE_DIR)][:4]  # 4 is for debug
    for cell_type_path in cell_types_paths:
        print('Loading %s' % cell_type_path)
        cell_data = SeqLoader.load_result_dict(cell_type_path)
        cell_data_new = dict()  # chromosome to down-sampled sparse matrix

        for k in (cell_data.keys() if chromosomes is None else chromosomes):
            cell_data_new[k] = scipy.sparse.coo_matrix(
                SeqLoader.down_sample(cell_data[k], resolution / 20))  #  TODO: use csr_matrix or coo_matrix?
        all_cells.append(cell_data_new)


    # organize be chromosomes
    if chromosomes is None:
        chromosomes = all_cells[0].keys()
    chrom_dic = dict()
    for chrom in chromosomes:
        max_length = max([cell[chrom].shape[1] for cell in all_cells])
        rows_matrices = []
        for cell in all_cells:
            if max_length > cell[chrom].shape[1]:
                tmp = np.zeros(max_length)
                tmp[0:cell[chrom].shape[1]] = cell[chrom].todense()
                rows_matrices.append(scipy.sparse.coo_matrix(tmp))
            else:
                rows_matrices.append(cell[chrom])
            chrom_mat = scipy.sparse.vstack(rows_matrices)

        chrom_dic[chrom] = chrom_mat
    return chrom_dic


def pca_chrom_matrix(data, dim=None, min_energy=0.8):
    new_data = dict()
    for k, chrom_matrix in data.items():
        chrom_matrix_centered = chrom_matrix - np.mean(chrom_matrix, 1)[:, None]
        co_var = np.cov(chrom_matrix_centered)
        eig_vals, eig_vecs = np.linalg.eig(co_var)
        eig_order = eig_vals.argsort()[::-1]  # just to be sure we have eigvalues ordered
        eig_vals = eig_vals[eig_order] / np.sum(eig_vals)
        eig_vecs = eig_vecs[eig_order]
        if dim is None:
            explains2 = np.cumsum(eig_vals)  # TODO: replace the next with it
            explains = [np.sum(eig_vals[0:d + 1]) for d in np.arange(0, len(eig_vals))]
            dim = [d for d, ex in enumerate(explains) if ex > min_energy][0]

        pca_matrix = np.dot(eig_vecs[0:dim, :], chrom_matrix_centered)
        new_data[k] = pca_matrix
    return new_data


def multichannel_discrete_transform(data, percentiles=[60, 75, 90]):
    """
    Transforms a matrix from continuous values to discrete values
    @param data: continuous values matrix
    @return: discrete values matrix
    """
    new_data = dict()
    for chrom, chrom_data in data.items():
        data = np.log(data + 1)
        prec_values = np.percentile(data, q=percentiles)
        max_val = np.max(data) + 1
        min_val = np.min(data) - 1
        new_chrom_data = np.zeros_like(chrom_data)
        for i, vals in enumerate(zip([min_val] + prec_values, prec_values + [max_val])):
            new_chrom_data[(data >= vals[0]) & (data < vals[1])] = i
        new_data[chrom] = new_chrom_data

    return new_data


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


def multichannel_hmm_discrete(resolution, model_name=None, output_p=False, out_file=None):
    """
    Use multichannel HMM to classify Dnase

    the following function implements the discrete approach
    """
    from scipy.stats import norm as gaussian

    default_name = 'discreteMultichannel'
    model_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                              '%s.Discrete%i.model' % (model_name or default_name, resolution))

    min_alpha = 0
    # load channels - cell types. use cell_type_mean() to create those files
    multichannel_data = load_multichannel(resolution)
    # uca PCA to reduce number of dimensions:
    # many cell types are expected to behave similarly
    # so it will be more efficient to reduce dimensions here
    cells_dimensions = 5
    multichannel_data = pca_chrom_matrix(multichannel_data)  # , cells_dimensions_

    # discrete transform
    multichannel_data = multichannel_discrete_transform(multichannel_data)
    n_words = np.power(2, np.arange(multichannel_data.shape[0] * np.max(multichannel_data)))
    # encode to words
    multichannel_data = encode_discrete_words(multichannel_data)

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

    model = DiscreteHMM(state_transition, emission, min_alpha=min_alpha)
    strategy = HMMClassifier(model)
    strategy.output_p = False
    classifier = DNaseClassifier(strategy)
    print('Training model')

    if os.path.exists(model_name):
        print('Skipped training - a model already exist')
        with open(model_name, 'rb') as model_file:
            strategy.model = pickle.load(model_file)
    else:
        start_training = datetime.datetime.now()
        classifier.fit([multichannel_data])
        print('Training took', datetime.datetime.now() - start_training)
        with open(model_name, 'wb') as model_file:
            pickle.dump(strategy.model, model_file)

    print('Classifying')
    class_mode = 'posterior' if output_p else 'viterbi'
    for classified_seq in classifier.classify([multichannel_data]):
        file_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                                 '%s.Discrete%i.%s.bg' % (out_file or default_name, resolution, class_mode))
        npz_file_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                                     '%s.Discrete%i.%s.npz' % (out_file or default_name, resolution, class_mode))
        print('Writing result file (bg format)')
        SeqLoader.build_bedgraph(classified_seq, resolution=resolution, output_file=file_name)
        print('Writing result file (pkl format)')
        SeqLoader.save_result_dict(npz_file_name, classified_seq)
        print('Writing raw file')
        data_publisher.publish_dic(multichannel_data, resolution, '%s.%i.rawDiscrete' % (default_name, resolution))


def soft_k_means_step(data, clusters):
    """
    Soft k means
    @param data: data to cluster
    @param clusters: number of clusters
    @return: new clusters means
    """
    W = np.array([np.sum(np.power(data - c, 2), axis=1) for c in clusters])
    W = np.minimum(W, 500)  # 500 is enough (to eliminate underflow)
    W = np.exp(-W)
    W = W / np.sum(W, 0)  # normalize for each point
    W = W / np.sum(W, 1)[:, None]  # normalize for all cluster
    new_clusters = np.dot(W, data)
    diff = np.sum((new_clusters - clusters) ** 2, axis=1)
    print(diff)
    return new_clusters


def continuous_state_selection(data, num_states=None, visualize=False):
    """
    Heuristic creation of emission for states in continuous multidimensional model.
    Instead of random selection of the emission matrix we find clusters of co-occurring values,
    and use those clusters as means for states and the close values as estimation for covariance matrix

    @param visualize: whether to visualize the state selection
    @param num_states: number of states in model
    @param data: dense data for specific chromosome
    @return: initial emission for gaussian mixture model HMM (array of (mean, covariance)
    """
    data = data.T
    if visualize:
        from matplotlib.pylab import plt
        # TODO: we can use PCA but for now just select 2 randoms dimensions
        dims = [2, 1]  #np.random.permutation(np.arange(data.shape[1]))[:2]
        plt.hexbin(data[:, dims[0]], data[:, dims[1]], alpha=0.7, edgecolors='w', cmap=plt.cm.OrRd,
                   extent=[np.floor(np.min(data[:, dims[0]]) - 1.0), np.ceil(np.max(data[:, dims[0]])),
                           np.floor(np.min(data[:, dims[1]]) - 1.0), np.ceil(np.max(data[:, dims[1]]))], bins='log')  #

    sub_indics = np.random.permutation(np.arange(data.shape[0] - data.shape[0] % 4))
    """
    # select sample for spectral clustering. spectral clustering
    # is used to assign number of states
    sub_data = data[sub_indics[:2000], :]
    scale = 0.5
    affinity_matrix = np.exp(-pdist(sub_data, 'sqeuclidean')/(2*scale**2))
    affinity_matrix = squareform(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    D = np.diag(1.0/np.sum(affinity_matrix, 1))
    L = np.dot(np.dot(D, affinity_matrix), D)
    eigvalues = np.linalg.eigvals(L)
    # sort by eigenvalues
    eigvalues = eigvalues[eigvalues.argsort()[::-1]]
    # select number of components according to eigvalues
    n_clusters = np.diff((eigvalues[eigvalues > 0])/np.sum(eigvalues[eigvalues > 0])) # 6

    n_clusters = np.where(n_clusters > -1e-3)[0]
    n_clusters = np.ceil((n_clusters[0]*0.8 + n_clusters[1]*0.2))
    """
    n_clusters = num_states or data.shape[1] * 2  # number of clustering will be subject to pruning


    clusters = np.random.random((n_clusters, data.shape[1])) * np.max(data)

    if visualize:
        plt.title('Chromatin accessibility')
        plt.xlabel('Heart [log (x+1))]')
        plt.ylabel('Gastric [log (x+1)]')
        plt.savefig(os.path.join(RES_DIR, "multivariateHeartVsGastric.png"))
        h1 = plt.scatter(clusters[: dims[0]], clusters[: dims[0]], c='b', marker='*')
        plt.ion()
        plt.show()

    # once we have assumption for clusters work with real sub batches of the data
    sub_indics = sub_indics.reshape(4, -1)

    different_clusters = False
    while not different_clusters:
        for step in range(5):
            sub_data = data[sub_indics[step % 4], :]
            clusters = soft_k_means_step(sub_data, clusters)
        if visualize:
            h1.set_offsets(clusters[:, dims])
            plt.draw()
            #time.sleep(0.2)
        if num_states:
            different_clusters = True
        else:
            dist_matrix = np.array([np.sum(np.power(clusters - c, 2), axis=1) for c in clusters])
            np.fill_diagonal(dist_matrix, 1000)
            closest_cluster = np.min(dist_matrix)

            if closest_cluster < 0.1:
                # pruning the closest point and add random to close points
                subject_to_next_prune = list(set(np.where(dist_matrix < 0.1)[0]))
                clusters[subject_to_next_prune, :] += 0.5 * clusters[subject_to_next_prune, :] * np.random.random(
                    (len(subject_to_next_prune), data.shape[1]))
                clusters = clusters[np.arange(n_clusters) != np.where(dist_matrix == closest_cluster)[0][0], :]
                n_clusters -= 1
            else:
                different_clusters = True

    # now assign points to clusters
    clusters = clusters[np.argsort(np.sum(clusters ** 2, 1))]  # to give some meaning
    W = np.array([np.sum(np.power(data - c, 2), axis=1) for c in clusters])
    W = np.minimum(W, 500)  # 500 is enough (to eliminate underflow)
    W = np.exp(-W)
    W = W / np.sum(W, 0)  # normalize for each point
    W = W / np.sum(W, 1)[:, None]  # normalize for all cluster
    means = np.dot(W, data)
    covs = []
    min_std = np.finfo(float).eps
    for mu, p in zip(means, W):
        seq_min_mean = data - mu
        new_cov = np.dot((seq_min_mean.T * p), seq_min_mean)
        new_cov = np.maximum(new_cov, min_std)
        covs.append(new_cov)
    means_covs = list(zip(means, covs))
    """
        n_clusters = len(set(clusters_assignments))
        means_covs = [data[clusters_assignments == i, :] for i in set(clusters_assignments)]
        means_covs = [(np.mean(sub_pop, 0), np.cov(sub_pop.T)) for sub_pop in means_covs]
        # sort them base on the distance from 0 (to keep it nice)
        means_covs.sort(key=lambda x: np.sum(x[0]**2))
        """
    if visualize:
        w_size = 1000.0

        x, y = np.meshgrid(*[np.arange(np.floor(np.min(data[:, d]) - 1.0), np.ceil(np.max(data[:, d])),
                                       (np.ceil(np.max(data[:, d])) - np.floor(np.min(data[:, d])) + 1.0) / w_size) for
                             d in dims])
        pos = np.array([x.ravel(), y.ravel()])
        for mean, cov in means_covs:
            rv = MultivariateNormal(mean[dims], cov[:, dims][dims, :])

            z = rv.pdf(pos)
            z[z < 0.1] = np.nan
            plt.contourf(x, y, z.reshape((w_size, w_size)), alpha=0.8, vmin=0, vmax=1)

        plt.draw()
        plt.ioff()
        plt.savefig(os.path.join(RES_DIR, "multivariateHeartVsGastricStates.png"))
        plt.show()
    return means_covs


def multichannel_hmm_continuous(resolution=1000, model_name=None, output_p=False, out_file=None):
    """
    Use multichannel HMM to classify DNase

    continuous approach (multivariate gaussian mixture model)
    @param resolution: rsolution to learn
    """
    if model_name is None:
        model_name = 'multichannel%i' % resolution
    all_chromotomes = ['chr8']
    train_chrom = 'chr8'
    sparse_data = load_multichannel(resolution, all_chromotomes)
    # log(x+1) transform
    genome_classification = dict()
    chromosomes = list(sparse_data.keys())
    chromosomes.remove(train_chrom)
    chromosomes.insert(0, train_chrom)
    np.seterr(all='raise')
    n_states = 4  # just for debug
    for chrom in chromosomes:
        sparse_data[chrom].data = np.log(sparse_data[chrom].data + 1)
        chrom_data = sparse_data[chrom].todense()
        # create emission matrix by initial guess based on simple clustering
        chrom_data = np.array(chrom_data)  # nd array is cool. matrix not cool
        if train_chrom == chrom:

            emission = continuous_state_selection(chrom_data, num_states=n_states)
            n_states = len(emission) + 1  # number of states plus begin state
            state_transition = np.random.random((n_states, n_states))
            # fill diagonal with higher values
            np.fill_diagonal(state_transition, np.sum(state_transition, 1))
            state_transition[:, 0] = 0  # set transition to begin state to zero
            # normalize
            state_transition /= np.sum(state_transition, 1)[:, np.newaxis]
            # initial guess
            initial_gaussian_model = GaussianHMM(state_transition, emission)
            # train
            new_model, p = bwiter.bw_iter(chrom_data, initial_gaussian_model, 10)

        # decode data
        genome_classification[chrom] = new_model.viterbi(chrom_data)

    # save
    if not os.path.exists(os.path.join(MODELS_DIR, model_name)):
        os.makedirs(os.path.join(MODELS_DIR, model_name))
    SeqLoader.save_result_dict(os.path.join(MODELS_DIR, model_name, "segmentation"), genome_classification)


def soft_k_means_step(data, clusters):
    W = np.array([np.sum(np.power(data - c, 2), axis=1) for c in clusters])
    W = np.minimum(W, 500)  # 500 is enough (to eliminate underflow)
    W = np.exp(-W)
    W = W / np.sum(W, 0)  # normalize for each point
    W = W / np.sum(W, 1)[:, None]  # normalize for all cluster
    new_clusters = np.dot(W, data)
    diff = np.sum((new_clusters - clusters) ** 2, axis=1)
    print(diff)
    return new_clusters


def continuous_state_selection(data, num_states=None, visualize=False):
    """
    Heuristic creation of emission for states  in continuous multidimensional model.
    Instead of random selection of the emission matrix we find clusters of co-occurring values,
    and use those clusters as means for states and the close values as estimation for covariance matrix

    @param data: dense data for specific chromosome
    @return: initial emission for gaussian mixture model HMM (array of (mean, covariance)
    """
    #from scipy.spatial.distance import pdist, squareform
    data = data.T
    if visualize:
        from matplotlib.pylab import plt
        # TODO: we can use PCA but for now just select 2 randoms dimensions
        dims = [2, 1]  #np.random.permutation(np.arange(data.shape[1]))[:2]
        plt.hexbin(data[:, dims[0]], data[:, dims[1]], alpha=0.7, edgecolors='w', cmap=plt.cm.OrRd,
                   extent=[np.floor(np.min(data[:, dims[0]]) - 1.0), np.ceil(np.max(data[:, dims[0]])),
                           np.floor(np.min(data[:, dims[1]]) - 1.0), np.ceil(np.max(data[:, dims[1]]))], bins='log')  #

    sub_indics = np.random.permutation(np.arange(data.shape[0] - data.shape[0] % 4))
    """
    # select sample for spectral clustering. spectral clustering
    #  is used to assign number of states
    sub_data = data[sub_indics[:2000], :]
    scale = 0.5
    affinity_matrix = np.exp(-pdist(sub_data, 'sqeuclidean')/(2*scale**2))
    affinity_matrix = squareform(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    D = np.diag(1.0/np.sum(affinity_matrix, 1))
    L = np.dot(np.dot(D, affinity_matrix), D)
    eigvalues = np.linalg.eigvals(L)
    # sort by eigenvalues
    eigvalues = eigvalues[eigvalues.argsort()[::-1]]
    # select number of components according to eigvalues
    n_clusters = np.diff((eigvalues[eigvalues > 0])/np.sum(eigvalues[eigvalues > 0]))  # 6

    n_clusters = np.where(n_clusters > -1e-3)[0]
    n_clusters = np.ceil((n_clusters[0]*0.8 + n_clusters[1]*0.2))
    """
    n_clusters = num_states or data.shape[1] * 2  # number of clustering will be subject to pruning
    clusters = np.random.random((n_clusters, data.shape[1])) * np.max(data)

    if visualize:
        plt.title('Chromatin accessibility')
        plt.xlabel('Heart [log (x+1))]')
        plt.ylabel('Gastric [log (x+1)]')
        plt.savefig(os.path.join(RES_DIR, "multivariateHeartVsGastric.png"))
        h1 = plt.scatter(clusters[: dims[0]], clusters[: dims[0]], c='b', marker='*')
        plt.ion()
        plt.show()

    # once we have assumption for clusters work with real sub batches of the data
    sub_indics = sub_indics.reshape(4, -1)

    different_clusters = False
    while not different_clusters:
        for step in range(5):
            sub_data = data[sub_indics[step % 4], :]
            clusters = soft_k_means_step(sub_data, clusters)
        if visualize:
            h1.set_offsets(clusters[:, dims])
            plt.draw()
            #time.sleep(0.2)
        if num_states:
            different_clusters = True
        else:
            dist_matrix = np.array([np.sum(np.power(clusters - c, 2), axis=1) for c in clusters])
            np.fill_diagonal(dist_matrix, 1000)
            closest_cluster = np.min(dist_matrix)

            if closest_cluster < 0.1:
                # pruning the closest point and add random to close points
                subject_to_next_prune = list(set(np.where(dist_matrix < 0.1)[0]))
                clusters[subject_to_next_prune, :] += 0.5 * clusters[subject_to_next_prune, :] * np.random.random(
                    (len(subject_to_next_prune), data.shape[1]))
                clusters = clusters[np.arange(n_clusters) != np.where(dist_matrix == closest_cluster)[0][0], :]
                n_clusters -= 1
            else:
                different_clusters = True

    # now assign points to clusters
    clusters = clusters[np.argsort(np.sum(clusters ** 2, 1))]  # to give some meaning
    W = np.array([np.sum(np.power(data - c, 2), axis=1) for c in clusters])
    W = np.minimum(W, 500)  # 500 is enough (to eliminate underflow)
    W = np.exp(-W)
    W = W / np.sum(W, 0)  # normalize for each point
    W = W / np.sum(W, 1)[:, None]  # normalize for all cluster
    means = np.dot(W, data)
    covs = []
    min_std = np.finfo(float).eps
    for mu, p in zip(means, W):
        seq_min_mean = data - mu
        new_cov = np.dot((seq_min_mean.T * p), seq_min_mean)
        new_cov = np.sqrt(np.maximum(new_cov, min_std))
        covs.append(new_cov)
    means_covs = list(zip(means, covs))
    """
    n_clusters = len(set(clusters_assignments))
    means_covs = [data[clusters_assignments == i, :] for i in set(clusters_assignments)]
    means_covs = [(np.mean(sub_pop, 0), np.cov(sub_pop.T)) for sub_pop in means_covs]
    # sort them base on the distance from 0 (to keep it nice)
    means_covs.sort(key=lambda x: np.sum(x[0]**2))
    """
    if visualize:
        w_size = 1000.0

        x, y = np.meshgrid(*[np.arange(np.floor(np.min(data[:, d]) - 1.0), np.ceil(np.max(data[:, d])),
                                       (np.ceil(np.max(data[:, d])) - np.floor(np.min(data[:, d])) + 1.0) / w_size) for
                             d in dims])
        pos = np.array([x.ravel(), y.ravel()])
        for mean, cov in means_covs:
            rv = MultivariateNormal(mean[dims], cov[:, dims][dims, :])

            z = rv.pdf(pos)
            z[z < 0.1] = np.nan
            plt.contourf(x, y, z.reshape((w_size, w_size)), alpha=0.8, vmin=0, vmax=1)

        plt.draw()
        plt.ioff()
        plt.savefig(os.path.join(RES_DIR, "multivariateHeartVsGastricStates.png"))
        plt.show()
    return means_covs


def multichannel_hmm_continuous(resolution=1000, model_name=None, output_p=False, out_file=None):
    """
    Use multichannel HMM to classify DNase

    continuous approach (multivariate gaussian mixture model)
    @param resolution: rsolution to learn
    """
    if model_name is None:
        model_name = 'multichannel%i' % resolution
    all_chromotomes = ['chr8']
    train_chrom = 'chr8'
    sparse_data = load_multichannel(resolution, all_chromotomes)

    # pca to 5 components

    # log(x+1) transform
    genome_classification = dict()
    chromosomes = list(sparse_data.keys())
    chromosomes.remove(train_chrom)
    chromosomes.insert(0, train_chrom)
    np.seterr(all='raise')
    for chrom in chromosomes:
        sparse_data[chrom].data = np.log(sparse_data[chrom].data + 1)
        chrom_data = sparse_data[chrom].todense()
        # create emission matrix by intial guess based on simple clustering
        chrom_data = np.array(chrom_data)  # nd array is cool. matrix not cool
        if train_chrom == chrom:
            emission = continuous_state_selection(chrom_data)
            n_states = len(emission) + 1  # number of states plus begin state
            state_transition = np.random.random((n_states, n_states))
            # fill diagonal with higher values
            np.fill_diagonal(state_transition, np.sum(state_transition, 1))
            state_transition[:, 0] = 0  # set transition to begin state to zero
            # normalize
            state_transition /= np.sum(state_transition, 1)[:, np.newaxis]
            # initial guess
            intial_gaussian_model = GaussianHMM(state_transition, emission)
            # train
            new_model, p = bwiter.bw_iter(chrom_data, intial_gaussian_model, 10)

        # decode data
        genome_classification[chrom] = new_model.viterbi(chrom_data)

    # save
    if not os.path.exists(os.path.join(MODELS_DIR, model_name)):
        os.makedirs(os.path.join(MODELS_DIR, model_name))
    SeqLoader.save_result_dict(os.path.join(MODELS_DIR, model_name, "segmentation"), genome_classification)
    publish_dic(genome_classification,
                resolution,
                '%sSegmentome' % model_name,
                short_label="Mutlicell %s" % out_file,
                long_label="%i states, multi-cell DNase HMM GMM"% n_states-1)


def raw_find_variable_regions():
    """
    Simple function to locate regions that behave differently in cell types
    based only on raw data
    """
    chrom = 'chr6'
    resolution = 10000
    chrom_data = load_multichannel(resolution, [chrom])[chrom]
    chrom_data.data = np.log(chrom_data.data + 1)
    chrom_data = np.array(chrom_data.todense())
    variance = np.var(chrom_data, 0)
    max_var = np.argsort(variance)[::-1]
    for i in max_var[:10]:
        print('%s: %i-%i' % (chrom, i * resolution - 10000, i * resolution + 10000))
    print(max_var[:10] * resolution)
    print('-----')


if __name__ == '__main__':
    commands = {
        'cell_type_mean': cell_type_mean,
        'multichannel_discrete': multichannel_hmm_discrete,
        'multichannel_continuous': multichannel_hmm_continuous
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help="command to execute: %s" % (', '.join(list(commands.keys()))))
    parser.add_argument('--posterior', help="use this option to output posterior probabilities instead of states",
                        action="store_true", default=False)
    parser.add_argument('--model', help="model file to be used")
    parser.add_argument('--resolution', help="resolution to use for classification", type=int, default=1000)

    parser.add_argument('--output', help="Output file prefix", default=None)

    args = parser.parse_args()
    if args.command.startswith('multichannel_'):
        commands[args.command](args.resolution, model_name=args.model, output_p=args.posterior, out_file=args.output)
    else:
        commands[args.command]()