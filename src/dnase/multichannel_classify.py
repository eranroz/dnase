"""
Handles multidimensional huge data.

since it requires huge size memory:
 - we use the mean from different cell types instead of just using samples.
 - we use PCA to reduce the number of cell types

There are two approaches:
1. Discrete - discretization for words for each sequence, and then building words by combining them
2. Continuous -  Use the real values of the channel with multi dimensional gaussian and covariance to evaluate


Assumptions:
    Just as an estimation for the size: 242 cells x 2492506 chromosome 1 size (bins of size 100)
    requires 4.5Gb


TODO:
- Make HMM methods work on multidimensional data
    - for discrete (should it just work?)
    - for continous - multivariate gaussian mixture requires fixing mainly the maximization step and
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
from config import OTHER_DATA, NCBI_DIR, BED_GRAPH_RESULTS_DIR, DATA_DIR
from data_provider import SeqLoader
from dnase.dnase_classifier import DNaseClassifier
from dnase.HMMClassifier import HMMClassifier
from hmm.HMMModel import DiscreteHMM

__author__ = 'eranroz'



def cell_type_mean_specific(cell_type):
    """
    Creates mean for specific cell type
    @param cell_type: cell type
    @return:
    """
    out_file = os.path.join(MEAN_DNASE_DIR, '%s.mean.npz' % cell_type)
    print(cell_type)
    #if os.path.exists(out_file):
    #    print('skipping cell type - already have mean')

    cell_data = []
    cell_type_dir = os.path.join(NCBI_DIR, cell_type)
    for sample in os.listdir(cell_type_dir):
        if '.wig' in sample:
            continue
        data = SeqLoader.load_dict(sample.replace('.20.npz', ''), 20, directory=cell_type_dir)
        cell_data += [data]
    if len(cell_data) == 0:
        return
    cell_represent = dict()
    for chrom in cell_data[0].keys():
        max_length = max([len(d[chrom]) for d in cell_data])
        combined = np.zeros((len(cell_data), max_length))
        for i, d in enumerate(cell_data):
            combined[i, 0:len(d[chrom])] = d[chrom]
        cell_represent[chrom] = combined.mean(0)
    SeqLoader.save_result_dict(out_file, cell_represent)


def read_chromosome_sparse(resolution=100, chromosome='chr1'):
    """
    Reads specific chromosome from ALL cell types
    """
    from scipy.sparse import lil_matrix
    import scipy.io
    import zlib

    orig_resolution = 20
    cell_files = os.listdir(DATA_DIR)
    # we sort the files to have same cells in near rows and to have some knowledge about the order
    cell_files = sorted(cell_files)
    n_cells = len(cell_files)
    # for chromosome 1 it takes about 4.5 Gb
    n_max_length = SeqLoader.chrom_sizes()[chromosome]*(20/resolution)

    chromosome_data = lil_matrix((n_cells, n_max_length))
    # TODO: works with pkl should use npz instead
    for i, f in enumerate(cell_files):
        print(f)
        file_name = os.path.join(DATA_DIR, f)

        with open(file_name, 'rb') as file:
            decompress = zlib.decompress(file.read())
            sequence_dict = pickle.loads(decompress)
            new_seq = SeqLoader.down_sample(sequence_dict[chromosome], int(resolution/orig_resolution))
            print(new_seq.shape)
            chromosome_data[i, 0:new_seq.shape[0]] = new_seq

    print('saving')
    scipy.io.mmwrite(os.path.join(OTHER_DATA, 'dnase/multicell/%s.%i.mtx' % (chromosome, resolution)), chromosome_data)
    print('end saving')

def cell_type_mean():
    """
    Create mean dnase for each cell type
    """
    global _CELL_TYPES_DIR
    from multiprocessing.pool import Pool
    pool_process = Pool()
    pool_process.map(cell_type_mean_specific, os.listdir(NCBI_DIR))



def load_multichannel(resolution=100):
    """
    Loads multichannel data.
    @param resolution: resolution of the required data
    @return: dict where keys are chromosomes and values are matrix: cell_types X genome position
    """
    global _CELL_TYPES_DIR
    all_cells = []
    #mean_dnase = [os.path.join(_CELL_TYPES_DIR, cell_type) for cell_type in os.listdir(_CELL_TYPES_DIR) if cell_type.endswith('.npz')]
    workaround_dnase = [next(os.path.join(NCBI_DIR, cell_type, f) for f in os.listdir(os.path.join(NCBI_DIR, cell_type)) if f.endswith('.npz')) for cell_type in os.listdir(NCBI_DIR)]
    for cell_type_path in workaround_dnase:
        print('Loading %s' % cell_type_path)
        cell_data = SeqLoader.load_result_dict(cell_type_path)
        cell_data_new = dict()
        for k, v in cell_data.items():
            cell_data_new[k] = SeqLoader.down_sample(v, resolution / 20)
        all_cells.append(cell_data_new)
        if len(all_cells) == 4:
            break  # for debuging purpose - aavarat hoter
    # organize be chromosomes
    chromosomes = all_cells[0].keys()
    n_samples = len(all_cells)
    chrom_dic = dict()
    for chrom in chromosomes:
        max_length = max([len(cell[chrom]) for cell in all_cells])
        chrom_mat = np.zeros((n_samples, max_length))
        for cell_i, cell in enumerate(all_cells):
            chrom_mat[cell_i, 0:len(cell[chrom])] = cell[chrom]
        chrom_dic[chrom] = chrom_mat
    return chrom_dic


def pca_chrom_matrix(data, dim=None):
    new_data = dict()
    for k, chrom_matrix in data.items():
        co_var = np.cov(chrom_matrix - np.mean(chrom_matrix, 1)[:, None])
        eig_vals, eig_vecs = np.linalg.eig(co_var)
        eig_order = eig_vals.argsort()[::-1]  # just to be sure we have eigvalues ordered
        eig_vals = eig_vals[eig_order]/np.sum(eig_vals)
        eig_vecs = eig_vecs[eig_order]
        if dim is None:
            explains = [np.sum(eig_vals[0:d + 1]) for d in np.arange(0, len(eig_vals))]
            dim = [d for d, ex in enumerate(explains) if ex > 0.8][0]

        pca_matrix = np.dot(eig_vecs[0:dim, :], cell_type_mean())
        new_data[k] = pca_matrix
    return pca_matrix


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

    multichannel_data = pca_chrom_matrix(multichannel_data) #, cells_dimensions_

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
        SeqLoader.publish_dic(multichannel_data, resolution, '%s.%i.rawDiscrete' % (default_name, resolution))


if __name__ == '__main__':
    commands = {
        'cell_type_mean': cell_type_mean,
        'multichannel_hmm': multichannel_hmm_discrete
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