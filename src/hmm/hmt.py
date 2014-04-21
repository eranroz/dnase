"""
Hidden Markov Tree model
"""
from abc import ABCMeta
from collections import namedtuple
import os

import scipy

from config import RES_DIR, CHROM_SIZES
from data_provider import SeqLoader
from hmm.HMMModel import _ContinuousEmission
from hmm.bwiter import bw_iter, IteratorCondition


__author__ = 'eranroz'
import numpy as np


class HMTModel(object):
    """
    base model for HMT
    see Crouse 1997 and Durand 2013
    """
    __metaclass__ = ABCMeta
    MIN_STD = 0.1

    def __init__(self, state_transition, mean_vars, emission_density=scipy.stats.norm):
        """
        Initializes a new HMT model.
        @param state_transition: state transition matrix.
                    with rows - source state, cols - target state.
                    0 state assumed to be the begin state (pi - distrbution for root of the tree)
        @param mean_vars: matrix with rows=num of states and cols =2,
                          where the first column is mean and second is variance
        """
        self.state_transition = state_transition
        self.mean_vars = mean_vars
        self.emission_density = emission_density
        self.emission = _ContinuousEmission(mean_vars, emission_density)
        self.min_alpha = None

    def num_states(self):
        """
        Get number of states in the model
        """
        return self.state_transition.shape[0]

    def level_emission(self, level):
        """
        Emission for level. override it to assign different emissions for different levels
        @param level: level where 0 is the root
        @return: a emission matrix (indexable object) with rows as states and columns as values for emission
        """
        return self.emission

    def maximize(self, sequence_tree, ud_output):
        """
        Maximization step for in Upward-Downward algorithm (EM)

        @param sequence_tree symbol sequence
        @param ud_output results of upward downward (scaling version)
        """

        self._maximize_emission(sequence_tree, ud_output.state_p)
        self.state_transition[0, 1:] = ud_output.state_p[-1]
        self.state_transition[1:, 1:] *= ud_output.transition_stat
        #normalize
        self.state_transition /= np.sum(self.state_transition, 1)[:, None]
        if self.min_alpha is not None:
            n_states = self.state_transition.shape[0]-1  # minus begin/root state
            diagonal_selector = np.eye(n_states, dtype='bool')
            self_transitions = self.state_transition[1:, 1:][diagonal_selector]
            n_self_transitions = np.maximum(self.min_alpha, self_transitions)
            # reduce the diff from the rest of transitions equally
            self.state_transition[1:, 1:][~diagonal_selector] -= (n_self_transitions-self_transitions)/(n_states-1)
            self.state_transition[1:, 1:][diagonal_selector] = n_self_transitions

        print('State transition')
        print(self.state_transition)

    def _maximize_emission(self, sequence_tree, gammas):
        n_states = self.num_states() - 1
        n_levels = len(sequence_tree)
        means_levels = np.zeros((n_levels, n_states))
        vars_levels = np.zeros((n_levels, n_states))
        state_norm_levels = np.zeros((n_levels, n_states))
        scale_level = 0
        for gamma, seq in zip(gammas, sequence_tree):
            state_norm = np.sum(gamma, 0)
            mu = np.sum(gamma * seq[:, None], 0) / state_norm
            sym_min_mu = np.power(seq[:, None] - mu, 2)
            std = np.sum(gamma * sym_min_mu, 0) / state_norm
            state_norm_levels[scale_level, :] = state_norm
            vars_levels[scale_level, :] = np.sqrt(std)
            means_levels[scale_level, :] = mu
            scale_level += 1

        state_norm_levels = state_norm_levels / np.sum(state_norm_levels, 0)
        state_means = np.sum(means_levels * state_norm_levels, 0)
        state_vars = np.maximum(HMTModel.MIN_STD, np.sum(vars_levels * state_norm_levels, 0))

        self.mean_vars = np.column_stack([state_means, state_vars])
        self.emission = _ContinuousEmission(self.mean_vars)
        print(self.emission)

    def viterbi(self, sequence_tree):
        """
        Viterbi algorithm based on Durand 2013 and in log space

        @param sequence_tree: tree-like array, where  sequence[0]=scale 1, sequence[1]=scale 2 etc...
        @return: most probable state for each node
        """
        # upward
        n_states = self.state_transition.shape[0] - 1  # the begin is fake
        transition = np.log(self.state_transition[1:, 1:])
        p_u = []

        wave_lvl_iterator = iter(sequence_tree)
        # initialization
        # leaves
        leaves = next(wave_lvl_iterator)
        scale_level = len(sequence_tree)
        emission = self.level_emission(scale_level)
        curr_b_u_tree = emission[:, leaves]
        curr_b_u_tree = np.log(curr_b_u_tree)
        p_u.append(curr_b_u_tree)

        back_map = []
        for lvl in wave_lvl_iterator:
            scale_level -= 1
            emission = self.level_emission(scale_level)
            prev_up = np.array([np.max(transition[state, :]+p_u[-1], 1) for state in np.arange(0, n_states)]).T
            back_map.append(np.array([np.argmax(transition[state, :]+p_u[-1], 1) for state in np.arange(0, n_states)]).T)
            curr_b_u_tree = (prev_up[::2, :]+prev_up[1::2, :])+emission[:, lvl]
            p_u.append(curr_b_u_tree)

        p = np.max(p_u[-1][0]+np.log(self.state_transition[0, 1:]))
        print('Log likelihood', p)
        probable_tree = [np.argmax(p_u[-1][0] + np.log(self.state_transition[0, 1:]))]
        # maximum "downward"
        for lvl_max in back_map[::-1]:
            likely_parent = probable_tree[-1]
            likely_parent = np.array([likely_parent, likely_parent]).T.reshape(lvl_max.shape[0])
            probable_tree.append(lvl_max[np.arange(0, lvl_max.shape[0]), likely_parent])

        return probable_tree

    def upward_downward(self, sequence_tree, iterations=3):
        """
        upward-downward algorithm/EM
        @param sequence_tree: tree-like array, where  sequence[0]=scale 1, sequence[1]=scale 2 etc...
        @param iterations

        Remarks:
        * implementation based on scaled version, see Durand & Goncalves
        * you may use dwt(prepare_sequence(seq)) to get the wavelet coefficients
        """
        res = None
        for em_iteration in range(1, iterations):
            res = self.forward_backward(sequence_tree)
            self.maximize(sequence_tree, res)
            print(em_iteration, 'P:', res.model_p)

        print(self.state_transition)
        print(self.mean_vars)
        return res.state_p[0]

    def forward_backward(self, sequence_tree):
        """
        Actual implementation for upward downward - calculates the likelihood for each node in the tree
        @param sequence_tree:
        @return:
        """
        n_states = self.state_transition.shape[0] - 1  # the begin is fake
        transition = self.state_transition[1:, 1:]

        # == initial distribution of hidden state (Algorithm 3 in Durand) ==
        initial_dist = np.zeros((len(sequence_tree), n_states), order='F')  # rows - tree levels, cols - state
        init_iterator = np.nditer(initial_dist[::-1], op_flags=['writeonly'], flags=['external_loop'], order='C')
        next(init_iterator)[...] = self.state_transition[0, 1:]

        for _ in sequence_tree[:0:-1]:
            next_lvl = np.dot(init_iterator, transition)
            next(init_iterator)[...] = next_lvl
            # end of algorithm 3

        # conditional upward algorithm (Algorithm 4 in Durand)
        b_u_tree = []
        b_up_tree = []
        init_iterator = np.nditer(initial_dist, op_flags=['readonly'], flags=['external_loop'], order='C')
        wave_lvl_iterator = iter(sequence_tree)
        # initialization
        # leaves
        leaves = next(wave_lvl_iterator)
        emission = self.level_emission(len(sequence_tree))
        #curr_b_u_tree = np.array([emission[:, w] for w in leaves]) * next(init_iterator)
        curr_b_u_tree = emission[:, leaves] * next(init_iterator)
        # normalize
        curr_b_u_tree = curr_b_u_tree / np.sum(curr_b_u_tree, 1)[:, None]
        b_u_tree.append(curr_b_u_tree)

        curr_b_u_tree = np.dot(curr_b_u_tree / init_iterator, transition.T)
        b_up_tree.append(curr_b_u_tree)

        lop_u = [np.zeros(len(sequence_tree[0]))]

        # induction
        m_u = []
        scale_level = len(sequence_tree)
        for lvl, lvl_scale in zip(wave_lvl_iterator, init_iterator):
            scale_level -= 1
            emission = self.level_emission(scale_level)
            prev_up = b_up_tree[-1]
            prev_up = np.product(np.array([prev_up[::2], prev_up[1::2]]), 0)
            lvl_emis = emission[:, lvl]

            p_of_u = lvl_emis * prev_up * lvl_scale
            curr_mu = np.sum(p_of_u, 1)
            m_u.append(curr_mu)
            prev_lu = np.sum(np.array([lop_u[-1][::2], lop_u[-1][1::2]]), 0)
            lop_u.append(np.log(curr_mu) + prev_lu)
            curr_b_u_tree = p_of_u / curr_mu[:, None]
            b_u_tree.append(curr_b_u_tree)
            if len(lvl) > 1:  # except at root node
                curr_b_u_tree = np.dot(curr_b_u_tree / lvl_scale, transition.T)  # or lvl_scale outside
                b_up_tree.append(curr_b_u_tree)

        # end of upward
        # downward (algorithm 5)
        #intiation
        alphas = [np.ones(n_states)]

        prev_alpha = np.array([np.ones(n_states)])
        b_u_tree_iterator = iter(b_u_tree[::-1])
        b_up_tree_iterator = iter(b_up_tree[::-1])

        init_iterator = np.nditer(initial_dist[::-1], op_flags=['readonly'], flags=['external_loop'], order='C')
        next(init_iterator)
        transition_statistic = np.zeros((n_states, n_states))

        cur_stat = np.array([[0, 0]])
        for bt, scale, b_up in zip(b_u_tree_iterator, init_iterator, b_up_tree_iterator):
            transition_statistic += np.dot(bt.T, cur_stat) / scale

            a_p = np.array([prev_alpha, prev_alpha]).T.reshape(n_states, len(prev_alpha) * 2).T
            b_p = np.array([bt, bt]).T.reshape(n_states, len(bt) * 2).T
            cur_stat = a_p * b_p / b_up
            prev_alpha = np.dot(cur_stat, transition) / scale
            alphas.append(prev_alpha)

        # M step
        #collecting statistics (expectations)
        state_p = [] # likelihood for each level in the tree
        for aa, bb, ww in zip(alphas, b_u_tree[::-1], sequence_tree[::-1]):
            gamma = aa * bb
            state_p.insert(0, gamma)

        ud_result = namedtuple('UDResult', 'model_p state_p transition_stat')
        return ud_result(lop_u[-1], state_p, transition_statistic)


def dwt(signal, h=np.array([1.0 / 2, -1.0 / 2]), g=np.array([1.0 / 2, 1.0 / 2])):
    """
    Simple discrete wavelet transform.
    for good reference: http://www.mathworks.com/help/wavelet/ref/dwt.html
    @param signal: signal to create dwt for. the signal must be log2(signal)%1=0
    @param h: high pass filter (for details space)
    @param g: low pass filter (for approximation space)
    @return: zip(scaling arrays, wavelet arrays)
    """
    scaling_coefficients = []
    wavelets_coefficients = []
    approx = signal
    while len(approx) != 1:
        details = np.convolve(approx, h)[h.size - 1::2]
        wavelets_coefficients.append(details)
        approx = np.convolve(approx, g)[g.size - 1::2]
        scaling_coefficients.append(approx)
    return scaling_coefficients, wavelets_coefficients


def idwt(approx, wavelets, h=np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]),
         g=np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])):
    """
    Simple inverse discrete wavelet transform.
    for good reference: http://www.mathworks.com/help/wavelet/ref/dwt.html
    @param approx: approximation of signal at low resolution
    @param h: high pass filter (for details space)
    @param g: low pass filter (for approximation space)
    @return: recovered signal
    """
    wave_level = iter(wavelets[::-1])

    h, g = g[::-1], h[::-1]
    recovered = approx
    for wave in wave_level:
        #upsample
        recovered = np.column_stack([recovered, np.zeros(recovered.size)]).flatten()
        wave_up = np.column_stack([wave, np.zeros(wave.size)]).flatten()
        recovered = np.convolve(recovered, h)[:-(h.size - 1)]
        recovered = recovered + np.convolve(wave_up, g)[:-(g.size - 1)]

    return recovered


def prepare_sequence(sequence):
    """
    Right zero padding for signal before actual processing
    @param sequence: sequence to prepare
    @return: new sequence that can be logged 2
    """
    new_signal_len = np.power(2, np.ceil(np.log2(len(sequence))))
    print(len(sequence), new_signal_len)
    new_signal = np.zeros(new_signal_len)
    new_signal[0:len(sequence)] = sequence
    return new_signal


def testNoise():
    """
    Two options tested:
    * Use the scaling coefficient to train a model, and then use the posterior probabilities of the leaves
      (or higher order?)
    * Use wavelet coefficients, then do inverse DWT and give the details weights according to posterior.
      In case we use db wavelets - have to use abs(signal) to recover

    Both options gives similar results.

    """
    training = SeqLoader.load_dict('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872', 20,
                                   transform=SeqLoader.continuous_transform)
    classified_seq = dict()

    orig_len = len(training['chr1'])  # chromosome 1: 12,462,035
    new_len = 2 ** 16  # np.floor(np.log2(orig_len))  # 2**23
    signal = training['chr1'][0:new_len]

    from scipy.signal import daub, qmf

    scaling_coefficients, wavelets_coefficients = dwt(signal, daub(2), qmf(daub(2)))
    model = HMTModel(np.array([
        [0, 0.8, 0.2],
        [0, 0.6, 0.4],
        [0, 0.2, 0.8],
    ]), np.array([
        (0, 2),  # closed
        (1, 4),  # open - more variance
    ]))
    iterations = 15
    model.upward_downward(wavelets_coefficients, iterations)

    for k, v in training.items():
        if k == 'chr7':
            print('Classifying %s' % k)
            orig_len = len(v)
            new_len = 2 ** np.ceil(np.log2(orig_len))
            signal = np.pad(v, [0, new_len - orig_len], 'constant')

            scaling_coefficients, wavelets_coefficients = dwt(signal, daub(2), qmf(daub(2)))
            wavelets_coefficients = [np.round(w, 4) for w in wavelets_coefficients]  # improve time performance
            ud_result = model.forward_backward(wavelets_coefficients)
            # revert dwt
            fixed_wave = [w * p[:, 1] for w, p in zip(wavelets_coefficients, ud_result.state_p)]
            isignal = idwt(scaling_coefficients[-1], fixed_wave)
            classified_seq[k] = isignal

    BED_GRAPH_RESULTS_DIR = os.path.join(RES_DIR, 'open-closed', 'data')
    res_file = os.path.join(BED_GRAPH_RESULTS_DIR, 'fetalBrain.%i.hmt.wavelet.bg' % iterations)
    SeqLoader.build_bedgraph(classified_seq, resolution=20, output_file=res_file)
    """
    BED_GRAPH_RESULTS_DIR = os.path.join(RES_DIR, 'open-closed', 'data')
    res_file = os.path.join(BED_GRAPH_RESULTS_DIR, 'fetalBrain.%i.hmt.bg' % iterations)
    SeqLoader.build_bedgraph(classified_seq, resolution=40, output_file=res_file)
    # publish
    SeqLoader.bg_to_bigwig(res_file)
    # save as pkl
    #res_file = os.path.join(BED_GRAPH_RESULTS_DIR, 'fetalBrain.%i.hmt.npz' % iterations)
    SeqLoader.save_result_dict(res_file, classified_seq)  # TODO: 31/1 - changed from training toclassified_seq  - needs rerun
    """
    #isignal = idwt(scaling_coef[-1], wavelets_coeff)
    #print(signal)
    #print(isignal)


def fixSize():
    classified = SeqLoader.load_result_dict(
        os.path.join(os.path.join(RES_DIR, 'open-closed', 'data'), 'fetalBrain.%i.hmt.npz' % 15))
    chromSizes = dict()
    with open(CHROM_SIZES, 'r') as chromSize:
        for r in chromSize.readlines():
            chromSizes[r.split('\t')[0]] = int(r.split('\t')[1])
            if r.split('\t')[0] in classified:
                classified[r.split('\t')[0]] = classified[r.split('\t')[0]][0:(int(r.split('\t')[1]) - 1) / 40]
    BED_GRAPH_RESULTS_DIR = os.path.join(RES_DIR, 'open-closed', 'data')
    res_file = os.path.join(BED_GRAPH_RESULTS_DIR, 'fetalBrain.%i.hmt.bg' % 15)
    SeqLoader.build_bedgraph(classified, resolution=40, output_file=res_file)
    SeqLoader.bg_to_bigwig(res_file)

def testViterbi():
    training = SeqLoader.load_dict('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872', 20,
                                   transform=SeqLoader.continuous_transform, chromosome=['chr1'])
    classified_seq = dict()

    orig_len = len(training['chr1'])  # chromosome 1: 12,462,035
    new_len = 2 ** 16  # np.floor(np.log2(orig_len))  # 2**23
    signal = training['chr1'][0:new_len]

    from scipy.signal import daub, qmf
    scaling_coefficients, wavelets_coefficients = dwt(signal, daub(2), qmf(daub(2)))
    model = HMTModel(np.array([
        [0, 0.8, 0.2],
        [0, 0.6, 0.4],
        [0, 0.2, 0.8],
    ]), np.array([
        (0, 2),  # closed
        (1, 4),  # open - more variance
    ]))

    new_model, p = bw_iter(wavelets_coefficients, model, IteratorCondition(10))
    classification = new_model.viterbi(wavelets_coefficients)
    print(classification[-1].shape)


if __name__ == "__main__":
    #import cProfile
    #cProfile.run('testNoise()')
    #testNoise()
    # fix reconstruction with abs?
    testViterbi()
