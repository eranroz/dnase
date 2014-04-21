__author__ = 'eranroz'
import scipy.stats
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from pyx import _hmmc


class HMMModel(object):
    """
    base model for HMM
    """
    __metaclass__ = ABCMeta

    def __init__(self, state_transition, emission, min_alpha=None):
        """
        Initializes a new HMM model.
        @param state_transition: state transition matrix.
                    (A & Pi in Rabiner's paper)
                    with rows - source state, cols - target state.
                    0 state assumed to be the begin state
                    (according to Durbin's book)
        @param emission: observation symbol probability distribution
                        (B in Rabiner's paper)
                        rows - states, cols - output
                        The begin state should have emission too (doesn't matter what)

        @param min_alpha: prior on the sequence length (transition to itself)
        """

        self.state_transition = state_transition
        self.emission = emission
        self.min_alpha = min_alpha

    def num_states(self):
        """
        Get number of states in the model
        """
        return self.state_transition.shape[0]

    def num_alphabet(self):
        """
        Get number of symbols in the alphabet
        """
        return self.emission.shape[1]

    def get_state_transition(self):
        """
        State transition matrix: rows - source state, cols - target state
        """
        return self.state_transition

    def get_emission(self):
        """
        Emission matrix:  rows states, cols - symbols
        """
        return self.emission

    @abstractmethod
    def _maximize_emission(self, seq, gammas):
        """
        part of the maximization step of the EM algorithm (Baum-Welsh)
        should update the emission probabilities according to the forward-backward results

        @param seq symbol sequence
        @param gammas from backward forward
        """
        pass

    def _maximize_transition(self, seq, bf_output):
        """
        part of the maximization step of the EM algorithm (Baum-Welsh)
        should state transition probabilities according to the forward-backward results

        @param seq: observation sequence
        @param bf_output: output of the forward-backward algorithm
        @return:
        """
        new_state_transition = self.state_transition.copy()
        emission = self.get_emission()
        unique_values = set(seq)
        back_emission_seq = np.zeros((len(seq), self.num_states() - 1))
        for v in unique_values:
            back_emission_seq[seq == v, :] = emission[1:, v]

        back_emission_seq *= bf_output.backward / bf_output.scales[:, None]

        new_state_transition[1:, 1:] *= np.dot(bf_output.forward[:-1, :].transpose(), back_emission_seq[1:, :])
        new_state_transition[1:, 1:] /= np.sum(new_state_transition[1:, 1:], 1)[:, None]  # normalize
        if self.min_alpha is not None:
            n_states = new_state_transition.shape[0] - 1  # minus begin state
            diagonal_selector = np.eye(n_states, dtype='bool')
            self_transitions = new_state_transition[1:, 1:][diagonal_selector]
            n_self_transitions = np.maximum(self.min_alpha, self_transitions)
            # reduce the diff from the rest of transitions equally
            new_state_transition[1:, 1:][~diagonal_selector] -= (n_self_transitions - self_transitions) / (n_states - 1)
            new_state_transition[1:, 1:][diagonal_selector] = n_self_transitions

        # start transition
        new_state_transition[0, 1:] = bf_output.state_p[0, :]
        new_state_transition[0, 1:] /= np.sum(new_state_transition[0, 1:])

        # end transition
        new_state_transition[1:, 0] = bf_output.forward[-1, :] * back_emission_seq[-1, :] / bf_output.scales[-1]
        new_state_transition[1:, 0] /= np.sum(new_state_transition[1:, 0])
        #update transition matrix
        self.state_transition = new_state_transition

    def maximize(self, seq, bw_output):
        """
        Maximization step for in Baum-Welsh algorithm (EM)

        @param seq symbol sequence
        @param bw_output results of backward forward (scaling version)
        """
        self._maximize_transition(seq, bw_output)
        self._maximize_emission(seq, bw_output.state_p)

    def __str__(self):
        return '\n'.join(
            ['Model parameters:', 'Emission:', str(self.emission), 'State transition:',
             str(self.state_transition)])

    def viterbi(self, symbol_seq):
        """
        Find the most probable path through the model

        Dynamic programming algorithm for decoding the states.
        Implementation according to Durbin, Biological sequence analysis [p. 57]

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        """

        n_states = self.num_states()
        unique_values = set(symbol_seq)
        emission_seq = np.zeros((len(symbol_seq), n_states - 1))
        for v in unique_values:
            emission_seq[symbol_seq == v, :] = np.log(self.get_emission()[1:, v])

        return _hmmc.viterbi(emission_seq, self.state_transition)

    def viterbi_old(self, symbol_seq):
        """
        Find the most probable path through the model

        Same as above, but not optimized. just for simplification or if you got to trouble with compilation

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        """
        n_states = self.num_states()
        unique_values = set(symbol_seq)
        emission_seq = np.zeros((len(symbol_seq), n_states - 1))
        for v in unique_values:
            emission_seq[symbol_seq == v, :] = np.log(self.get_emission()[1:, v])

        ptr_mat = np.zeros((len(symbol_seq), n_states - 1))
        l_state_trans_mat_T = self.get_state_transition().T

        emission_iterator = iter(emission_seq)
        ptr_iterator = iter(ptr_mat)
        #intial condition is begin state
        prev = next(emission_iterator) + 1 + np.log(l_state_trans_mat_T[1:, 0])
        next(ptr_iterator)[...] = np.argmax(prev)
        end_state = 0  # termination step
        end_transition = np.log(l_state_trans_mat_T[end_state, 1:])
        l_state_trans_mat_T = np.log(l_state_trans_mat_T[1:, 1:])
        #recursion step
        for emission_symbol in emission_iterator:
            p_state_transition = prev + l_state_trans_mat_T
            max_k = np.max(p_state_transition, 1)
            next(ptr_iterator)[...] = np.argmax(p_state_transition, 1)
            prev = emission_symbol + max_k

        p_state_transition = prev + end_transition
        #last_mat = np.max(p_state_transition)
        #traceback step and without begin state
        most_probable_path = np.zeros(len(symbol_seq), int)
        most_probable_path[-1] = np.argmax(p_state_transition)

        for i in np.arange(len(symbol_seq) - 1, 0, -1):
            most_probable_path[i - 1] = ptr_mat[i, most_probable_path[i]]

        return most_probable_path

    def forward_backward(self, symbol_seq, model_end_state=False):
        """
        Calculates the probability for the model and each step in it

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        @param model_end_state: whether to consider end state or not

        Remarks:
        this implementation uses scaling variant to overcome floating points errors.
        """
        n_states = self.num_states()
        emission = self.get_emission()
        state_trans_mat = self.get_state_transition()

        unique_values = set(symbol_seq)
        emission_seq = np.zeros((len(symbol_seq), n_states - 1))
        for v in unique_values:
            emission_seq[symbol_seq == v, :] = emission[1:, v]

        dot = np.dot  # shortcut for performance
        real_transitions_T = state_trans_mat[1:, 1:].T.copy(order='C')
        real_transitions_T2 = state_trans_mat[1:, 1:].copy(order='C')
        summing_arr = np.ones(n_states - 1)

        # emission * transition
        e_trans = iter(emission_seq[1:, :, None] * real_transitions_T)
        forward_iterator = np.nditer([emission_seq, None, None],
                                     flags=['external_loop', 'reduce_ok'],
                                     op_flags=[
                                         ['readonly'],
                                         ['readwrite', 'allocate', 'no_broadcast'],
                                         ['readwrite', 'allocate', 'no_broadcast']
                                     ],
                                     op_axes=[[-1, 0, 1], [-1, 0, 1], [-1, 0, -1]])

        #-----	  forward algorithm	  -----
        #intial condition is begin state (in Durbin there is another forward - the begin = 1)
        tup = next(forward_iterator)  #emission_i, forward_i,scaling_i
        tup[1][...] = state_trans_mat[0, 1:] * tup[0]
        tup[2][...] = np.sum(tup[1])
        tup[1][...] /= tup[2]
        prev_forward = tup[1]

        #recursion step
        for tup in forward_iterator:  #emission_i, forward_i,scaling_i
            prev_forward = dot(next(e_trans), prev_forward)  #== tup[0]*dot(real_transitions_T, prev_forward)

            # scaling - see Rabiner p. 16, or Durbin p. 79
            scaling = tup[2]
            scaling[...] = dot(summing_arr, prev_forward)  # dot is actually faster then np.sum(prev_forward)
            tup[1][...] = prev_forward = prev_forward / scaling

        forward = forward_iterator.operands[1]
        s_j = forward_iterator.operands[2]
        #end transition
        log_p_model = np.sum(np.log(s_j))
        if model_end_state:  # Durbin - with end state
            end_state = 0  # termination step
            end_transition = forward[emission_seq.shape[0] - 1, :] * state_trans_mat[1:, end_state]
            log_p_model += np.log(sum(end_transition))

        #-----	backward algorithm	-----
        #intial condition is end state
        if model_end_state:
            prev_back = (state_trans_mat[1:, 0])  # Durbin p.60
        else:
            prev_back = np.ones(n_states - 1)  # Rabiner p.7 (24)

        backward_iterator = np.nditer([emission_seq[:0:-1], None],  # / s_j[:, None]
                                      flags=['external_loop', 'reduce_ok'],
                                      op_flags=[['readonly'],
                                                ['readwrite', 'allocate']],
                                      op_axes=[[-1, 0, 1], [-1, 0, 1]])

        #recursion step
        e_trans = iter((emission_seq / s_j[:, None])[:0:-1, None, :] * real_transitions_T2)
        for tup in backward_iterator:  # emission_i/scale_i, backward_i
            prev_back = dot(next(e_trans), prev_back, tup[1])  # = dot(real_transitions_T2, prev_back * tup[0])

        if model_end_state:
            backward = np.append(backward_iterator.operands[1][::-1], state_trans_mat[1:, 0][None, :],
                                 axis=0)  # Durbin p.60
        else:
            backward = np.append(backward_iterator.operands[1][::-1], np.ones((1, n_states - 1)),
                                 axis=0)  # Rabiner p.7 (24)

        #return bf_result
        bf_result = namedtuple('BFResult', 'model_p state_p forward backward scales')
        return bf_result(log_p_model, backward * forward, forward, backward, s_j)

    def forward_backward_old(self, symbol_seq, model_end_state=False):
        """
        Calculates the probability for the model and each step in it

        Same as above but with less optimizations. for easier reading
        """
        n_states = self.num_states()
        emission = self.get_emission()
        state_trans_mat = self.get_state_transition()

        s_j = np.ones(len(symbol_seq))
        forward = np.zeros((len(symbol_seq), n_states - 1), order='F')  # minus the begin state
        backward = np.zeros((len(symbol_seq), n_states - 1), order='F')

        #-----    forward algorithm   -----
        #intial condition is begin state (in Durbin there is another forward - the begin = 1)
        forward[0, :] = state_trans_mat[0, 1:] * emission[1:, symbol_seq[0]]
        s_j[0] = sum(forward[0, :])
        forward[0, :] /= s_j[0]
        prev_forward = forward[0, :]
        #recursion step
        #transform to emission array
        unique_values = set(symbol_seq)
        emission_seq = np.zeros((len(symbol_seq), n_states - 1))
        for v in unique_values:
            emission_seq[symbol_seq == v, :] = emission[1:, v]

        real_transitions = state_trans_mat[1:, 1:]
        t_real_transitions = real_transitions.transpose()
        p_state_transition = np.zeros(n_states - 1)
        summing_arr = np.array([1] * (n_states - 1))
        emission_iterator = iter(emission_seq)
        next(emission_iterator)  # skip the first (instead of condition in for loop
        s_j_iterator = np.nditer(s_j, op_flags=['writeonly'])
        forward_iterator = np.nditer(forward, flags=['external_loop'], op_flags=['writeonly'], order='C')
        next(s_j_iterator)
        next(forward_iterator)
        for sym_emission in emission_iterator:
            #p_state_transition = np.sum(real_transitions * prev_forward[:, None], 0)
            np.dot(t_real_transitions, prev_forward, p_state_transition)
            prev_forward = sym_emission * p_state_transition

            # scaling - see Rabiner p. 16, or Durbin p. 79
            scaling = np.dot(summing_arr, prev_forward)  # dot is actually faster then np.sum(prev_forward)
            next(s_j_iterator)[...] = scaling
            #s_j[i] = scaling
            #forward[i, :] = prev_forward = prev_forward / scaling
            next(forward_iterator)[...] = prev_forward = prev_forward / scaling

        #end transition
        log_p_model = np.sum(np.log(s_j))
        if model_end_state:  # Durbin - with end state
            end_state = 0  # termination step
            end_transition = forward[len(symbol_seq) - 1, :] * state_trans_mat[1:, end_state]
            log_p_model += np.log(sum(end_transition))

        #-----  backward algorithm  -----
        #intial condition is end state
        if model_end_state:
            prev_back = backward[len(symbol_seq) - 1, :] = (state_trans_mat[1:, 0])  # Durbin p.60
        else:
            prev_back = backward[len(symbol_seq) - 1, :] = [1, 1]  # Rabiner p.7 (24)

        backward_iterator = np.nditer(backward[::-1], flags=['external_loop'], op_flags=['writeonly'], order='C')
        next(backward_iterator)
        s_j_iterator = iter(s_j[::-1])

        #recursion step
        for sym_emission in emission_seq[:0:-1]:
            np.dot(prev_back * sym_emission, t_real_transitions, p_state_transition)
            #backward[i - 1, :] = prev_back = p_state_transition / s_j[i]  # same scaling as in the forward
            next(backward_iterator)[...] = prev_back = p_state_transition / next(s_j_iterator)

        bf_result = namedtuple('BFResult', 'model_p state_p forward backward scales')
        return bf_result(log_p_model, backward * forward, forward, backward, s_j)

    def forward_backward_log(self, symbol_seq, model_end_state=False):
        """
        Calculates the probability for the model and each step in it

        @param symbol_seq: observed sequence (array). Should be numerical (same size as defined in model)
        @param model: an HMM model to calculate on the given symbol sequence
        @param model_end_state: whether to consider end state or not

        Remarks:
        this implementation uses log variant to overcome floating points errors instead of scaling.
        """
        interpolation_res = 0.01
        interpolation_res_i = 1.0 / interpolation_res
        interpol_tbl = np.log(1 + np.exp(-np.arange(0, 35, interpolation_res)))
        last_interpol = len(interpol_tbl) - 1

        def interpolate(prev):
            """
            Uses interpolation table to calculate log r=log(p+log(1+exp(q-p))) which is approx equals to
            log r = log (max)+(exp(1+log(x)) [x=min-max from prev].
            see Durbin p. 79
            @param prev: previous probabilities plus the transition
            @return: result of interpolation of log r=log(p+log(1+exp(q-p)))
            """
            maxes = np.max(prev, 1)
            interpolation_i = np.minimum(np.round(-interpolation_res_i * (np.sum(prev, 1) - 2 * maxes)),
                                         last_interpol).astype(int)
            return maxes + interpol_tbl[interpolation_i]

        l_emission = np.log(self.get_emission()[1:, :])
        forward = np.zeros((len(symbol_seq), self.num_states() - 1))
        backward = np.zeros((len(symbol_seq), self.num_states() - 1))

        l_state_transition = np.log(self.get_state_transition()[1:, 1:])
        l_t_state_transition = l_state_transition.transpose()
        #-----    forward algorithm   -----
        #intial condition is begin state (in Durbin there is another forward - the begin = 1)
        prev_forward = forward[0, :] = np.log(self.get_state_transition()[0, 1:]) + l_emission[:, symbol_seq[0]]
        #recursion step
        emission_seq = list(enumerate([l_emission[:, s] for s in symbol_seq]))

        for i, sym_emission in emission_seq:
            if i == 0:
                continue

            from_prev = prev_forward + l_t_state_transition  # each row is different state, each col - CHECKED
            # now the sum approximation
            from_prev = interpolate(from_prev)
            forward[i, :] = prev_forward = sym_emission + from_prev

        # termination step
        if model_end_state:
            end_state = 0
            log_p_model = forward[len(symbol_seq) - 1, :] + np.log(self.get_state_transition()[1:, end_state])
            log_p_model = interpolate(np.array([log_p_model]))
        else:
            log_p_model = interpolate([forward[len(symbol_seq) - 1, :]])

        #-----  backward algorithm  -----
        last_index = len(symbol_seq) - 1
        if model_end_state:
            prev_back = backward[last_index, :] = np.log(self.get_state_transition()[1:, 0])
        else:
            prev_back = backward[last_index, :] = [0, 0]  # Rabiner p.7 (24)

        for i, sym_emission in reversed(emission_seq):
            if i == 0:
                continue
            p_state_transition = interpolate(l_state_transition + (prev_back + sym_emission))
            prev_back = backward[i - 1, :] = p_state_transition

        # posterior probability 3.14 (P.60 Durbin)
        l_posterior = forward + backward - log_p_model
        bf_result = namedtuple('BFResult', 'model_p state_p forward backward')
        return bf_result(log_p_model, l_posterior, forward, backward)


class DiscreteHMM(HMMModel):
    """
    Discrete Hidden Markov Model

     Handles sequences with discrete alphabet
    """

    def _maximize_emission(self, seq, gammas):
        new_emission_matrix = np.zeros((self.num_states(), self.num_alphabet()))

        state_p = gammas
        for sym in range(0, self.num_alphabet()):
            where_sym = (seq == sym)
            new_emission_matrix[1:, sym] = np.sum(state_p[where_sym, :], 0)

        # normalize
        new_emission_matrix[1:, :] /= np.sum(new_emission_matrix[1:, :], 1)[:, None]

        self.emission = new_emission_matrix


class ContinuousHMM(HMMModel):
    """
    Continuous HMM for observations of real values

    The states are gaussian (or gaussian mixtures)
    @param state_transition: state transition matrix
    @param mean_vars: array of mean, var tuple (or array of such for mixtures)
    @param emission_density: log-concave or elliptically symmetric density
    @param mixture_coef: mixture coefficients
    """

    def __init__(self, state_transition, mean_vars, emission_density=scipy.stats.norm, mixture_coef=None,
                 min_alpha=None):
        emission = _ContinuousEmission(mean_vars, emission_density, mixture_coef)
        super().__init__(state_transition, emission, min_alpha=min_alpha)

    def _maximize_emission(self, seq, gammas):
        mean_vars = np.zeros((self.num_states(), 2))
        min_std = 0.5
        if self.emission.mixtures is None:
            state_norm = np.sum(gammas, 0)
            mu = np.sum(gammas * seq[:, None], 0) / state_norm
            sym_min_mu = np.power(seq[:, None] - mu, 2)
            std = np.sqrt(np.sum(gammas * sym_min_mu, 0) / state_norm)

            std = np.maximum(std, min_std)  # it must not get to zero
            mean_vars[1:, :] = np.column_stack([mu, std])
            self.emission = _ContinuousEmission(mean_vars, self.emission.dist_func)
        else:  # TODO: not yet fully tested
            mean_vars = [(0, 0)]
            mixture_coeff = [1]
            for state in np.arange(0, self.num_states() - 1):
                has_coeff = True
                try:
                    if len(self.emission.mixtures[state + 1]) > 1:
                        coeff_pdfs = [self.emission.dist_func(mean, var).pdf for mean, var in
                                      self.emission.mean_vars[state + 1]]
                        coeff_obs = np.array([[p(s) for p in coeff_pdfs] for s in seq])
                        coeff_obs /= np.sum(coeff_obs, 1)[:, None]
                        gamma_coeff = coeff_obs * gammas[:, state][:, None]
                        seq_man = seq[:, None]
                    else:
                        gamma_coeff = gammas[:, state]
                except TypeError:
                    gamma_coeff = gammas[:, state]
                    seq_man = seq
                    has_coeff = False

                sum_gamma = np.sum(gamma_coeff, 0)

                mu = np.sum(gamma_coeff * seq_man, 0) / sum_gamma
                mu *= self.emission.mixtures[state + 1]
                sym_min_mu = np.power(seq_man - mu, 2)
                std = np.sqrt(np.sum(gamma_coeff * sym_min_mu, 0) / sum_gamma)
                std = np.maximum(std, min_std)  # it must not get to zero
                if has_coeff:
                    mean_vars.append(list(zip(mu, std)))
                else:
                    mean_vars.append((mu, std))
                mixture_coeff.append(sum_gamma / np.sum(sum_gamma))

            self.emission = _ContinuousEmission(mean_vars, self.emission.dist_func, mixture_coeff)


class _ContinuousEmission():
    """
    Emission for continuous HMM.
    """

    def __init__(self, mean_vars, dist=scipy.stats.norm, mixture_coef=None):
        """
        Initializes a new continuous distribution states.
        @param mean_vars: np array of mean and variance for each state
        @param dist: distribution function
        @return: a new instance of ContinuousDistStates
        """
        self.dist_func = dist
        self.mean_vars = mean_vars
        self.mixtures = mixture_coef
        self.cache = dict()
        self.min_p = 1e-100
        self.states = self._set_states()
        self._set_states()

    def _set_states(self):
        from functools import partial

        if self.mixtures is None:
            states = ([self.dist_func(mean, var).pdf for mean, var in self.mean_vars])
        else:
            states = []
            for mean_var, mixture in zip(self.mean_vars, self.mixtures):
                try:
                    mix_pdf = [self.dist_func(mean, var).pdf for mean, var in mean_var]
                    #mix = lambda x: _ContinuousEmission.mixture_pdf(mix_pdf, mixture, x)
                    mix = partial(_ContinuousEmission.mixture_pdf, mix_pdf, mixture)

                    if np.abs(np.sum(mixture) - 1) > 1e-6:
                        raise Exception("Bad mixture - mixture for be summed to 1")
                except TypeError:
                    mix = self.dist_func(mean_var[0], mean_var[1]).pdf
                states.append(mix)
        return states

    @staticmethod
    def mixture_pdf(pdfs, mixtures, val):
        """
        Mixture distrbution
        @param pdfs:
        @param mixtures:
        @param val:
        @return:
        """
        return np.dot([p(val) for p in pdfs], mixtures)

    def __getitem__(self, x):
        """
        Get emission for state
        @param x:  first index is state (or slice for all states), second is value or array of values
        @return: p according to pdf
        """
        if isinstance(x[0], slice):
            if isinstance(x[1], np.ndarray):  # this new case improves performance if you give emission array of values
                pdfs = np.array([dist(x[1]) for dist in self.states[x[0]]]).T
                pdfs = np.maximum(pdfs, self.min_p)
                return pdfs
            else:
                try:
                    return self.cache[x[1]]
                except KeyError:
                    pdfs = np.array([dist(x[1]) for dist in self.states[x[0]]])
                    pdfs = np.maximum(pdfs, self.min_p)
                    self.cache[x[1]] = pdfs
                    return self.cache[x[1]]
        else:
            return self.states[x[0]].pdf(x[1])

    def __getstate__(self):
        return {
            'mean_vars': self.mean_vars,
            'mixture_coef': self.mixtures
        }

    def __setstate__(self, state):
        self.mean_vars = state['mean_vars']
        self.mixtures = state['mixture_coef']
        self.dist_func = scipy.stats.norm  # TODO: save name for the method to support other dist funcs
        self.min_p = 1e-5
        self.cache = dict()
        self.states = self._set_states()

    def __str__(self):
        return '\n'.join([str(self.dist_func.name) + ' distribution', 'Mean\t Var', str(self.mean_vars)])