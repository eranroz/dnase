"""
Class for discretization.
"""
import numpy as np

__author__ = 'eranroz'


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
        # print('finding percentiles')
        if self.percentiles_values is None:
            self.percentiles_values = np.percentile(lg_data_n_zero, q=self.percentiles)

        mapped2 = np.zeros(len(lg_data_n_zero), dtype=int)  # * len(self.percentiles_values)
        for i, bounds in enumerate(self.percentiles_values):
            mapped2[lg_data_n_zero > bounds] = i + 1
        return mapped2


class MultiDimDiscreteTransformer(object):
    """
    A class to transform multi-dimensional (multiple feature) from continuous to a discrete alphabets

    """

    def __init__(self, percentiles=[0, 60, 75, 90], flat=True):
        """
        @param flat: whether to flat the output to 1 dimension [alphabet size+sum(alphabet size for each dimension)
        @param percentiles: 1 dimensional for same alphabet size in all features, 2 dimensional to specify different
                            alphabets thresholds for different dimensions
        """
        self.percentiles = percentiles
        self.percentiles_values = None
        self.flatten = flat

    def __call__(self, *args, **kwargs):
        """
        Transforms the sequence to discrete alphabet
        """
        lg_data_n_zero = np.log(np.asfarray(args[0]) + 1)  # we don't want to get 0 for log
        if self.percentiles_values is None:
            self.fit(lg_data_n_zero)

        percentiles_values_max = np.column_stack([self.percentiles_values, np.max(lg_data_n_zero, 0)])
        mapped2 = np.argmin(lg_data_n_zero[:, :, np.newaxis] > percentiles_values_max, 2)

        if self.flatten:
            mapped2 = self.flat(mapped2)

        return mapped2

    def fit(self, lg_data_n_zero, alphabet_size=4):
        """
        Fits the transformer based on real data. It fix the value for percentiles.
        @param alphabet_size: the size of the alphabet
        @param lg_data_n_zero: Log(x)+1 of the data
        """
        if self.percentiles is None:
            # auto select the percentiles based on the data: 4 maximum diff in q
            hist_edges = [np.histogram(lg_data_n_zero[:, i], bins=20) for i in range(lg_data_n_zero.shape[1])]
            edges = np.array([edge for hist, edge in hist_edges])
            cdf_hist = [np.cumsum(hist) for hist, edge in hist_edges]
            selected_arg_hist = np.argsort(np.diff(cdf_hist, 1), 1)[:, -(alphabet_size - 1):] + 1

            # feature_i = 0
            # pylab.figure()
            # y_vals = cdf_hist[feature_i]/np.sum(cdf_hist[feature_i])
            # pylab.plot(y_vals, c='g')
            # pylab.plot(np.arange(y_vals.shape[0]-1)+1,
            #            np.diff(cdf_hist/np.sum(cdf_hist, 1)[:, np.newaxis], 1)[feature_i, :], c='r')
            # for i in selected_arg_hist[feature_i, :]:
            # #pylab.annotate(str(i), xy=(edges[feature_i, i], y_vals[i]))
            # pylab.axvspan(i-0.2, i+0.2, facecolor='0.5', alpha=0.5)
            # pylab.show()

            selected_arg_hist = np.sort(selected_arg_hist, 1)
            self.percentiles_values = np.array(
                [edge[selected_arg_edge] for selected_arg_edge, edge in zip(selected_arg_hist, edges)])

            # percentiles_step = 5
            # percentiles = np.array(
            #     np.percentile(lg_data_n_zero, q=np.arange(0, 100, percentiles_step).tolist(), axis=0)).T
            # selected_percentiles = np.argsort(np.diff(percentiles, 1), 1)[:, -(alphbet_size-1):]
            # selected_percentiles = np.sort(selected_percentiles, 1) + 1
            # selected_percentiles = np.column_stack([[0] * lg_data_n_zero.shape[1], selected_percentiles])
            # self.percentiles = selected_percentiles * percentiles_step
            # print('Selected percentiles')
            # print(self.percentiles)
            # self.percentiles_values = np.array([feature[sel_percentile]
            #                                     for sel_percentile, feature in
            #                                     zip(selected_percentiles, percentiles)])
            # print(self.percentiles_values)

        elif isinstance(self.percentiles[0], list):
            self.percentiles_values = [np.percentile(feature, q=percentiles)
                                       for percentiles, feature in zip(self.percentiles, lg_data_n_zero)]
        elif isinstance(self.percentiles[0], int):
            self.percentiles_values = np.array(np.percentile(lg_data_n_zero, q=self.percentiles, axis=0)).T

    def flat(self, sequence):
        """
        Flats a multidimensional data of words to a unique word sequence.
        Useful for reduction from multidimensional to one dimensional model
        @param sequence: a multidimensional sequence
        @return: equal one dimensional sequence
        """
        # if percentiles value isn't known we assume the sequence is reach enough and contains all the words
        if self.percentiles_values is None:
            num_words = np.max(sequence, 1)
        else:
            try:
                # if percentiles_values is numpy array - we have equal size alphabet in all features
                num_words = [self.percentiles_values.shape[1] + 1 for _ in np.arange(self.percentiles_values.shape[0])]
            except NameError:
                num_words = [feature.shape[0] + 1 for feature in self.percentiles_values]

        num_words = np.ceil(np.log2(num_words))  # num of bits for each feature
        to_discrete = np.cumsum(num_words)[:-1]

        to_discrete = np.insert(to_discrete, 0, 0)
        to_discrete = np.power(2, to_discrete, dtype=int)
        flatten_sequence = np.sum(sequence * to_discrete[np.newaxis, :], 1)
        return flatten_sequence

    def reverse_flat(self, data):
        """
        Reverse a flatten data back to non flatten data
        @param data: data to reverse the flat: array[nxm]: n observations, m features
        @return: array[n x j x k]: n observations, j alphabets (features), k words in each feature
        """
        if self.percentiles_values is None:
            raise Exception('Percentiles values not defined')
        else:
            try:
                # if percentiles_values is numpy array - we have equal size alphabet in all features
                num_words = [self.percentiles_values.shape[1] + 1 for _ in np.arange(self.percentiles_values.shape[0])]
            except NameError:
                num_words = [feature.shape[0] + 1 for feature in self.percentiles_values]

        num_words = np.ceil(np.log2(num_words))  # num of bits for each feature
        to_discrete = np.cumsum(num_words)[:-1]

        to_discrete = np.insert(to_discrete, 0, 0)
        to_indics_pow2 = np.power(2, to_discrete, dtype=int)

        unflatten_sequence = (data[:, np.newaxis] & to_indics_pow2) > 0
        to_discrete = list(zip(to_discrete, np.cumsum(num_words)))
        flatten_sequence = np.array([np.sum(
            unflatten_sequence[:, np.arange(feature_start, feature_start_end, dtype=int)] *
            np.power(2, np.arange(0, feature_start_end - feature_start))[np.newaxis, :], 1)
            for feature_start, feature_start_end in to_discrete]).T
        return flatten_sequence

    def alphabet_size(self, flatten):
        """
        Get the alphabet size induced by the transformer
        @param flatten: whether to get alphabet size when flatten alphabet used or size for each feature
        @return: the alphabet size
        """
        if flatten:
            num_words = [feature.shape[0] + 1 for feature in self.percentiles_values]
            num_words = np.ceil(np.log2(num_words))
            return np.power(2, np.sum(num_words), dtype=int)
        else:
            return [feature.shape[0] + 1 for feature in self.percentiles_values]

    def histogram(self, data, feature_i):
        """
        Create histogram of the data, and the discretization used
        @param feature_i: feature index
        @param data: data for creating the histogram
        @return: create a plot of the data as pdf/cdf
        """
        from matplotlib import pyplot as plt

        lg_data_n_zero = np.log(np.asfarray(data) + 1)
        # notice here we use more sensitive (50 bins) rather then the #bins used in the fit
        # for visualization it is better...
        hist, edges = np.histogram(lg_data_n_zero, bins=50)
        cdf_hist = np.cumsum(hist)

        percentiles_values_max = np.append(self.percentiles_values[feature_i, :], np.max(lg_data_n_zero))
        percentiles_values_max = np.append([0], percentiles_values_max)
        plt.figure()
        y_vals = cdf_hist / cdf_hist[-1]
        hist2, edges2 = np.histogram(lg_data_n_zero, bins=50)
        plt.plot((edges2[:-1] + edges2[1:]) / 2, hist2 / cdf_hist[-1], c='b', label='pdf')
        plt.plot((edges[:-1] + edges[1:]) / 2, y_vals, c='g', label='cdf')
        plt.plot(edges[1:-1], np.diff(cdf_hist / np.sum(cdf_hist)), c='r', label='d/dx(cdf)')
        plt.legend()
        plt.xlabel('log (x+1)')
        for i in range(percentiles_values_max.shape[0] - 1):
            plt.axvspan(percentiles_values_max[i], percentiles_values_max[i + 1],
                        facecolor='#ccccff', alpha=np.float(i + 1) / percentiles_values_max.shape[0])


class MultiDimDiscreteTransformerP(object):
    """
    A class to transform multi-dimensional (multiple feature) from continuous to a discrete alphabets,
    using standard probability functions
    """

    def __init__(self, flat=True):
        """
        @param flat: whether to flat the output to 1 dimension [alphabet size+sum(alphabet size for each dimension)
        """
        self.alphabet_funcs = None
        self.flatten = flat
        self.noise_threshold = 0.0001
        # maximum kolmogorov smirnov distance to use standard distributions otherwise use empirical
        self.max_ks_val = 0.15

    def __call__(self, *args, **kwargs):
        """
        Transforms the sequence to discrete alphabet
        """
        if self.alphabet_funcs is None:
            self.fit(args[0])

        data = args[0]
        mapped2 = np.array(
            [feature_func.to_alphabet(feature) for feature, feature_func in zip(data.T, self.alphabet_funcs)]).T
        if self.flatten:
            mapped2 = self.flat(mapped2)

        return mapped2

    def fit(self, orig_data):
        """
        Fits the transformer based on real data. Uses kolmogorov smirnov test to select the correct distribution
        @param orig_data: original data with no pre-processing
        """
        alphabet_funcs = []
        for feature_data in orig_data.T:
            fit_func = self.fit_feature(feature_data)
            alphabet_funcs.append(fit_func)

        self.alphabet_funcs = alphabet_funcs

    def create_hypothesis(self, feature_data):
        """
        Creates different hypothesis standard probability functions for estimate the underlying data distribution
        @param feature_data: data to fit to
        @return: tuple of hypothesis functions and their cdfs, and kolmogorov-smironov distance
        """
        from scipy.stats import norm, poisson, expon

        # prefer less parameters distributions
        ks_val = []  # kolmogorov smironov test values for various distributions
        hypothesis_funcs = []  # tuple: function used for alphabet binarization, and its cdf

        # here we assume there is some "bias" or uniform in addition to regular signal.
        # this bias can come from non mappable area for example
        step = 0.1
        y_emp_reg_cdf = np.array(np.percentile(feature_data, q=np.arange(0, 100, step).tolist()))
        cdf_p_vals = np.arange(0, 100 * (1 / step), dtype=int)[np.diff(y_emp_reg_cdf) > 0]
        y_emp_reg_cdf = y_emp_reg_cdf[cdf_p_vals]
        cdf_p_vals = (cdf_p_vals + 1) * (step / 100)
        bias_p = np.sum(feature_data == 0) / feature_data.shape[0]
        cdf = poisson.cdf(np.around(y_emp_reg_cdf), np.mean(feature_data)) * (1 - bias_p) + bias_p
        poission_bias_func = NoiseAlphabet('poission-bias', np.mean(feature_data), self.noise_threshold, bias_p)
        hypothesis_funcs.append((poission_bias_func, cdf))
        ks_val.append(np.max(np.abs(cdf - cdf_p_vals)))

        cdf_exp = expon.cdf(y_emp_reg_cdf, 1 / np.mean(feature_data)) * (1 - bias_p) + bias_p
        ks_val.append(np.max(np.abs(cdf_exp - cdf_p_vals)))
        expon_bias_func = NoiseAlphabet('expon-bias', np.mean(feature_data), self.noise_threshold, bias_p)
        hypothesis_funcs.append((expon_bias_func, cdf_exp))

        # poission mixture
        # assignments = np.random.random(feature_data.shape[0])>0.5
        # means = np.array([np.mean(feature_data[assignments]), np.mean(feature_data[~assignments])])
        # diff = np.infty
        # while diff > 0.5:
        # assignments = np.argmax([poisson.pmf(np.around(feature_data), pos) for pos in means], 0)
        #     new_means = np.array([np.mean(feature_data[assignments==i]) for i in [0, 1]])
        #     diff = np.max(np.abs(means - new_means))
        #     means = new_means
        #
        # mix_vals = np.bincount(assignments)/feature_data.shape[0]
        #
        # ks_val_pos_dual = np.max(np.abs(np.dot(mix_vals, [poisson.cdf(np.around(y_emp_reg_cdf), pos)
        #                                                   for pos in means])-cdf_p_vals))
        # print('KS dual poission: {}'.format(ks_val_pos_dual))

        # here we assume 0 is just not mappable so we ignore zeros
        no_zero_data = feature_data[feature_data > 0]

        y_emp_reg_cdf = np.array(np.percentile(no_zero_data, q=np.arange(0, 100, step).tolist()))
        cdf_p_vals = np.arange(0, 100 * (1 / step), dtype=int)[np.diff(y_emp_reg_cdf) > 0]
        y_emp_reg_cdf = y_emp_reg_cdf[cdf_p_vals]
        cdf_p_vals = (cdf_p_vals + 1) * (step / 100)

        cdf = poisson.cdf(np.around(y_emp_reg_cdf), np.mean(no_zero_data))
        ks_val.append(np.max(np.abs(cdf - cdf_p_vals)))
        poission_func = NoiseAlphabet('poission', np.mean(no_zero_data), self.noise_threshold)

        hypothesis_funcs.append((poission_func, cdf))

        cdf_exp = expon.cdf(y_emp_reg_cdf, 1 / np.mean(no_zero_data))
        ks_val.append(np.max(np.abs(cdf_exp - cdf_p_vals)))
        expon_func = NoiseAlphabet('expon', np.mean(no_zero_data), self.noise_threshold)
        hypothesis_funcs.append((expon_func, cdf_exp))

        log_data = np.log(feature_data + 1)
        mix_gaussian = [2]
        gaussians_mean_stds = []
        step = 0.1
        y_emp_cdf = np.array(np.percentile(log_data, q=np.arange(0, 100, step).tolist()))
        cdf_p_vals = np.arange(0, 100 * (1 / step), dtype=int)[np.diff(y_emp_cdf) > 0]
        y_emp_cdf = y_emp_cdf[cdf_p_vals]
        cdf_p_vals = (cdf_p_vals + 1) * (step / 100)

        for mix_num in mix_gaussian:
            # use k-means to fit gaussian mixture
            diff = np.infty
            centers = np.random.random(mix_num) * np.max(log_data)
            centers = np.sort(centers)  # just to have a nicely indexes (e.g larger index=larger value)
            while diff > 0.1:
                dist = (log_data[:, np.newaxis] - centers[np.newaxis, :]) ** 2
                assignments = np.argmin(dist, 1)
                new_centers = np.array([np.mean(log_data[assignments == j]) for j in range(mix_num)])
                diff = np.max(np.abs(new_centers - centers))
                centers = new_centers
            means = centers
            stds = np.array([np.std(log_data[assignments == j]) for j in range(mix_num)])
            gaussians = [norm(means[j], stds[j]) for j in range(mix_num)]
            mix_vals = np.array([np.sum(gauss.pdf(log_data)) for gauss in gaussians])
            mix_vals /= np.sum(mix_vals)
            gaussians_mean_stds.append((means, stds, mix_vals))

            cdf = np.dot(mix_vals, [gauss.cdf(y_emp_cdf) for gauss in gaussians])
            ks_val.append(np.max(np.abs(cdf - cdf_p_vals)))

            hypothesis_funcs.append((GaussianMixtureAlphabet(means, stds, mix_vals), cdf))

        # with bias (no zeros)
        bias_p = np.sum(log_data == 0)/log_data.shape[0]
        log_data = log_data[log_data > 0]
        mix_gaussian = [2]
        gaussians_mean_stds = []
        y_emp_cdf = y_emp_cdf[1:]
        cdf_p_vals = cdf_p_vals[1:]
        for mix_num in mix_gaussian:
            # use k-means to fit gaussian mixture
            diff = np.infty
            centers = np.random.random(mix_num) * np.max(log_data)
            centers = np.sort(centers)  # just to have a nicely indexes (e.g larger index=larger value)
            while diff > 0.1:
                dist = (log_data[:, np.newaxis] - centers[np.newaxis, :]) ** 2
                assignments = np.argmin(dist, 1)
                new_centers = np.array([np.mean(log_data[assignments == j]) for j in range(mix_num)])
                diff = np.max(np.abs(new_centers - centers))
                centers = new_centers
            means = centers
            stds = np.array([np.std(log_data[assignments == j]) for j in range(mix_num)])
            gaussians = [norm(means[j], stds[j]) for j in range(mix_num)]
            mix_vals = np.array([np.sum(gauss.pdf(log_data)) for gauss in gaussians])
            mix_vals /= np.sum(mix_vals)
            gaussians_mean_stds.append((means, stds, mix_vals))

            cdf = bias_p+(1-bias_p)*np.dot(mix_vals, [gauss.cdf(y_emp_cdf) for gauss in gaussians])
            ks_val.append(np.max(np.abs(cdf - cdf_p_vals)))

            hypothesis_funcs.append((GaussianMixtureAlphabet(means, stds, mix_vals), cdf))

        return hypothesis_funcs, ks_val

    def fit_feature(self, feature_data):
        """
        selects the best hypothesis to describe a feature
        @param feature_data:
        @return:
        """
        hypothesis_funcs, ks_vals = self.create_hypothesis(feature_data)

        # empirical alphabet fits itself based on the data, so ks distance isn't used here
        ks_vals.append(self.max_ks_val)
        hypothesis_funcs.append((EmpiricalAlphabet(feature_data), None))
        selected_hypothesis_index = np.argmin(ks_vals)
        selected_hypothesis, selected_cdf = hypothesis_funcs[selected_hypothesis_index]

        print(ks_vals)
        return hypothesis_funcs[selected_hypothesis]

    def visual_fit(self, feature_data, feature_name):
        """
        shows a histogram of different hypothesis functions
        @param feature_name: name of the feature
        @param feature_data: real data of the feature
        @return:creates histogram of the data
        """
        from matplotlib import pyplot as plt

        hypothesis_funcs, ks_vals = self.create_hypothesis(feature_data)
        plt.figure()
        plt.title(feature_name)
        step = 0.1
        y_emp_reg_cdf = np.array(np.percentile(feature_data, q=np.arange(0, 100, step).tolist()))
        cdf_p_vals = np.arange(0, 100 * (1 / step), dtype=int)[np.diff(y_emp_reg_cdf) > 0]
        y_emp_reg_cdf = y_emp_reg_cdf[cdf_p_vals]
        cdf_p_vals = (cdf_p_vals + 1) * (step / 100)
        y_emp_reg_cdf = np.log(y_emp_reg_cdf + 1)

        y_emp_reg_cdf_no_zeros = np.array(
            np.percentile(feature_data[feature_data > 0], q=np.arange(0, 100, step).tolist()))
        cdf_p_vals_no_zeros = np.arange(0, 100 * (1 / step), dtype=int)[np.diff(y_emp_reg_cdf_no_zeros) > 0]
        y_emp_reg_cdf_no_zeros = np.log(y_emp_reg_cdf_no_zeros[cdf_p_vals_no_zeros] + 1)

        log_data = np.log(feature_data + 1)
        y_emp_cdf_log = np.array(np.percentile(log_data, q=np.arange(0, 100, step).tolist()))
        cdf_p_vals_log = np.arange(0, 100 * (1 / step), dtype=int)[np.diff(y_emp_cdf_log) > 0]
        y_emp_cdf_log = y_emp_cdf_log[cdf_p_vals_log]

        plt.plot(y_emp_reg_cdf, cdf_p_vals, label='cdf empirical')
        min_ks = np.min(ks_vals)
        for hyp, ks in zip(hypothesis_funcs, ks_vals):
            hypothesis, cdf = hyp
            line_style = '--'
            if ks == min_ks:
                line_style = '-'
            if hypothesis.name == 'poission' or hypothesis.name == 'expon':
                plt.plot(y_emp_reg_cdf_no_zeros, cdf, line_style, label='{}-{:.3}'.format(hypothesis.name, ks))
            elif hypothesis.name.startswith('gaussian'):
                if y_emp_cdf_log.shape[0] == cdf.shape[0]+1:
                    plt.plot(y_emp_cdf_log[1:], cdf, line_style, label='{}-{:.3}'.format(hypothesis.name, ks))
                else:
                    plt.plot(y_emp_cdf_log, cdf, line_style, label='{}-{:.3}'.format(hypothesis.name, ks))
            else:
                plt.plot(y_emp_reg_cdf, cdf, line_style, label='{}-{:.3}'.format(hypothesis.name, ks))
        plt.xlabel('log(data+1)')
        plt.ylabel('cdf')
        plt.legend()
        print(ks_vals)
        plt.show()

    def flat(self, sequence):
        """
        Flats a multidimensional data of words to a unique word sequence.
        Useful for reduction from multidimensional to one dimensional model
        @param sequence: a multidimensional sequence
        @return: equal one dimensional sequence
        """
        # if percentiles value isn't known we assume the sequence is reach enough and contains all the words
        if self.percentiles_values is None:
            num_words = np.max(sequence, 1)
        else:
            try:
                # if percentiles_values is numpy array - we have equal size alphbet in all features
                num_words = [self.percentiles_values.shape[1] + 1 for i in np.arange(self.percentiles_values.shape[0])]
            except NameError:
                num_words = [feature.shape[0] + 1 for feature in self.percentiles_values]

        num_words = np.ceil(np.log2(num_words))  # num of bits for each feature
        to_indics = np.cumsum(num_words)[:-1]

        to_indics = np.insert(to_indics, 0, 0)
        to_indics = np.power(2, to_indics, dtype=int)
        flatten_sequence = np.sum(sequence * to_indics[np.newaxis, :], 1)
        return flatten_sequence

    def reverse_flat(self, data):
        """
        Reverse a flatten data back to non flatten data
        @param data: data to reverse the flat: array[nxm]: n observations, m features
        @return: array[n x j x k]: n observations, j alphabets (features), k words in each feature
        """
        if self.percentiles_values is None:
            raise Exception('Percentiles values not defined')
        else:
            try:
                # if percentiles_values is numpy array - we have equal size alphabet in all features
                num_words = [self.percentiles_values.shape[1] + 1 for i in np.arange(self.percentiles_values.shape[0])]
            except NameError:
                num_words = [feature.shape[0] + 1 for feature in self.percentiles_values]

        num_words = np.ceil(np.log2(num_words))  # num of bits for each feature
        to_discrete = np.cumsum(num_words)[:-1]

        to_discrete = np.insert(to_discrete, 0, 0)
        to_indics_pow2 = np.power(2, to_discrete, dtype=int)

        unflatten_sequence = (data[:, np.newaxis] & to_indics_pow2) > 0
        to_discrete = list(zip(to_discrete, np.cumsum(num_words)))
        flatten_sequence = np.array([np.sum(
            unflatten_sequence[:, np.arange(feature_start, feature_start_end, dtype=int)] *
            np.power(2, np.arange(0, feature_start_end - feature_start))[np.newaxis, :], 1)
            for feature_start, feature_start_end in to_discrete]).T
        return flatten_sequence

    def alphabet_size(self, flatten):
        """
        Gets the alphabet size induced by the transformer
        @param flatten: whether flatten alphabet size used (all feature together) or independently
        @return: the alphabet size
        """
        if flatten:
            num_words = [2 for _ in range(len(self.alphabet_funcs))]
            num_words = np.ceil(np.log2(num_words))
            return np.power(2, np.sum(num_words), dtype=int)
        else:
            return [2 for _ in range(len(self.alphabet_funcs))]

    def histogram(self, data, feature_i):
        """
        Create histogram of the data, and the discretization used
        @param data:
        @return:
        """
        from matplotlib import pyplot as plt

        func_name = self.alphabet_funcs[feature_i].name
        if func_name.startswith('gaussian'):
            data = np.log(np.asfarray(data) + 1)

        plt.figure()
        hist2, edges2 = np.histogram(data, bins=30)
        plt.plot((edges2[:-1] + edges2[1:]) / 2, hist2 / np.sum(hist2), c='b', label='pdf-data')
        self.alphabet_funcs[feature_i].histogram((edges2[:-1] + edges2[1:]) / 2)

        if func_name.startswith('gaussian'):
            alphbet_convert = self.alphabet_funcs[feature_i].to_alphabet(np.exp((edges2[:-1] + edges2[1:]) / 2) - 1)
        else:
            alphbet_convert = self.alphabet_funcs[feature_i].to_alphabet((edges2[:-1] + edges2[1:]) / 2)
        diff_vals = ((edges2[:-1] + edges2[1:]) / 2)[np.append(True, np.diff(alphbet_convert) > 0)]
        for i in range(diff_vals.shape[0] - 1):
            plt.axvspan(diff_vals[i], diff_vals[i + 1],
                        facecolor='#ccccff', alpha=np.float(i + 1) / diff_vals.shape[0])
        plt.legend()


class NoiseAlphabet:
    """
    Describes a signal for which most of the data is noise, and only high values are real data.
    """

    def __init__(self, name, mean, threshold, bias=0):
        self.name = name
        self.mean = mean
        self.bias = bias
        self.noise_threshold = threshold

    def get_pdf_func(self):
        """
        gets the pdf function associated with this type of data

        @return: a pdf function
        """
        from scipy.stats import poisson, expon

        if self.name.startswith('poission'):
            pdf_func = lambda x: poisson(self.mean).pmf(np.around(x))
        elif self.name.startswith('expon'):
            pdf_func = lambda x: expon(1 / self.mean).pdf(x)
        return pdf_func

    def to_alphabet(self, data):
        """
        transform data to alphabet
        @param data: data to transform
        @return: transformed alphabet
        """
        pdf_func = self.get_pdf_func()
        not_noise = pdf_func(data) < self.noise_threshold * (1 - self.bias)
        not_missing = data > 0  # remove zeros which are expected to be just a noise
        return np.array(not_noise & not_missing, dtype=int)

    def histogram(self, x):
        """
        creates pdf histogram for the data using the assumed background distribution
        @param x:
        """
        import matplotlib.pyplot as plt

        pdf_func = self.get_pdf_func()
        plt.plot(x, pdf_func(x) * (1 - self.bias), '--', label=self.name)


class GaussianMixtureAlphabet:
    """
    Describes a signal that comes from multiple gaussian sources,
    where the maximum p describes the origin source for observation
    """

    def __init__(self, means, stds, mix_vals):
        self.name = 'gaussian mix{}'.format(len(means))
        self.means = means
        self.stds = stds
        self.mix_vals = mix_vals  # TODO: testing this normalization

    def to_alphabet(self, data):
        """
        transform data to alphabet
        @param data: data to transform
        @return: transformed alphabet
        """
        from scipy.stats import norm

        gaussians = [norm(*mix_vals) for mix_vals in zip(self.means, self.stds)]
        # TODO: testing normalization. noramlize by 1-mixture so only strong evidence will be considered as signal
        pdfs = (1 - self.mix_vals[:, np.newaxis]) * np.array([gaussian.pdf(np.log(data + 1)) for gaussian in gaussians])
        # return the pdf that best fits to the data
        return np.argmax(pdfs, 0)

    def histogram(self, x):
        """
        Creates pdf for the gaussian mixture of the underling data
        @param x: data to get pdf for
        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        gaussians = [norm(*mix_vals) for mix_vals in zip(self.means, self.stds)]
        for gaus_i, gaus in enumerate(gaussians):
            plt.plot(x, gaus.pdf(x) * self.mix_vals[gaus_i], '--', label='gaussian-{}'.format(gaus_i))


class EmpiricalAlphabet:
    """
    Describes a signal that comes from multiple gaussian sources,
    where the maximum p describes the origin source for observation
    """

    def __init__(self, data):
        self.name = 'empiricalAlphabet'
        self.threshold = self._fit(data)

    @staticmethod
    def _fit(data):
        step = 10
        y_emp_reg_cdf = np.array(np.percentile(data, q=np.arange(0, 100, step).tolist()))
        cdf_p_vals = np.arange(0, 100 * (1 / step), dtype=int)[np.diff(y_emp_reg_cdf) > 0]
        y_emp_reg_cdf = y_emp_reg_cdf[cdf_p_vals]
        pdf = np.diff(np.append(0, cdf_p_vals / (100 / step)))
        return y_emp_reg_cdf[np.argmax(np.diff(pdf) > 0) + 1]

    def to_alphabet(self, data):
        """
        transform data to alphabet
        @param data: data to transform
        @return: transformed alphabet
        """
        return np.array(data > self.threshold, dtype=int)

    def histogram(self, x):
        """
        There is no real histogram for empirical alphabet
        @param x:
        """
        pass
