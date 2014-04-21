"""
Score segmentation based on number of genes that fall within domains
"""
import numpy as np
from data_provider import featureLoader

__author__ = 'eranroz'


def seg_permutation_boundaries(segmentation, num_permutations):
    """
    random permutations for segmentation with [Flase, False, True, False...] etc
    @param num_permutations: number of permutations to yield
    @param segmentation: original segmentation according to which to create random segments (same stats)
    """
    vv = np.convolve(segmentation, [1, -1])
    boundaries = np.append(0, np.where(vv))  # add boundary at beginning
    lengths = np.diff(np.append(boundaries, len(vv) - 1))  # add the last
    open_pos_sel = np.append(False, vv[vv != 0] == 1)
    lengths_open = lengths[open_pos_sel]
    lengths_close = lengths[~open_pos_sel]
    extend = (lengths_open.shape != lengths_close.shape)
    for _ in range(num_permutations):
        # permutation in place
        np.random.shuffle(lengths_open)
        np.random.shuffle(lengths_close)
        if extend:
            lengths = np.array([lengths_close, np.pad(lengths_open, [0, 1], 'constant')]).T.flatten()
            lengths = lengths[:-1]
        else:
            lengths = np.array([lengths_close, lengths_open]).T.flatten()

        boundaries = np.cumsum(lengths)
        yield boundaries


class SegmentedGenesScorer:
    """
    Scores segments based on number of genes that fall within border of segments
    @param score_chromosome: chromosome name used for scoring

    Score assumptions:
    * breaking genes between segments isn't cool
    """
    def __init__(self, score_chromosome):
        gene_signals = featureLoader.load_known_genes()

        # only on scored chromosome
        self.gene_signals = gene_signals[score_chromosome]

    def calculate_break(self, boundaries):
        """
        Calculates genes breaks
        @param boundaries: positions of boundaries between regions
        @return:  number of broken genes
        """
        # region boundary that falls within transcript
        broken_genes = np.zeros(self.gene_signals.shape[0], dtype='bool')
        for boundary in boundaries:
            # start before boundary but ends after it
            broken_genes |= (self.gene_signals[:, 0] < boundary) & (self.gene_signals[:, 1] > boundary)
        n_breaks = np.sum(broken_genes)
        return n_breaks

    def score(self, segmentation, resolution):
        """
        Score compared to random segmentation - e.g where it fall for different permutations
        @param segmentation: segmentation to score
        @param resolution: number of bp for bins
        """

        vv = np.convolve(segmentation, [1, -1])
        boundaries = np.where(vv)[0]
        breaks = self.calculate_break(boundaries * resolution)

        break_stats = []
        num_permutations = 500
        for per in seg_permutation_boundaries(segmentation, num_permutations):
            break_stats.append(self.calculate_break(per * resolution))

        # using cdf of gaussian
        #p_func = scipy.stats.gaussian_kde(break_stats)
        #score = p_func.cdf(breaks)
        # using empirical score

        score = np.sum(break_stats > breaks) / num_permutations
        print(np.mean(break_stats))
        print(score)
        print(breaks)
        #import pylab

        #pylab.hist(break_stats, bins=20)
        #pylab.show()

        return score