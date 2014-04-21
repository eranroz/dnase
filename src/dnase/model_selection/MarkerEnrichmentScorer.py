"""
Score function for signal enrichment within open regions
"""
import numpy as np
from data_provider import SeqLoader
from dnase.model_selection.BaseScoreModel import BaseScoreModel

__author__ = 'eranroz'


class MarkerEnrichmentScorer(BaseScoreModel):
    """
    Score function for enrichment of markers (histone modifications) within open regions
    @param train_cell_type: cell type used for training
    @param score_chromosome: chromosome name used for scoring

    Assumptions:
    * larger bins tend to smooth signals, so as we take larger bins the score should be lower
    * H3K27Ac modification is common in open regions
    """
    def __init__(self, train_cell_type, score_chromosome, signal_type='H3K27ac'):
        signal = SeqLoader.load_experiments(train_cell_type.split('_')[0], [signal_type], [score_chromosome])[score_chromosome]
        self.signal = signal[0]  # only one experiment

    @staticmethod
    def reg_enrichment(segmentation, enrichment):
        """
        Ratio of enrichment in open compared to close region (normalized by their lengths)
        @param segmentation: segmentation to score
        @param enrichment: signal to use for enrichment lookup
        @return: ratio of enrichment in open compared to close
        """
        markers_per_open = np.sum(enrichment[segmentation]) / np.sum(segmentation)
        markers_per_close = np.sum(enrichment[~segmentation]) / np.sum(~segmentation)
        return markers_per_open / markers_per_close

    def score(self, model=None, resolution=None, training_set=None):
        """
        Score according to L{MarkerEnrichmentScorer}.
        @param model: model used for creating segmentation
        @param training_set: array of segmentations
        @param resolution: number of bp for bins
        """
        enrichment = SeqLoader.down_sample(self.signal, resolution / 20)
        hmm_seg = [MarkerEnrichmentScorer.reg_enrichment(seq, enrichment[0:seq.shape[0]]) for seq in training_set]
        return np.mean(hmm_seg)