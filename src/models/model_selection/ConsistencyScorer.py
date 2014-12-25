"""
Score for segmentation based on consistency between segmentation of different samples of same cell type
"""
import numpy as np
from models.model_selection.BaseScoreModel import BaseScoreModel

__author__ = 'eranroz'


class ConsistencyScorer(BaseScoreModel):
    """
    Scores segmentation model based on consistent between different samples of same cell type
    Assumptions:
    * Larger bins give greater consistency
    """

    def score(self, model=None, resolution=None, training_set=None):
        min_len = min(seq.shape[0] for seq in training_set)
        segmentation_arr = []
        for seq in training_set:
            segmentation_arr.append(seq[0:min_len])

        mean_seg = np.zeros(min_len)
        for seq in training_set:
            mean_seg += (seq[0:min_len] * 1)
        mean_seg /= len(segmentation_arr)
        mean_seg[mean_seg <= 0.2] = 0
        mean_seg[mean_seg >= 0.8] = 1
        sum_good = np.sum((mean_seg == 0)) + np.sum((mean_seg == 1))
        consistent = [(np.sum((mean_seg[seg] == 1)) + np.sum((mean_seg[~seg] == 0))) / sum_good for seg in
                      segmentation_arr]
        return np.mean(consistent)
