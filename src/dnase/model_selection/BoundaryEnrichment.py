"""
Score function by enrichment of "marker" such as CTCF around boundary

TODO: not implemented
"""
import numpy as np
from data_provider import featureLoader

__author__ = 'eranroz'


def reg_start_len(reg_classification, only_open=False):
    """
    transforms segmentation to bins [True, false...]=>[[Start, Length],...]
    @param reg_classification: segmentation e.g [True, false...]
    @param only_open: only open regions (True)
    @return: 2d array. each row is [Start, Length]
    """
    vv = np.convolve(reg_classification, [1, -1])
    #vv = np.append(-1, vv)  # start always with closed
    boundaries = np.append(0, np.where(vv))  # add boundary at beginning
    lengths = np.diff(np.append(boundaries, len(vv) + 1))  # add the last
    if only_open:
        open_pos_sel = np.append(False, vv[vv != 0] == 1)
        lengths = lengths[open_pos_sel]
        return np.array([boundaries[open_pos_sel], lengths]).T
    else:
        return np.array([boundaries, lengths]).T


class CTCFModelScore:
    """
    Score based on ctcf signal around segments.
    CTCF insulator from ENCODE: mammary epithelial cells, brest tissue
    Score assumptions:
    * segments are surrounded by enriched CTCF signal
    * since we use breast tissue consider whether the cell type have similar CTCF peaks patterns
    @param p_val_threshold:threshold for selecting CTCF signals
    """

    def __init__(self, score_chromosome, p_val_threshold=20):
        ctcf_signals = featureLoader.load_hmec_ctcf_peaks()
        ctcf_signals = ctcf_signals[score_chromosome]  # only on scored chromosome

        # create simple segmentation
        ctcf_signals = ctcf_signals[ctcf_signals[:, 2] <= p_val_threshold]

        # sometimes peak are wide so set their average as boundary

        peaks_avg = (ctcf_signals[:, 0] + ctcf_signals[:, 1]) / 2.0
        peaks_avg.sort()
        golden_segment = np.array([peaks_avg[:-1], np.diff(peaks_avg)]).T

        self.golden_model = golden_segment

    def score(self, segmentation, resolution):
        """
        calculate heatmap for ctcf for all regions (in and around)
        and calculate it for different permutations and find where
        @param resolution: resolution for bins
        @param segmentation: genome segmentation to score
        """
        raise NotImplementedError

    def old_score(self, segmentation, resolution):
        """
        score according to number of segments that have close ctcf to their boundaries
        @param resolution: resolution for bins
        @param segmentation: genome segmentation to score
        """
        def find_closest(A, target):
            idx = A.searchsorted(target)
            idx = np.clip(idx, 1, len(A) - 1)
            left = A[idx - 1]
            right = A[idx]
            idx -= target - left < right - target
            return idx
        new_regions_start = reg_start_len(segmentation)
        # transform from bins to real bp location
        new_regions_start *= resolution
        # finds the closest segment from golden model to the one in the segmentation
        # e.g mapping golden -> segment index
        closest_in_segment = find_closest(new_regions_start[:, 0], self.golden_model[:, 0])
        # keep it unique
        distance_mapper = np.array(np.abs(self.golden_model[:, 0] - new_regions_start[closest_in_segment, 0]))
        closest_in_segment_uniq = list(set(closest_in_segment))
        closest_in_segment_uniq.sort()
        closest_in_segment_golden = []
        for v in closest_in_segment_uniq:
            relevant_seg = closest_in_segment == v
            indices = np.where(relevant_seg)[0]
            min_indic = np.argmin(distance_mapper[relevant_seg])
            closest_in_segment_golden.append(indices[min_indic])

        golden_len = self.golden_model[closest_in_segment_golden, 1]
        segmentation_len = new_regions_start[closest_in_segment_uniq, 1]

        # number of regions that have roughly the same length as expected
        segment_length_ratio = np.abs(np.log(segmentation_len / golden_len))
        score = np.sum(np.abs(segment_length_ratio - 1.0) < 0.1)  # about 90% agreement
        return score / segment_length_ratio.shape[0]