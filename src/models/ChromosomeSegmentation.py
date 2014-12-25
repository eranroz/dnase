"""
Data model for genome domains

This class enrich a segmentation of genome with extra features, integrate the extra feature with the domains.
"""
import os
import re

import numpy as np

from config import BED_GRAPH_RESULTS_DIR, MEAN_DNASE_DIR
from data_provider import SeqLoader
from data_provider import featureLoader

__author__ = 'eranroz'


class SegmentationFeatures:
    """
    Supported features that can be added to regions in segmentation
    """
    Position = 1
    RegionLengths = 2
    OpenClosed = 3
    AverageAccessibility = 4
    Markers = 5
    Motifs = 6
    ChIP = 7  # chip form encode??
    HasCTCFBoundaries = 8

    @staticmethod
    def name(feature):
        """
        Feature to string name
        @param feature: feature to cover
        @return: name of feature
        """
        features_names = [
            'Position',
            'RegionLengths',
            'OpenClosed',
            'AverageAccessibility',
            'Markers',
            'Motifs',
            'ChIP',
            'HasCTCFBoundaries'
        ]
        return features_names[feature - 1]


class ChromosomeSegmentation:
    """
    Class for chromosome segmentation to regions
    """

    def __init__(self, cell_type, segmentation, chromosome, model):
        # (cell_type, seg, chromosome, model)
        # cell_type, segmentation, resolution, chromosome
        # meta data:
        """
        @type model: DNaseClassifier
        @param cell_type: name of cell type to get additional data for
        @param segmentation: segmentation array
        @param chromosome: name of chromosome
        @param model: model used for creating the segmentation
        """
        self.cell_type = cell_type
        self.resolution = model.resolution
        self.model = model
        self.chromosome = chromosome
        self.data = self._set_segmentation(segmentation)
        # core features - loaded during construction
        self.feature_mapping = [SegmentationFeatures.Position, SegmentationFeatures.RegionLengths,
                                SegmentationFeatures.OpenClosed]
        self.markers = []


    def _set_segmentation(self, segmentation):
        """
        Transforms bins of open/closed to compact array of [[start, length, open/closed]]
        @param segmentation: 1 for open, 0 for closed
        """

        # TODO: assign open and closed correctly based on cell type
        """
        cell_data = self.model.classify_file(os.path.join(MEAN_DNASE_DIR, self.cell_type), chromosomes=self.chromosome)
        boundaries = np.convolve(segmentation[self.chromosome], [1, -1])
        boundaries = np.append(0, np.where(boundaries))  # add boundary at beginning
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            cell_data[self.chromosome][start:end] = np.mean(cell_data[self.chromosome][start:end]) > 0.5  # threshold
        """

        vv = np.convolve(segmentation, [1, -1])
        # vv = np.append(-1, vv)  # start always with closed
        boundaries = np.where(vv)[0]
        if boundaries[0] != 0:  # add boundary at beginning
            boundaries = np.append(0, boundaries)
        lengths = np.diff(np.append(boundaries, len(vv) + 1))  # add the last
        open_closed = [0, 1] * np.floor(len(boundaries) / 2) + ([0] if len(boundaries) % 2 == 1 else [])
        return np.array([boundaries, lengths, open_closed]).T

    def feature_matrix(self):
        """
        Get regions feature matrix - rows are regions and column are features

        @return: list of column mapping {SegmentationFeatures} and regions data
        """
        return self.feature_mapping, self.data

    def load(self, features, arg=None):
        """
        Loads features or feature
        @param features: {SegmentationFeatures} or list of {SegmentationFeatures}
        """
        if not isinstance(features, list):
            features = [features]

        extra_features = []
        for feature in features:
            if feature in self.feature_mapping:
                continue  # feature already loaded

            if feature == SegmentationFeatures.AverageAccessibility:
                extra_features.append(self._load_average_accessibility())
                self.feature_mapping.append(SegmentationFeatures.AverageAccessibility)
            elif feature == SegmentationFeatures.Markers:
                extra_features.append(self._load_markers(arg))
                self.feature_mapping += self.markers
            elif feature == SegmentationFeatures.Motifs:
                extra_features.append(self._load_motifs())
                self.feature_mapping.append(SegmentationFeatures.Motifs)
            elif feature == SegmentationFeatures.HasCTCFBoundaries:
                extra_features.append(self._load_has_ctcf_boundaries())
                self.feature_mapping.append(SegmentationFeatures.HasCTCFBoundaries)
            else:
                raise Exception("Unknown feature")
        if len(extra_features) > 0:
            self.data = np.column_stack([self.data] + extra_features)

    def get(self, features):
        """
        Get regions matrix with specific features.
        @param features: features to get
        @return: matrix of regions data (rows: regions, columns: features (in same order))
                for multiple columns features such as Markers use get_labels to get labels for columns
        """
        if not isinstance(features, list):
            features = [features]
        features_to_load = [f for f in features if f not in self.feature_mapping]
        self.load(features_to_load)
        feature_indics = []
        for f in features:
            if f == SegmentationFeatures.Markers:
                for marker in self.markers:
                    feature_indics.append(self.feature_mapping.index(marker))
            else:
                feature_indics.append(self.feature_mapping.index(f))

        return self.data[:, feature_indics]

    def get_labels(self, features):
        """
        Get the corresponding labels for get
        @param features: request features
        """
        labels = []
        for f in features:
            if f == SegmentationFeatures.Markers:
                for marker in self.markers:
                    labels.append(marker)
            else:
                labels.append(SegmentationFeatures.name(f))
        return labels

    def _load_average_accessibility(self):
        # load original dnase results and select only the required chromosome
        accessibility_data = SeqLoader.load_result_dict(os.path.join(MEAN_DNASE_DIR, "%s.mean.npz" % self.cell_type))
        accessibility_data = accessibility_data[self.chromosome]
        # down-sample to the resolution. original resolution for mean dnase is 20
        accessibility_data = SeqLoader.down_sample(accessibility_data, (self.resolution / 20))
        region_matrix = self.get([SegmentationFeatures.Position, SegmentationFeatures.RegionLengths])

        region_accessibility = np.zeros(region_matrix.shape[0])
        end_position = region_matrix[-1, 0] + region_matrix[-1, 1]
        if accessibility_data.shape[0] < end_position:
            accessibility_data = np.pad(accessibility_data, [0, end_position - accessibility_data.shape[0]], 'constant')
        lengths = region_matrix[:, 1]
        length_set = list(set(lengths))
        length_set.sort()
        prev_length = 0

        for curr_length in length_set:
            selector = lengths >= curr_length
            sel_start = region_matrix[selector, 0]
            while prev_length < curr_length:
                region_accessibility[selector] += accessibility_data[sel_start + prev_length]
                prev_length += 1

        # normalize by length
        region_accessibility /= region_matrix[:, 1]

        return region_accessibility

    def _load_markers(self, markers=None):
        markers, experiments = SeqLoader.load_experiments(self.cell_type, markers, [self.chromosome],
                                                          resolution=self.resolution)
        markers = markers[self.chromosome]
        self.markers = experiments
        region_matrix = self.get([SegmentationFeatures.Position, SegmentationFeatures.RegionLengths])
        region_matrix[:, 1] += region_matrix[:, 0]
        marker_avg = np.zeros((region_matrix.shape[0], markers.shape[0]))

        for reg_i, start_end in enumerate(region_matrix):
            marker_avg[reg_i, :] = np.average(markers[:, start_end[0]:start_end[1]], 1).T

        return marker_avg

    def _load_has_ctcf_boundaries(self):
        # TODO: use specific cell type instead of hmec
        ctcf_signals = featureLoader.load_hmec_ctcf_peaks()
        ctcf_signals = ctcf_signals[self.chromosome]  # only on scored chromosome

        # create simple segmentation
        p_val_threshold = 20  # not sure
        ctcf_signals = ctcf_signals[ctcf_signals[:, 2] <= p_val_threshold]

        # sometimes peak are wide so set their average as boundary

        peaks_avg = (ctcf_signals[:, 0] + ctcf_signals[:, 1]) / 2.0
        peaks_avg.sort()
        golden_segment = np.array([peaks_avg[:-1], np.diff(peaks_avg)]).T
        position_length = self.get([SegmentationFeatures.Position, SegmentationFeatures.RegionLengths])
        position_length *= self.resolution

        distance_threshold = 0.05  # threshold multiplied by length
        return np.array(
            [np.any(np.abs(golden_segment[:, 0] - reg[0]) < reg[1] * distance_threshold) for reg in position_length])

    def _load_motifs(self):
        # TODO: implement
        raise NotImplementedError


def load(cell_type, model, chromosomes=None, segmentation=None):
    """
    Loads genome or sub genome and assigns states for specific cell type with the given segmentation

    @param segmentation: segmentation to use or none (for uniform model segmentation)
    @type model: DNaseClassifier
    @param model: model used for creating segmentation
    @param cell_type: name of cell type to analyze
    @param chromosomes: chromosomes to load
    """

    if segmentation is None:
        # in that case a uniform segmentation for all based on model segmentation
        segmentation_file = model.segmentation_file_path()
        segmentation = SeqLoader.load_result_dict(segmentation_file)
    chromosomes = chromosomes or segmentation.keys()
    seg_dict = dict()
    for chromosome in chromosomes:
        seg = segmentation[chromosome]
        seg_dict[chromosome] = ChromosomeSegmentation(cell_type, seg, chromosome, model)
    return seg_dict