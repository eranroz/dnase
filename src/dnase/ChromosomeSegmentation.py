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
        #(cell_type, seg, chromosome, model)
        #cell_type, segmentation, resolution, chromosome
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
        self.markers = None


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
        #vv = np.append(-1, vv)  # start always with closed
        boundaries = np.append(0, np.where(vv))  # add boundary at beginning
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
                self.feature_mapping.append(SegmentationFeatures.Markers)
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
        """
        if not isinstance(features, list):
            features = [features]
        features_to_load = [f for f in features if f not in self.feature_mapping]
        self.load(features_to_load)
        feature_indics = [self.feature_mapping.index(f) for f in features]
        return self.data[:, feature_indics]

    def _load_average_accessibility(self):
        # load original dnase results and select only the required chromosome
        accessibility_data = SeqLoader.load_result_dict(os.path.join(MEAN_DNASE_DIR, "%s.mean.npz" % self.cell_type))
        accessibility_data = accessibility_data[self.chromosome]
        # down-sample to the resolution. original resolution for mean dnase is 20
        accessibility_data = SeqLoader.down_sample(accessibility_data, (self.resolution / 20))
        region_matrix = self.get([SegmentationFeatures.Position, SegmentationFeatures.RegionLengths])
        region_matrix[:, 1] += region_matrix[:, 0]
        # keep same size as the matrix of segmentations
        avg_accessibility = np.zeros(region_matrix[-1, 1]+1)
        avg_accessibility[0:np.minimum(region_matrix[-1, 1], accessibility_data.shape[0])] = accessibility_data[0:np.minimum(
            region_matrix[-1, 1], accessibility_data.shape[0])]
        region_accessibility = np.array([np.average(avg_accessibility[start:end]) for start, end in region_matrix])
        return region_accessibility

    def _load_markers(self, markers=None):
        if markers is None:
            markers = ['Histone_H2A.Z', 'Histone_H2AK5ac', 'Histone_H2BK120ac', 'Histone_H2BK12ac', 'Histone_H2BK15ac',
                       'Histone_H2BK20ac', 'Histone_H2BK5ac', 'Histone_H3K14ac', 'Histone_H3K18ac', 'Histone_H3K23ac',
                       'Histone_H3K23me2', 'Histone_H3K27ac', 'Histone_H3K27me3', 'Histone_H3K36me3', 'Histone_H3K4ac',
                       'Histone_H3K4me1', 'Histone_H3K4me2', 'Histone_H3K4me3', 'Histone_H3K56ac', 'Histone_H3K79me1',
                       'Histone_H3K79me2', 'Histone_H3K9ac', 'Histone_H3K9me3', 'Histone_H3T11ph', 'Histone_H4K20me1',
                       'Histone_H4K5ac', 'Histone_H4K8ac', 'Histone_H4K91ac', 'Histone_H3K9me1' 'Histone_H2AK9ac']
        #dataDownloader.download_experiment(self.cell_type, markers)
        markers = SeqLoader.load_experiments(self.cell_type, markers, [self.chromosome])
        # TODO: maybe select only significant markers
        # TODO: always 20?
        markers = SeqLoader.down_sample(markers, self.resolution / 20)

        region_matrix = self.get([SegmentationFeatures.Position, SegmentationFeatures.RegionLengths])
        region_matrix[:, 1] += region_matrix[:, 0]
        marker_avg = np.array([np.average(markers[:, start:end], 1) for start, end in region_matrix])
        return marker_avg
        # TODO: sum any find of marker a more details map is preserved in markers

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
        #TODO: implement
        raise NotImplementedError


def load(cell_type, model, chromosomes=None):
    """
    Loads genome or sub genome and assigns states for specific cell type with the given segmentation

    @type model: DNaseClassifier
    @param model: model used for creating segmentation
    @param cell_type: name of cell type to analyze
    @param chromosomes: chromosomes to load
    """
    segmentation = SeqLoader.load_result_dict(model.segmentation_file_path())
    chromosomes = chromosomes or segmentation.keys()
    seg_dict = dict()
    #items = list(segmentation.items())
    for chromosome in chromosomes:
        seg = segmentation[chromosome]
        seg_dict[chromosome] = ChromosomeSegmentation(cell_type, seg, chromosome, model)
    return seg_dict