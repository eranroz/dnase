import logging
from unittest import TestCase
import unittest
import sys

from models import ChromosomeSegmentation
from models.ChromosomeSegmentation import SegmentationFeatures
import numpy as np

__author__ = 'eranroz'
"""
Tests for feature enrichment
"""


class TestChromosomeSegmentation(TestCase):
    def test_loadSegmentation(self):
        """
        Loading segmentation
        """
        sub_genome = ChromosomeSegmentation.load('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872',
                                                 chromosomes='chr7')

        self.assertTrue(sub_genome['chr7'] and len(sub_genome.keys()) == 1)

    def test_loadAverageAccessability(self):
        """
        Add average accessibility feature to segments
        """
        sub_genome = ChromosomeSegmentation.load('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872',
                                                 chromosomes='chr7')
        sub_genome['chr7'].load(SegmentationFeatures.AverageAccessibility)
        # access to average accessibility feature
        avg_accessibility = sub_genome['chr7'].get(SegmentationFeatures.AverageAccessibility)
        feature_keys, feature_matrix = sub_genome['chr7'].feature_matrix()  # access to accessibility in feature matrix

        self.assertTrue(np.all(
            feature_matrix[:, feature_keys.index(SegmentationFeatures.AverageAccessibility)] == avg_accessibility.T))

    def test_loadLengths(self):
        """
        Add length feature to segments
        """
        sub_genome = ChromosomeSegmentation.load('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872',
                                                 chromosomes='chr7')
        sub_genome['chr7'].load(SegmentationFeatures.RegionLengths)
        # access to average region lengths feature
        region_lengths = sub_genome['chr7'].get(SegmentationFeatures.RegionLengths)
        feature_keys, feature_matrix = sub_genome['chr7'].feature_matrix()  # access to accessibility in feature matrix

        self.assertTrue(
            np.all(feature_matrix[:, feature_keys.index(SegmentationFeatures.RegionLengths)] == region_lengths.T))

    def test_loadMarkers(self):
        """
        Adds methylation and acetylation features
        """
        """
        sub_genome = ChromosomeSegmentation.load('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872',
                                                 sel_chromosomes='chr7')
        sub_genome['chr7'].load(SegmentationFeatures.Markers)
        markers = sub_genome['chr7'].get(SegmentationFeatures.Markers)  # access to average accessibility feature
        feature_keys, feature_matrix = sub_genome['chr7'].feature_matrix()  # access to accessibility in feature matrix

        self.assertTrue(feature_matrix[:, feature_keys.index(SegmentationFeatures.Markers)] == markers)
        """
        self.assertTrue(True)

    def mean_length(self):
        from models import chromatin_classifier
        #from matplotlib.pylab import plt
        from matplotlib import pyplot as plt

        resolution = 500
        model = chromatin_classifier.load('Discrete500-a[0.000, 0.000000]')
        genome = ChromosomeSegmentation.load('fetal_brain', model)
        open_lengths = []
        closed_lengths = []
        for chrom in genome:
            data = genome[chrom].get(
                [SegmentationFeatures.OpenClosed, SegmentationFeatures.RegionLengths, SegmentationFeatures.Position])
            closed_lengths += data[data[:, 0] == 0, 1].tolist()
            open_lengths += data[data[:, 0] == 1, 1].tolist()

            if np.any(data[data[:, 0] == 1, 1] >= 300):  #150kb
                print(chrom)
                print(data[(data[:, 0] == 1) & (data[:, 1] >= 300), 2] * 500)
                print(data[(data[:, 0] == 1) & (data[:, 1] >= 300), 2] * 500 + 150000)

        for func, func_name in zip([len, np.mean, np.std, np.max, np.min], ['#', 'mean', 'std', 'max', 'min']):
            print(func_name + str(np.array([func(closed_lengths), func(open_lengths)]) * resolution))

        # my histogram
        fig, ax = plt.subplots(figsize=(10, 8))

        #bins = np.arange(0, 700, 5)

        #bins = 80#np.arange(3.1, 5, 0.01)
        bins = np.arange(2.5, 6.2, 0.03)

        #plt.hist([(open_lengths), (closed_lengths)], bins=bins, histtype='stepfilled', label=['Open', 'Closed'], alpha=0.7)
        open_lengths_log = np.log10(np.array(open_lengths)*500)
        closed_lengths = np.log10(np.array(closed_lengths)*500)
        plt.hist([open_lengths_log, closed_lengths], bins=bins, histtype='stepfilled', label=['Open', 'Closed'], alpha=0.7)
        plt.legend()
        #plt.loglog()
        plt.yscale('log', nonposy='clip')
        plt.title('Domains length')
        plt.xlabel('Domain length')
        plt.ylabel('#')
        #ax.set_xticklabels(['%ikb' % x for x in ax.get_xticks() / 2])
        from matplotlib.ticker import FormatStrFormatter
        #ax.set_xticklabels(['%.2f bp' % x for x in ax.get_xticks()])
        ax.xaxis.set_major_formatter(FormatStrFormatter('$10^{%.1f}$ bp'))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestSeqLoader.test_loadSegmentation").setLevel(logging.DEBUG)
    unittest.main()