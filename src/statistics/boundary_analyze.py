"""
This script analyzes the boundary of regions
"""
import argparse
import os
import numpy as np
from config import MEAN_MARKERS, MEAN_DNASE_DIR
from data_provider import SeqLoader
from models import chromatin_classifier
from models import ChromosomeSegmentation
from models.ChromosomeSegmentation import SegmentationFeatures

__author__ = 'eranroz'


def _load_markers(model, chromosome, cell_type):
    if not os.path.exists(model.segmentation_file_path()):
        raise Exception("Segmentation doesn't exist." % model.segmentation_file_path())

    segmentation_cell_type = ChromosomeSegmentation.load(cell_type, model, chromosomes=[chromosome])
    segmentation_cell_type = segmentation_cell_type[chromosome]
    data = segmentation_cell_type.get([SegmentationFeatures.OpenClosed, SegmentationFeatures.Markers])
    labels = segmentation_cell_type.get_labels([SegmentationFeatures.OpenClosed, SegmentationFeatures.Markers])
    return data, labels


def _load_regions(model, cell_type, chromosome=None):
    if not os.path.exists(model.segmentation_file_path()):
        raise Exception("Segmentation doesn't exist." % model.segmentation_file_path())

    segmentation_cell_type = ChromosomeSegmentation.load(cell_type, model,
                                                         chromosomes=chromosome if chromosome is None else [chromosome])
    segmentation_cell_type = segmentation_cell_type[chromosome]
    data = segmentation_cell_type.get(
        [SegmentationFeatures.Position, SegmentationFeatures.RegionLengths, SegmentationFeatures.OpenClosed])
    return data


def interpolate_region(region_signal):
    """
    Calculates the signal in region with resizing region to 100
    @param region_signal: signal within region

    @description: since different regions have different sizes, we fix them to 100 and interpolate their sizes
    """
    """
    from scipy.interpolate import interp1d

    inter = interp1d(np.arange(region_signal.shape[0]), region_signal, bounds_error=False)  # , 'cubic' or linear?
    return inter(np.arange(100))
    """
    before = region_signal[np.array(np.floor(np.arange(100) * region_signal.shape[0] / 100), dtype=int)]
    delta = region_signal.shape[0] / 100 - np.floor(region_signal.shape[0] / 100)
    after = region_signal[np.minimum(np.array(np.ceil(np.arange(100) * region_signal.shape[0] / 100), dtype=int),
                                     region_signal.shape[0] - 1)]
    return before + delta * (after - before)


def heatmap_boundary(model_name):
    """
    Creates heatmaps for the boundary of regions
    :param model_name:
    """
    import matplotlib

    # matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    model = chromatin_classifier.load(model_name)
    #cell_type = 'brain_fetal'
    cell_type = 'IMR90_cell_line'
    #cell_type = 'pancreas'
    #/cs/stud/eranroz/dnase/data/data/heart_fetal/GSM530654_UW.Fetal_Heart.ChromatinAccessibility.H-22662.DS12531.20.pkl
    from config import SIGNAL_DIR
    #segmentation = next(model.classify_file(os.path.join(NCBI_DIR, 'heart_fetal', 'GSM530654_UW.Fetal_Heart.ChromatinAccessibility.H-22662.DS12531.20.pkl')))
    segmentation = next(model.classify_file(os.path.join(MEAN_DNASE_DIR, '%s.mean.npz' % cell_type)))
    #segmentation = None  # for debug
    #chromosome = 'chr7'
    model_resolution = model.resolution
    orig_resolution = 20
    interesting_area = 10000  # one side

    interesting_area /= orig_resolution  # use same resolution as orig resolution
    # start, length, openclosed
    regions_data_dict = dict()
    segmentation_cell_type = ChromosomeSegmentation.load(cell_type, model, segmentation=segmentation)
    for chromosome, segmentation_cell_type in segmentation_cell_type.items():
        regions_data = segmentation_cell_type.get(
            [SegmentationFeatures.Position, SegmentationFeatures.RegionLengths, SegmentationFeatures.OpenClosed])

        regions_data[:, [0, 1]] *= model_resolution / orig_resolution  # translate to resolution of signals (20bp)
        regions_data[:, 1] += regions_data[:, 0]  # replace length by end position
        if chromosome != 'chrM':
        #if chromosome == 'chr12':  # for debug
            regions_data_dict[chromosome] = regions_data

    # load experiments
    experiments = list(SeqLoader.available_experiments(cell_type).keys())
    #experiments = experiments[:3]  # for debug
    open_region = 1
    closed_region = 0
    states = ["open"]  #, "closed"
    states_codes = [open_region]  # , ]

    ex_to_state_to_freq = dict()
    # for debug only open regions
    for region_type in [open_region, closed_region]:  #
        ex_to_state_to_freq[region_type] = dict()

    np.seterr(all='raise')
    n_chromosomes = len(regions_data_dict.keys())
    experiments = experiments + ['DNase']
    for ex_i, ex in enumerate(experiments):
        # experiment data
        if ex != 'DNase':
            experiment_data_all_chrom, _ = SeqLoader.load_experiments(cell_type, [ex],
                                                                      chromosomes=regions_data_dict.keys())
        else:
            experiment_data_all_chrom = SeqLoader.load_result_dict(
                os.path.join(MEAN_DNASE_DIR, '%s.mean.npz' % cell_type))
        print('%s %i/%i' % (ex, ex_i + 1, len(experiments)))
        for region_type in states_codes:
            inside = np.zeros((n_chromosomes, 100))
            left_side = np.zeros((n_chromosomes, interesting_area))
            right_side = np.zeros((n_chromosomes, interesting_area))
            chromosomes_w = np.zeros(n_chromosomes)
            chrom_i = 0
            for chrom, regions_data in regions_data_dict.items():
                if ex == 'DNase':
                    experiment_data = experiment_data_all_chrom[chrom]
                else:
                    experiment_data = np.array(experiment_data_all_chrom[chrom])[0, :]

                # extract position and length from regions
                region_data_subtype = regions_data[regions_data[:, 2] == region_type, 0:2]
                # limit by the size of the expirment data
                region_data_subtype = region_data_subtype[region_data_subtype[:, 0] < experiment_data.shape[0], :]

                region_left = np.column_stack([region_data_subtype[:, 0] - interesting_area,
                                               region_data_subtype[:, 0]])
                region_left = region_left[region_left[:, 0] > 0, :]

                region_right = np.column_stack([region_data_subtype[:, 1],
                                                region_data_subtype[:, 1] + interesting_area])
                region_right = region_right[region_right[:, 1] < experiment_data.shape[0], :]
                inside[chrom_i, :] = np.average(
                    [interpolate_region(experiment_data[reg[0]:reg[1]]) for reg in region_data_subtype], 0)
                left_side[chrom_i, :] = np.average([experiment_data[reg[0]:reg[1]] for reg in region_left], 0)
                right_side[chrom_i, :] = np.average([experiment_data[reg[0]:reg[1]] for reg in region_right[:-1, :]], 0)
                chromosomes_w[chrom_i] = region_data_subtype.shape[0]
                chrom_i += 1

            chromosomes_w /= np.sum(chromosomes_w)
            inside = np.sum(inside * chromosomes_w.T[:, None], 0)
            left_side = np.sum(left_side * chromosomes_w.T[:, None], 0)
            right_side = np.sum(right_side * chromosomes_w.T[:, None], 0)

            ex_to_state_to_freq[region_type][ex] = {
                'inside': inside,
                'left_side': left_side,
                'right_side': right_side
            }

    # visualizations
    if not os.path.exists(model.model_dir("enrichments")):
        os.makedirs(model.model_dir("enrichments"))



    #experiments = (np.array(experiments)[experiments_to_show]).tolist()
    for plt_i in [0, 1, 2]:
        if plt_i == 0:
            # all
            experiments_to_show = experiments
            experiments_to_show.remove('DNase')
            experiments_to_show.sort()
            experiments_to_show.insert(0, 'DNase')
        elif plt_i == 1:
            # only dnase
            experiments_to_show = ['DNase']
        elif plt_i == 2:
            # only the most important
            experiments_to_show = ['DNase', 'H3K27me3', 'H3K36me3', 'H3K9ac', 'H3K4me3', 'H3K9me3']
            experiments_to_show = [ex for ex in experiments_to_show if ex in experiments]


        for state_i, state in zip(states_codes, states):
            plot_lines = []

            #print(experiments_to_show)
            #experiments_to_show = np.array(experiments)[np.where((experiments_to_show > 1.20) | (experiments_to_show < 0.8))[0]]
            #plt.figure()
            #plt.suptitle('%s enrichment' % ex)
            f, axarr = plt.subplots(1, 3, sharey=True, figsize=(16, 5))
            plt.subplots_adjust(wspace=0, top=0.85, left=0.03, right=0.98)

            #f.set_size_inches(20.5, 1.5)

            axLeft, axInside, axRight = axarr
            ax = axLeft
            if plt_i == 0:
                markers = ['.', 'o', 'v', 'x', 'd']
                linewidth = 0
            else:
                markers = [None]
                linewidth = 2
            if len(plot_lines) == 0:
                for ex_i, ex in enumerate(experiments_to_show):
                    #for state_i, state in zip(states_codes, states):
                    plot_lines.append(
                        ax.plot(np.arange(interesting_area), ex_to_state_to_freq[state_i][ex]['left_side'],
                                linewidth=linewidth, marker=markers[ex_i % len(markers)])[0])
            else:
                for ex in experiments_to_show:
                    #for state_i, state in zip(states_codes, states):
                    ax.plot(np.arange(interesting_area), ex_to_state_to_freq[state_i][ex]['left_side'])

            ax.set_xticklabels(
                ['%ikb' % (x / 1000) for x in np.arange(-interesting_area, 0, 100, dtype=int) * orig_resolution])
            ax.set_xticks(np.arange(0, interesting_area, 100))
            #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax = axInside
            for ex_i, ex in enumerate(experiments_to_show):
                ax.plot(np.arange(100), ex_to_state_to_freq[state_i][ex]['inside'], label=state, linewidth=linewidth,
                        marker=markers[ex_i % len(markers)])
            ax.set_xticks(np.arange(0, 100, 20))
            ax.set_xticklabels(['%i%%' % x for x in np.arange(0, 100, 20)])

            ax = axRight
            for ex_i, ex in enumerate(experiments_to_show):
                ax.plot(np.arange(interesting_area), ex_to_state_to_freq[state_i][ex]['right_side'], label=state,
                        linewidth=linewidth, marker=markers[ex_i % (1 if plt_i == 0 else len(markers))])
            ax.set_xticklabels(
                ['%ikb' % (x / 1000) for x in np.arange(0, interesting_area, 100, dtype=int) * orig_resolution])
            ax.set_xticks(np.arange(0, interesting_area, 100))
            f.legend(plot_lines, experiments_to_show, loc='upper center', ncol=8)#, fontsize='small' if len(experiments_to_show)>4 else 'medium')
            #plt.tight_layout()
            if plt_i == 2:
                plt.savefig(model.model_dir('enrichments/%sEnrichment-%s.png' % (state, cell_type)))

    #plt.show()
    print('finished')

if __name__ == "__main__":
    commands = {
        'markersHeatmap': heatmap_boundary
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help="command to execute: %s" % (', '.join(list(commands.keys()))))
    parser.add_argument('--model', help="model used for segmentation")

    args = parser.parse_args()
    commands[args.command](args.model)
