# coding=utf-8
"""
This script go over different models (grid search) and tries to find the best model according to score function.
Since there is no "correct" score function, we don't use any heuristic to maximize/minimize them

HMM model parameters are as follows:
* alpha - transition probabilities constraints
* bin size - size of bins (20, 100 etc)
* discretization values (TODO - add to grid search)


The possible score functions are as follows:
* model likelihood - the likelihood given for a model
* correlation with "golden model" - agreement with other models
* based on segmented genes (penalty for segmented gene) - see L{dnase.model_selection.SegmentedGenesScorer}
* based on CTCF insulator segmentation of the genome (based on ENCODE, mammary epithelial cells, brest tissue)
* based on H3k27ac enrichment for open regions

"""
import os
import argparse
import numpy as np
from config import PUBLISH_DIR, DATA_DIR
from data_provider.DiscreteTransformer import DiscreteTransformer
from dnase.HMMClassifier import HMMClassifier
from dnase.model_selection.ConsistencyScorer import ConsistencyScorer
from dnase.model_selection.BoundaryEnrichment import CTCFModelScore
from dnase.model_selection.MarkerEnrichmentScorer import MarkerEnrichmentScorer
from hmm.HMMModel import DiscreteHMM, ContinuousHMM
from hmm.bwiter import bw_iter, IteratorCondition
from hmm.hmt import dwt, HMTModel
from data_provider import SeqLoader
from config import RES_DIR


__author__ = 'eranroz'

TRAIN_CHROMOSOME = 'chr6'  # chromosome used for baum welch training
SCORE_CHROMOSOME = 'chr7'  # chromosome used for getting score

#training_file = 'UW.Breast_vHMEC.ChromatinAccessibility.RM035.DS18406'  # HMEC - breast

#score_file_dir = 'ncbi_data/muscle_fetal'
#TRANING_DIR = 'data/muscle_fetal_dnase'
#training_file = os.path.join(TRANING_DIR, os.listdir(TRANING_DIR)[0])
# model parameters
_RESOLUTIONS = [5000, 1000, 500, 200]
_ALPHAS_OPENED = [0, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]
_MODEL_TYPES = ['Discrete', 'Continuous']


class LikelihoodScore:
    """
    Score based on likelihood (EM)
    """

    @staticmethod
    def score(model, data):
        """
        scores according to maximum probability we get after EM iterations
        @param model: initial model for training
        @param data: data to fit to
        @return:log likelihood
        """
        _, p = bw_iter(data[TRAIN_CHROMOSOME], model, IteratorCondition(10))
        return p


def score_hmt(training_dir, ctcf_scorer, genes_scorer):
    """
    Calculate scores for HMT for different resolutions
    The segmentation based on resolutions is just choosing level in the trained tree
    @param genes_scorer: scorer of L{SegmentedGenesScorer}
    @param ctcf_scorer: scorer of L{CTCFModelScore}
    @param training_dir: directory with Dnase signal of different samples
    """
    from scipy.signal import daub, qmf

    training_file = os.listdir(training_dir)[0].replace('.20.npz', '')
    wavelet_data = SeqLoader.load_dict(training_file, 20, transform=SeqLoader.continuous_transform,
                                       chromosome=[TRAIN_CHROMOSOME, SCORE_CHROMOSOME], directory=training_dir)
    h_filter = daub(1)
    g_filter = qmf(h_filter)
    orig_len = len(wavelet_data[TRAIN_CHROMOSOME])
    new_len = 2 ** np.floor(np.log2(orig_len))
    scaling_coefficients, wavelets_coefficients = dwt(wavelet_data[TRAIN_CHROMOSOME][0:new_len], h_filter, g_filter)
    wavelet_data[TRAIN_CHROMOSOME] = wavelets_coefficients
    orig_lengths = dict()
    for k, v in wavelet_data.items():
        if k != TRAIN_CHROMOSOME:
            orig_len = len(v)
            new_len = 2 ** np.ceil(np.log2(orig_len))
            signal = np.pad(v, [0, new_len - orig_len], 'constant')  # pad with zeros
            scaling_coefficients, wavelets_coefficients = dwt(signal, h_filter, g_filter)
            wavelet_data[k] = wavelets_coefficients
            orig_lengths[k] = orig_len

    # run it:
    model = HMTModel(np.array([
        [0, 0.8, 0.2],  # root
        [0, 0.6, 0.4],  # closed
        [0, 0.2, 0.8],  # open
    ]), np.array([
        (0, 2),  # closed
        (1, 4),  # open - more variance
    ]))
    model.min_alpha = 0.01  # constraint on minimum transition so open state exist
    strategy = HMMClassifier(model)
    strategy.training_chr([TRAIN_CHROMOSOME])
    strategy.output_p = False  # output viterbi and not posterior in this case

    p_score = strategy.fit([wavelet_data])  # fit the classifier to train_chromosome
    viterbi_tree = strategy.classify({SCORE_CHROMOSOME: wavelet_data[SCORE_CHROMOSOME]})
    viterbi_tree = viterbi_tree[SCORE_CHROMOSOME]
    # select resolution
    score_matrix = np.zeros((8, 3))

    for level, score_iter in zip(np.arange(1, 9), score_matrix):
        segmentation = viterbi_tree[-level] == 1
        res = 20 * np.power(2, level)
        ctcf_score = ctcf_scorer.score(segmentation, res)  # positive - number of segments of near length and positions
        genes_score = genes_scorer.score(segmentation, res)  # negative - loose function
        score_iter[...] = [res, ctcf_score, genes_score]

    # print score:
    print('Res\tCTCF\tGenes')
    for res, ctcf_score, genes_penality in score_matrix:
        print('%i\t%i\t%i' % (res, ctcf_score, genes_penality))

    res = 160
    classified_seq = {SCORE_CHROMOSOME: viterbi_tree[-4][0:orig_lengths[SCORE_CHROMOSOME] / (res / 20)]}
    BED_GRAPH_RESULTS_DIR = os.path.join(RES_DIR, 'open-closed', 'data')
    res_file = os.path.join(BED_GRAPH_RESULTS_DIR, '%s.hmt.haar.%i.bg' % (training_file, res))
    SeqLoader.build_bedgraph(classified_seq, resolution=res, output_file=res_file)

    print('Finished printing')


def eval_res(training_dir, res, score_functions=None):
    """
    Evaluates model in specific resolution with different parameters and constraints
    @param training_dir: directory with Dnase signal samples from same cell type
    @param res:
    @param score_functions: score functions according to which to evaluate the model
    @return:
    """
    global _MODEL_TYPES, _ALPHAS_OPENED, SCORE_CHROMOSOME
    if score_functions is None:
        score_functions = [MarkerEnrichmentScorer(SCORE_CHROMOSOME), ConsistencyScorer]

    def create_segmentation(scores_data, segmentation_strategy):
        """
        Creates segmentation with same model for all files in training_dir
        @param model_type_transform: model type, according to which we load the data
        @param segmentation_strategy: strategy used for creating segmentation
        """
        for seq in scores_data:
            segmented = segmentation_strategy.classify({SCORE_CHROMOSOME: seq[SCORE_CHROMOSOME]})
            segmented = segmented[SCORE_CHROMOSOME] == 1  # change to true/false matrix
            yield segmented

    training_file = os.listdir(training_dir)[0].replace('.20.npz', '')
    discrete_data = SeqLoader.load_dict(training_file, res, DiscreteTransformer(),
                                        chromosome=[TRAIN_CHROMOSOME], directory=training_dir)
    continuous_data = SeqLoader.load_dict(training_file, res, transform=SeqLoader.continuous_transform,
                                          chromosome=[TRAIN_CHROMOSOME], directory=training_dir)
    #score_types = ['p', 'ctcf', 'genes']

    score_matrix = np.zeros((len(_MODEL_TYPES), len(_ALPHAS_OPENED), len(score_functions) + 1))

    for model_type in _MODEL_TYPES:
        score_files = [infile.replace('.20.npz', '') for infile in os.listdir(training_dir) if
                       infile.endswith('.20.npz') and infile not in training_file]
        scores_data = []
        for infile in score_files:
            if model_type == 'Continuous':
                infile_data = SeqLoader.load_dict(infile, res, transform=SeqLoader.continuous_transform,
                                                  chromosome=[SCORE_CHROMOSOME], directory=training_dir)
            elif model_type == 'Discrete':
                infile_data = SeqLoader.load_dict(infile, res, DiscreteTransformer(),
                                                  chromosome=[SCORE_CHROMOSOME], directory=training_dir)
            scores_data.append(infile_data)

        for alpha_opened_i, alpha_opened in enumerate(_ALPHAS_OPENED):
            alpha_closed = 0
            min_alpha = np.array([alpha_closed, alpha_opened])
            if model_type in ['Discrete', 'Continuous']:
                state_transition = np.array(
                    [
                        [0.0, 0.9, 0.1],  # begin
                        # closed (very small change to get to open)
                        [0.7, max(alpha_closed, 0.99), 1 - max(alpha_closed, 0.99)],
                        # open (may go to close but prefers to keep the state)
                        [0.3, 1 - max(0.9, alpha_opened), max(0.9, alpha_opened)],
                    ]
                )
            if model_type == 'Discrete':
                emission = np.array([
                    np.zeros(4),
                    [0.8, 0.1, 0.09, 0.01],  # closed - prefers low values
                    [0.02, 0.4, 0.5, 0.08]  # open - prefers high values
                ])

                model = DiscreteHMM(state_transition, emission, min_alpha=min_alpha)
                data = discrete_data
            elif model_type == 'Continuous':
                emission = np.array([
                    [0, 1],
                    [0, 1],  # closed - guess mean almost 0
                    [2, 2.5]  # open - more variable
                ])
                model = ContinuousHMM(state_transition, emission, min_alpha=min_alpha)
                data = continuous_data

            strategy = HMMClassifier(model)
            strategy.training_chr([TRAIN_CHROMOSOME])
            strategy.output_p = False  # output viterbi and not posterior in this case

            p_score = strategy.fit([data], 5)  # fit the classifier to train_chromosome

            print('Model: %s\tMin alpha: %s' % (model_type, str(min_alpha)))
            segmentations = list(create_segmentation(scores_data, strategy))
            additional_scores = [score_func.score(training_set=segmentations, resolution=res, model=model)
                                 for score_func in score_functions]
            score_matrix[_MODEL_TYPES.index(model_type), alpha_opened_i, :] = [p_score] + additional_scores

    return score_matrix


def grid_search_models(train_cell_type, use_multiprocessing=False):
    """
    Grid search for good parameters for segmentation
    @param train_cell_type: training cell type
    @param use_multiprocessing: whether to use use multiprocessing
    """
    global _RESOLUTIONS, _MODEL_TYPES, _ALPHAS_OPENED

    acetylationH2k27 = MarkerEnrichmentScorer(train_cell_type, SCORE_CHROMOSOME, signal_type="H3K27ac")
    h3K27Me3 = MarkerEnrichmentScorer(train_cell_type, SCORE_CHROMOSOME, signal_type="H3K27me3")
    h3k36me3 = MarkerEnrichmentScorer(train_cell_type, SCORE_CHROMOSOME, signal_type="H3K36me3")

    consistent_scorer = ConsistencyScorer()
    score_functions = [acetylationH2k27, h3K27Me3, h3k36me3, consistent_scorer]
    score_types = ['p', 'k27ac_enrichment', 'H3K27me3 enrichment', 'H3K36me3 enrichment', 'consistent']
    scores_matrix = np.zeros((len(_RESOLUTIONS), len(_MODEL_TYPES), len(_ALPHAS_OPENED), len(score_types)))
    train_dir = os.path.join(DATA_DIR, train_cell_type)
    if use_multiprocessing:
        from multiprocessing import Pool
        from functools import partial

        eval_res_core = partial(eval_res, training_dir=train_dir, score_functions=score_functions)
        pool_process = Pool()
        for res_i, res in enumerate(pool_process.map(eval_res_core, _RESOLUTIONS)):
            scores_matrix[res_i, :] = res
    else:
        for res_i, res in enumerate(_RESOLUTIONS):
            print('res: %i' % res)
            scores_matrix[res_i, :] = eval_res(training_dir=train_dir, res=res,
                                               score_functions=score_functions)

    print('Finished')
    # p, k27 average, consistency
    np.save(os.path.join(RES_DIR, 'modelEvaluation', 'gridSearchScoreConsistency-%s' % (','.join(score_types))),
            scores_matrix)


def printCTCF():
    scorer = CTCFModelScore(20)
    gold = scorer.golden_model
    with open(os.path.join(PUBLISH_DIR, "ctcfSeg.bg"), 'w') as ctcfSeg:
        #print("track type=bedGraph", file=ctcfSeg)
        val = 1
        for seg in gold:
            print("%s\t%i %i %i" % (SCORE_CHROMOSOME, seg[0], seg[0] + seg[1], val), file=ctcfSeg)
            val = (val + 1) % 2
    print('Finished')
    #SeqLoader.build_bedgraph(classified_seq, resolution=20, output_file=file_name)


def showGrid():
    scores_matrix = np.load(os.path.join(RES_DIR, 'modelEvaluation', 'gridSearchScore.npy'))
    score_types = ['p', 'ctcf', 'genes']
    np.set_printoptions(precision=3)
    for res_i, res in enumerate(_RESOLUTIONS):
        #scores_matrix = np.zeros((len(_MODEL_TYPES), len(_ALPHAS_OPENED), len(score_types)))

        print('Resolution: %i' % res)
        discrete = scores_matrix[res_i, 0, :]
        continous = scores_matrix[res_i, 1, :]

        print('ALPHA/scores\tP\tCTCF\tgenes')
        print('discrete')
        print(np.column_stack([_ALPHAS_OPENED, discrete]))
        print('continous')
        print(np.column_stack([_ALPHAS_OPENED, continous]))
        #continous = scores_matrix[res_i, 0, :]


def show_grid_search():
    global _RESOLUTIONS, _MODEL_TYPES, _ALPHAS_OPENED
    import matplotlib.pyplot as plt

    score_types = ['p', 'mean_k27_enrichment', 'consistent']
    scores_matrix = np.load(os.path.join(RES_DIR, 'modelEvaluation', 'gridSearchScoreConsistency.npy'))

    fig, ax = plt.subplots()
    for model_i, model in enumerate(_MODEL_TYPES):
        data = []
        labels = []
        for res_i, res in enumerate(_RESOLUTIONS):
            for alpha_i, alpha in enumerate(_ALPHAS_OPENED):
                data.append(scores_matrix[res_i, model_i, alpha_i, 1:])
                labels.append('%i-α%f' % (res, alpha))
        data = np.array(data)

        ax.scatter(data[:, 0], data[:, 1], label=model, c='rb'[model_i])
        for i, txt in enumerate(labels):
            ax.annotate(txt, data[i, :], rotation=15 * (-1 if model_i == 0 else 1), size=10)
    plt.legend()
    plt.xlabel('k27Enrichment')
    plt.ylabel('Consistent')
    plt.title('Model evaluation for different models')
    resolutions = ','.join([str(s) for s in _RESOLUTIONS])
    alphas = ','.join([str(s) for s in _ALPHAS_OPENED])
    plt.figtext(0.4, 0.02, 'Resolutions: %s\nAlphas: %s' % (resolutions, alphas))
    plt.show()


def default_training_cell_type():
    """
    Get default training cell type - e.g cell type with most number of samples
    @return: cell type to train according to
    """
    candidates = []
    for cell_type in os.listdir(DATA_DIR):
        candidate_dir = os.path.join(DATA_DIR, cell_type)
        if os.path.isdir(candidate_dir):
            candidates.append((cell_type, len(os.listdir(candidate_dir))))

    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def profile_model_select():
    grid_search_models(default_training_cell_type(), False)


if __name__ == "__main__":
    #import cProfile
    #cProfile.run('profile_model_select()')

    commands = {
        'gridSearch': grid_search_models,
        'showGrid': show_grid_search
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('command', help="command to execute: %s" % (', '.join(list(commands.keys()))))
    parser.add_argument('--trainCellType', help="Directory of DNase data with more or less similar samples",
                        default=default_training_cell_type())
    parser.add_argument('--multicore', help="Whether to use multiprocessing", action="store_true")
    args = parser.parse_args()
    if args.command == 'gridSearch':
        grid_search_models(args.trainCellType, args.multicore)
    else:
        commands[args.command]()

