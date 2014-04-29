"""
This script uses DNase data to classify open and closed regions in chromosomes.
E.g - it creates segmentation of the genome
"""
import os
import datetime

from config import BED_GRAPH_RESULTS_DIR
import data_provider.DiscreteTransformer
from dnase import dnase_classifier
from dnase.HMMClassifier import HMMClassifier
from dnase.dnase_classifier import DNaseClassifier
from hmm.HMMModel import DiscreteHMM, ContinuousHMM


__author__ = 'eran'

from data_provider import SeqLoader
import numpy as np
import pickle
import argparse

resolution = 20
min_alpha = None


def classify_continuous(in_file='UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872', output_p=False,
                        model_name=None, out_file=None):
    """
    Classifies genome to open and closed regions based on gaussian hmm
    @param in_file: name of sample (according to filename in data directory, without resolution and extension)
    @param output_p: false for viterbi, true for posterior
    @param model_name: name of model to be used or none to train a new model
    """
    state_transition = np.array(
        [
            [0.0, 0.9, 0.1],  # begin
            [0.7, 0.99, 0.01],  # closed (very small change to get to open)
            [0.3, 0.1, 0.9]  # open (may go to close but prefers to keep the state)
        ]
    )
    emission = np.array([
        [0, 1],
        [0, 1],  # closed - guess mean almost 0
        [2, 2.5]  # open - more variable
    ])

    print('Creating model')
    model = ContinuousHMM(state_transition, emission, min_alpha=min_alpha)
    strategy = HMMClassifier(model)
    strategy.output_p = output_p
    #strategy.trainChm = ['chr1']
    classifier = DNaseClassifier(strategy)

    print('Loading data')
    training = SeqLoader.load_dict(in_file, resolution,
                                   transform=lambda x: np.log(np.array(x) + 1))
    print('Training model')
    # save the trained model
    model_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                              '%s.Continuous%s.model' % (model_name or in_file, str(resolution)))
    # do we have to train the model?
    if os.path.exists(model_name):
        print('Skipped training - a model already exist')
        with open(model_name, 'rb') as model_file:
            strategy.model = pickle.load(model_file)
    else:
        start_training = datetime.datetime.now()
        classifier.fit([training])
        print('Training took', datetime.datetime.now() - start_training)
        with open(model_name, 'wb') as model_file:
            pickle.dump(strategy.model, model_file)

    print('Classifying')
    class_mode = 'posterior' if output_p else 'viterbi'
    for classified_seq in classifier.classify([training]):
        file_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                                 '%s.Continous%i.%s.bg' % (out_file or in_file, resolution, class_mode))
        npz_filename = os.path.join(BED_GRAPH_RESULTS_DIR,
                                    '%s.Continous%i.%s.npz' % (out_file or in_file, resolution, class_mode))
        raw_file = os.path.join(BED_GRAPH_RESULTS_DIR,
                                '%s.%i.rawContinous.bg' % (in_file, resolution))
        print('Writing result file (bg format)')
        SeqLoader.build_bedgraph(classified_seq, resolution=resolution, output_file=file_name)
        print('Writing result file (pkl format)')
        SeqLoader.save_result_dict(npz_filename, classified_seq)
        print('Writing raw file')
        SeqLoader.build_bedgraph(training, resolution=resolution, output_file=raw_file)


def classify_discrete(in_file='UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872', output_p=False,
                      model_name=None, out_file=None):
    """
    Classifies genome to open and closed regions based on discrete classic hmm
    @param in_file: name of sample (according to filename in data directory, without resolution and extension)
    @param output_p: false for viterbi, true for posterior
    @param model_name: name of model to be used or none to train a new model
    """
    state_transition = np.array(
        [
            [0.0, 0.9, 0.1],  # begin
            [0.7, 0.99, 0.01],  # closed (very small change to get to open)
            [0.3, 0.1, 0.9],  # open (may go to close but prefers to keep the state)
        ]
    )
    emission = np.array([
        np.zeros(4),
        [0.8, 0.1, 0.09, 0.01],  # closed - prefers low values
        [0.02, 0.4, 0.5, 0.08]  # open - prefers high values
    ])

    print('Loading data')
    training = SeqLoader.load_dict(in_file, resolution,
                                   data_provider.DiscreteTransformer())
    print('Creating model')

    model = DiscreteHMM(state_transition, emission, min_alpha=min_alpha)
    strategy = HMMClassifier(model)
    strategy.output_p = output_p
    #strategy.trainChm = ['chr1']
    classifier = dnase_classifier.DNaseClassifier(strategy)
    print('Training model')
    model_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                              '%s.Discrete%s.model' % (model_name or in_file, str(resolution)))

    if os.path.exists(model_name):
        print('Skipped training - a model already exist')
        with open(model_name, 'rb') as model_file:
            strategy.model = pickle.load(model_file)
    else:
        start_training = datetime.datetime.now()
        classifier.fit([training])
        print('Training took', datetime.datetime.now() - start_training)
        with open(model_name, 'wb') as model_file:
            pickle.dump(strategy.model, model_file)

    print('Classifying')
    class_mode = 'posterior' if output_p else 'viterbi'
    for classified_seq in classifier.classify([training]):
        file_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                                 '%s.Discrete%s.%s.bg' % (out_file or in_file, str(resolution), class_mode))
        npz_file_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                                     '%s.Discrete%s.%s.npz' % (out_file or in_file, str(resolution), class_mode))
        raw_file = os.path.join(BED_GRAPH_RESULTS_DIR,
                                '%s.%s.rawDiscrete.bg' % (in_file, str(resolution)))
        print('Writing result file (bg format)')
        SeqLoader.build_bedgraph(classified_seq, resolution=resolution, output_file=file_name)
        print('Writing result file (pkl format)')
        SeqLoader.save_result_dict(npz_file_name, classified_seq)
        print('Writing raw file')

        SeqLoader.build_bedgraph(training, resolution=resolution, output_file=raw_file)


def run_simple_classification():
    """
    to be used to classify from code. prefer to use the _main_ below
    @return:
    """
    global resolution
    #import cProfile
    #cProfile.run('transformProfile()')
    #classify()
    #cProfile.run('classifyContinous()')
    classify_continuous(output_p=True)
    classify_discrete(output_p=True)
    resolution = 100
    print('Classify with 100 resolution')
    classify_continuous(output_p=True)
    classify_discrete(output_p=True)


if __name__ == "__main__":
    """
    Example:
    python3 -m dnase_classify --posterior --resolution 1000 UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help="input file, included in the data directory. with no file and resolution" +
                                       " extensions (suffixes)")
    parser.add_argument('--posterior', help="use this option to output posterior probabilities instead of states",
                        action="store_true")
    parser.add_argument('--resolution', help="resolution to use for classification", type=int, default=500)
    parser.add_argument('--model', help="model file to be used")
    parser.add_argument('--model_type', help="Model type: discrete (d) or continuous (c)", type=str,
                        default='discrete')
    parser.add_argument('--min_alpha', help="Prior for transition probabilities", type=float, default=0)
    parser.add_argument('--min_alpha_open', help="Prior for transition probabilities", type=float, default=None)
    parser.add_argument('--min_alpha_closed', help="Prior for transition probabilities", type=float, default=None)
    parser.add_argument('--output', help="Output file prefix", default=None)
    args = parser.parse_args()
    print('Args')
    print(args)

    resolution = args.resolution

    if args.min_alpha is not None or (args.min_alpha_open is not None and args.min_alpha_closed is not None):
        min_alpha = np.array([
            args.min_alpha_closed if args.min_alpha_closed is not None else args.min_alpha,
            args.min_alpha_open if args.min_alpha_open is not None else args.min_alpha
        ])

    model_name = args.model
    if args.infile.endswith('.20.pkl') or args.infile.endswith('.20.npz'):
        print('Wartning: input file includes file and resolution extensions')

    output_path = args.output or os.path.basename(args.infile)
    if model_name is None or not dnase_classifier.model_exist(model_name):
        is_discrete = args.model_type[0].upper() == 'D'
        if model_name is None:
            model_name = '%s%i-a[%.3f, %3f]' % ('Discrete' if is_discrete else 'Continuous', resolution, min_alpha[0], min_alpha[1])
        strategy = HMMClassifier.default(is_discrete, min_alpha)
        strategy.output_p = args.posterior  # viterbi or posterior
        classifier = dnase_classifier.DNaseClassifier(strategy, resolution, model_name)
        classifier.fit_file(args.infile)
        classifier.save()  # create directory for the model
    else:
        classifier = dnase_classifier.load(model_name)
    classifier.save_classify_file(args.infile, output_path)

    #classify_continuous(in_file=args.infile, output_p=args.posterior, model_name=args.model, out_file=args.output)
    #classify_discrete(in_file=args.infile, output_p=args.posterior, model_name=args.model, out_file=args.output)
    print('Finished')