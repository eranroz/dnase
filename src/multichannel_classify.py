"""
Script for multi channel classifications.

see also:
    multichannel_classifier - business logic for multichannel classifiers
"""
import argparse
import logging
import os
import pickle
import datetime

import numpy as np

from config import BED_GRAPH_RESULTS_DIR, MEAN_DNASE_DIR, MODELS_DIR
from data_provider import SeqLoader
from data_provider.data_publisher import publish_dic
from dnase.multichannel_classifier import GMMClassifier, DiscreteMultichannelHMM
from dnase.dnase_classifier import DNaseMultiChannelClassifier
from data_provider import data_publisher


__author__ = 'eranroz'


def multichannel_hmm_discrete(resolution, model_name=None, out_file=None):
    """
    Use multichannel HMM to classify DNase

    the following function implements the discrete approach
    @param out_file: output file name
    @param model_name: name of model to be used
    @param resolution: binning resolution
    """
    default_name = 'discreteMultichannel'
    model_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                              '%s.Discrete%i.model' % (model_name or default_name, resolution))

    min_alpha = 0

    strategy = DiscreteMultichannelHMM()
    classifier = DNaseMultiChannelClassifier(strategy, resolution, model_name)
    #data = LazyChromosomeLoader(lambda x: np.log(strategy.data_transform()(chromosomes=[x], directory=MEAN_DNASE_DIR)[x]+1))
    multichannel_data = classifier.load_data(directory=MEAN_DNASE_DIR)

    if os.path.exists(model_name):
        print('Skipped training - a model already exist')
        with open(model_name, 'rb') as model_file:
            strategy.model = pickle.load(model_file)
    else:
        start_training = datetime.datetime.now()
        classifier.fit([multichannel_data])
        print('Training took', datetime.datetime.now() - start_training)
        with open(model_name, 'wb') as model_file:
            pickle.dump(strategy.model, model_file)

    print('Classifying')
    class_mode = 'viterbi'
    for classified_seq in classifier.classify([multichannel_data]):
        file_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                                 '%s.Discrete%i.%s.bg' % (out_file or default_name, resolution, class_mode))
        npz_file_name = os.path.join(BED_GRAPH_RESULTS_DIR,
                                     '%s.Discrete%i.%s.npz' % (out_file or default_name, resolution, class_mode))
        print('Writing result file (bg format)')
        SeqLoader.build_bedgraph(classified_seq, resolution=resolution, output_file=file_name)
        print('Writing result file (pkl format)')
        SeqLoader.save_result_dict(npz_file_name, classified_seq)
        print('Writing raw file')
        data_publisher.publish_dic(multichannel_data, resolution, '%s.%i.rawDiscrete' % (default_name, resolution))


def multichannel_hmm_continuous(resolution=1000, model_name=None, out_file=None, in_dir=MEAN_DNASE_DIR):
    """
    Use multichannel HMM to classify DNase

    continuous approach (multivariate gaussian mixture model)
    @param out_file: output file name
    @param model_name: name of model
    @param resolution: resolution to learn
    """
    if model_name is None:
        model_name = 'multichannel%i' % resolution

    train_chromosome = 'chr8'
    num_states = 10
    strategy = GMMClassifier()
    model = DNaseMultiChannelClassifier(strategy, resolution, model_name)
    #data = LazyChromosomeLoader(lambda x: np.log(strategy.data_transform()(chromosomes=[x], directory=MEAN_DNASE_DIR)[x]+1))
    data = model.load_data(directory=MEAN_DNASE_DIR)
    strategy.default(data, train_chromosome=train_chromosome, num_states=num_states)
    strategy.training_chr = [train_chromosome]

    model.fit(data)  # pca projection matrix selection + forward-backward
    classification = model.classify(data)  # viterbi

    # save
    if not os.path.exists(os.path.join(MODELS_DIR, model_name)):
        os.makedirs(os.path.join(MODELS_DIR, model_name))
    SeqLoader.save_result_dict(os.path.join(MODELS_DIR, model_name, out_file or "segmentation"), classification)
    publish_dic(classification,
                resolution,
                '%sSegmentome' % model_name,
                short_label="Mutlicell-PCA %i" % num_states,
                long_label="%i states, multi-cell DNase HMM GMM" % num_states)


def raw_find_variable_regions():
    """
    Simple function to locate regions that behave differently in cell types
    based only on raw data
    """
    chrom = 'chr6'
    resolution = 10000
    chrom_data = load_multichannel(resolution, [chrom])[chrom]
    chrom_data.data = np.log(chrom_data.data + 1)
    chrom_data = np.array(chrom_data.todense())
    variance = np.var(chrom_data, 0)
    max_var = np.argsort(variance)[::-1]
    for i in max_var[:10]:
        print('%s: %i-%i' % (chrom, i * resolution - 10000, i * resolution + 10000))
    print(max_var[:10] * resolution)
    print('-----')


if __name__ == '__main__':
    commands = {
        'multichannel_discrete': multichannel_hmm_discrete,
        'multichannel_continuous': multichannel_hmm_continuous
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help="command to execute: %s" % (', '.join(list(commands.keys()))))
    parser.add_argument('--model', help="model file to be used")
    parser.add_argument('--resolution', help="resolution to use for classification", type=int, default=1000)
    parser.add_argument('--output', help="Output file prefix", default=None)

    parser.add_argument('--no-verbose', dest='verbose', action='store_false')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    if args.command.startswith('multichannel_'):
        commands[args.command](args.resolution, model_name=args.model, out_file=args.output)
    else:
        commands[args.command]()