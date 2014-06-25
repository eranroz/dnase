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
from data_provider.LazyLoader import LazyChromosomeLoader
from data_provider.data_publisher import publish_dic, create_description_html
from dnase.PcaTransformer import PcaTransformer
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

    strategy = DiscreteMultichannelHMM()
    classifier = DNaseMultiChannelClassifier(strategy, resolution, model_name)
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
    @param in_dir: directory of data to learn (fit) and classify
    @param out_file: output file name
    @param model_name: name of model
    @param resolution: resolution to learn
    """
    num_states = 8
    pca_energy = 0.85
    if model_name is None:
        model_name = 'multichannel-%ibp-%iStates' % (resolution, num_states)

    train_chromosome = 'chr7'
    strategy = GMMClassifier()
    model = DNaseMultiChannelClassifier(strategy, resolution, model_name)
    data = model.load_data(directory=in_dir)
    strategy.default(data, train_chromosome=train_chromosome, num_states=num_states, pca_energy=pca_energy)
    strategy.training_chr = [train_chromosome]

    likelihood, fit_params = model.fit(data)  # pca projection matrix selection + forward-backward

    description = model.html_description(in_dir)
    methods = "See above"
    verification = "TODO"
    credits_details = "These data were generated in labs from the " \
                      "<a href=\"http://www.roadmapepigenomics.org/\">Roadmap Epigenomics Project.</a>"
    references = "-"
    track_description = create_description_html(description, methods, verification, credits_details, references)

    classification = model.classify_data(data)  # viterbi

    # save
    model.save()
    SeqLoader.save_result_dict(os.path.join(MODELS_DIR, model_name, out_file or "segmentation"), classification)

    publish_dic(classification,
                resolution,
                '%sSegmentome' % model_name,
                short_label="Mutlicell-PCA%idims-%iStates" % (strategy.pca_ndims()[0], num_states),
                long_label="%i states, multi-cell DNase HMM GMM" % num_states, colors=True,
                description_html=track_description)
    model.readme(in_dir, strategy.pca_reduction, fit_params['likelihoods'])


def simple_pca(resolution=1000, in_dir=MEAN_DNASE_DIR):
    """
    Simple PCA with no HMM
    @param in_dir: directory of data to learn (fit) and classify
    @param resolution: binning resolution
    """
    pca_energy = 0.85
    pca_components = None
    train_chromosome = 'chr7'
    pca_reduction = PcaTransformer()

    loader = LazyChromosomeLoader(lambda x: np.log(DNaseMultiChannelClassifier.
                                                   load_multichannel_data(False, resolution, chromosomes=[x],
                                                   directory=in_dir)[x]+1))
    pca_reduction.fit(loader[train_chromosome], min_energy=pca_energy, ndim=pca_components)
    pca_dims = []
    for i in range(pca_reduction.w.shape[0]):
        pca_dims.append(dict())

    for chrom, chrom_val in loader.items():
        reduction = pca_reduction(chrom_val)
        for dim_i, dim in enumerate(reduction):
            pca_dims[dim_i][chrom] = dim

    for dim_i in range(pca_reduction.w.shape[0]):
        publish_dic(pca_dims[dim_i],
                    resolution,
                    'SegmentomePCA%i-%idim' % (dim_i+1, pca_reduction.w.shape[0]),
                    short_label="Mutlicell-PCA%i-%idim" % (dim_i+1, pca_reduction.w.shape[0]),
                    long_label="Mutlicell-PCA%i-%idim" % (dim_i+1, pca_reduction.w.shape[0]))

if __name__ == '__main__':
    commands = {
        'multichannel_discrete': multichannel_hmm_discrete,
        'multichannel_continuous': multichannel_hmm_continuous,
        'simple_pca': simple_pca,
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