"""
Classifies the genome using HMM model.
"""
import numpy as np
import data_provider
from dnase.ClassifierStrategy import ClassifierStrategy
from hmm.HMMModel import HMMModel, ContinuousHMM, DiscreteHMM
from hmm.bwiter import bw_iter, IteratorCondition

__author__ = 'eranroz'


class HMMClassifier(ClassifierStrategy):
    """
    A classifier based on HMM.
    """

    def __init__(self, model):
        """
        Initializes a new instance of C{HMMClassifier}
        @param model: a HMM model the classifier is based upon
        @type model: C{HMMModel}
        """
        self.model = model
        self.train_chromosomes = ['chr1']
        # whether to output the probability for second state (posterior) or
        # of the maximum likelihood states path (viterbi)
        self.output_p = False

    def training_chr(self, chromosomes):
        """
        Specifies on which chromosome we want to train or fit the model
        @param chromosomes: chromosomes names for training
        @return: None
        """
        self.train_chromosomes = chromosomes

    def fit(self, training_sequences, iterations=10):
        """
        fits the classifiers to training sequences and returns the log likelihood after fitting
        @param iterations: number of iterations
        @param training_sequences:
        @return:
        """
        old_model = self.model
        # TODO: the current implementation of bw_iter assumes one sequence, should consider more sequences?
        # assumption that DNase behaviour in chromosome 1 (the longest) is enough
        print("Starting fitting")
        training_seqs = []
        for train_chrm in self.train_chromosomes:
            training_seqs += [seq[train_chrm] for seq in training_sequences]

        #TODO: use different sequences?
        self.model, p = bw_iter(training_sequences[0][self.train_chromosomes[0]], self.model,
                                IteratorCondition(iterations))

        print("Model fitting finished. likelihood", p)
        print("Old model")
        print(old_model)
        print("New model")
        print(self.model)
        return p

    def classify(self, sequence_dict):
        """
        Classifies all chromosomes within genome dictionary
        @param sequence_dict: genome dictionary - where keys are chromosomes and values are scores
        @return:
        """
        classified = dict()

        for chromosome, sequence in sequence_dict.items():
            print('Classifying chromosome', chromosome)
            if self.output_p:
                bf_output = self.model.forward_backward(sequence)
                classified[chromosome] = bf_output.state_p[:, 1]  # 0 is closed and 1 is open
            else:
                classified[chromosome] = self.model.viterbi(sequence)

        return classified

    @staticmethod
    def default(discrete=False, min_alpha=0):
        """
        Creates hmm classifier with default model
        @param min_alpha: constraint on minimum self transition in EM
        @param discrete: whether to sue discrete or continuous HMM
        @return: {HMMClassifier} with default model
        """
        if discrete:
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
            model = DiscreteHMM(state_transition, emission, min_alpha=min_alpha)
            strategy = HMMClassifier(model)
        else:
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

            model = ContinuousHMM(state_transition, emission, min_alpha=min_alpha)
            strategy = HMMClassifier(model)
        return strategy

    def data_transform(self):
        """
        Get data transformer for transforming raw data before classify or fit
        """
        if isinstance(self.model, DiscreteHMM):
            transformer = data_provider.DiscreteTransformer()
        elif isinstance(self.model, ContinuousHMM):
            transformer = lambda x: np.log(np.array(x) + 1)
        else:
            raise Exception("Unknown model type")
        return transformer