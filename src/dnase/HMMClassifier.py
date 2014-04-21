"""
Classifies the genome using HMM model.
"""

from dnase.ClassifierStrategy import ClassifierStrategy
from hmm.HMMModel import HMMModel
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