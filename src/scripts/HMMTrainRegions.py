"""
This script trains a model based on different regions of chromosome 1,
and evaluates the model as the likelihood of it for the whole chromosome 2.

Useful to know whether it is sufficient to train only based on smaller data
"""
import os

import numpy as np

from config import RES_DIR
from data_provider import SeqLoader
from data_provider.DiscreteTransformer import DiscreteTransformer
from hmm import bwiter
from hmm.HMMModel import DiscreteHMM


__author__ = 'eranroz'


MODEL_EVALUATION_RESULTS = os.path.join(RES_DIR, 'modelEvaluation')

resolution = 100
iterations = 7
res_file = open(os.path.join(MODEL_EVALUATION_RESULTS, 'trainRegions.10n100.%s.txt' % str(resolution)), 'w')
training = SeqLoader.load_dict('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872', resolution,
                               DiscreteTransformer())
chromosomeLength = len(training['chr1'])

for regionSize in [10, 100]:
    res_file.write('region sizes %s' % (str(regionSize)))
    res_matrix = np.zeros(regionSize)
    max_p = -999999999
    max_region = None
    for itera, start in enumerate(np.arange(0, chromosomeLength, chromosomeLength / regionSize)):
        region = training['chr1'][start: start + chromosomeLength / regionSize]

        state_transition = np.array(
            [
                [0.0, 0.99, 0.01],  # begin
                [0.3, 0.9, 0.1],  # open (may go to close but prefers to keep the state)
                [0.7, 0.1, 0.9]  # closed (very small change to get to open)
            ]
        )
        emission = np.array([
            np.zeros(4),
            [0.02, 0.4, 0.5, 0.08],  # open - prefers high values
            [0.8, 0.1, 0.09, 0.01],  # closed - prefers low values
        ])

        model = DiscreteHMM(state_transition, emission)
        res_file.write('-------------------')
        res_file.write('Trained on region: %s - %s' % (str(start), str(start + chromosomeLength / 10)))
        new_model, p = bwiter.bw_iter(region, model, iterations)
        #res_file.write(str(new_model))
        bw_output = new_model.forward_backward(training['chr2'])
        res_file.write('Likelihood (chr2): %s' % str(bw_output.model_p))
        res_matrix[itera] = bw_output.model_p
        if bw_output.model_p > max_p:
            max_p = bw_output.model_p
            max_region = str(start)

    print('likelihood as function of region in chromosome 1')
    print(str(res_matrix))
    print('Max region: %s' % max_region)
    print('Max P: %s' % max_p)

res_file.close()
print('Finished')


