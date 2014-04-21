import os

import numpy as np

from config import RES_DIR
from data_provider import SeqLoader
from data_provider.DiscreteTransformer import DiscreteTransformer
from hmm import bwiter
from hmm.HMMModel import DiscreteHMM


__author__ = 'eranroz'

"""
This script evaluates different transition probabilities
for hmm model
Use it to estimate whether there are local maxima points the the bw iterations getting into.
"""
#resolution = 100
resolution = 100
iterations = 7

MODEL_EVALUATION_RESULTS = os.path.join(RES_DIR, 'modelEvaluation')

# transitions from open to closed
alphas = 0.1 ** np.arange(1, 6)
# transitions from closed to open
betas = 0.1 ** np.arange(1, 6)


def calcMatrix():
    res_file = open(os.path.join(MODEL_EVALUATION_RESULTS, 'modelTransitionsEvaluation.%s.txt' % str(resolution)), 'w')
    print('Loading data')
    training = SeqLoader.load_dict('UW.Fetal_Brain.ChromatinAccessibility.H-22510.DS11872', resolution,
                                   DiscreteTransformer())
    print('Creating model')

    res_matrix = np.zeros((len(alphas), len(betas)))

    for r_i, alpha in enumerate(alphas):
        for c_i, beta in enumerate(betas):
            print('alpha', alpha)
            print('beta', beta)
            state_transition = np.array(
                [
                    [0.0, 0.99, 0.01], # begin
                    [0.3, 1 - alpha, alpha], # open (may go to close but prefers to keep the state)
                    [0.7, beta, 1 - beta]  # closed (very small change to get to open)
                ]
            )
            emission = np.array([
                np.zeros(4),
                [0.02, 0.4, 0.5, 0.08], # open - prefers high values
                [0.8, 0.1, 0.09, 0.01], # closed - prefers low values
            ])

            model = DiscreteHMM(state_transition, emission)
            res_file.write('\n-------------------\n')
            res_file.write('Closed-> Open: %s\t,\t Open->Closed: %s\n' % (str(beta), str(alpha)))
            res_file.write(str(model))
            print('bw start')
            new_model, p = bwiter.bw_iter(training['chr1'], model, iterations)
            res_file.write('\nnew model\n')
            res_file.write(str(new_model))
            res_file.write('\np:%s' % str(p))
            res_matrix[r_i, c_i] = p

    res_file.write('Local maxima as function of different guess parameters')
    res_file.write(str(res_matrix))

    res_file.close()
    np.save(os.path.join(MODEL_EVALUATION_RESULTS, 'modelTransitionsEvaluationMatrix.%s.txt' % str(resolution)), res_matrix)


def create_plot():
    import matplotlib.pyplot as plt
    res = np.array([
        [-2181835.89867419, -2192408.7940744, -2200215.6704851, -2204726.1856845, -2207530.84739521],
        [-2189644.41857069, -2199110.289325, -2203881.83793141, -2206813.13797173, -2208797.0438533],
        [-2197568.64216762, -2203697.64678551, -2206737.7255108, -2208738.40633111, -2210177.4799934],
        [-2202566.21615843, -2206645.72639486, -2208725.58699014, -2210171.86069338, -2211272.36491941],
        [-2205750.18901925, -2208654.02659629, -2210165.36927431, -2211271.38763584, -2212139.36257863]
    ])
    alphasA = np.log(alphas)
    betasA = np.log(betas)
    X, Y = np.meshgrid(alphasA, betasA)
    zs = res

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, zs)
    #fig.savefig('alphaBeta.png')
    plt.show()
#create_plot()
calcMatrix()
