from hmm.bwiter import IteratorCondition

__author__ = 'eran'
"""
Test for Baum-Welch, or iterative backward forward
"""


from hmm import bwiter
from hmm.HMMModel import HMMModel, DiscreteHMM
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
__author__ = 'eran'


def simpleBwIter():
    n_tiles = 15000
    fair = True
    dice = []
    realDice = []
    for i in range(0, n_tiles):
        if fair:
            fair = np.random.randint(1, 101) <= 95
        else:
            fair = np.random.randint(1, 101) >= 90

        realDice.append(1 if fair else 2)
        if fair:
            dice.append(np.random.randint(1, 7))
        else:
            #in unfair dice 1/10 for 1-5 and 9/10 for 6
            dice.append(6 if np.random.randint(0, 2) < 1 else np.random.randint(1, 6))

    #state to state
    a_zero = 1e-300
    state_transition = np.array([
        [a_zero, 1, a_zero],
        [a_zero, 0.75, 0.25],
        [a_zero, 0.1, 0.9]
    ])

    #satet to outputs
    beginEmission = np.ones(6)

    fairEmission = np.ones(6) / 6  # all equal
    #fairEmission=[0.1, 0.2, 0.1, 0.2, 0.2, 0.2]
    unfairEmission = np.ones(6) / 10  # all equal to 1/10 and 6 to 0.5
    unfairEmission[5] = 0.5

    emission = np.array([beginEmission, fairEmission, unfairEmission])
    initial_model = DiscreteHMM(state_transition, emission)
    translated_dice = np.array(dice)  # translate to emission alphabet
    translated_dice -= 1
    #rand_model = bwiter.bw_iter(translated_dice, n_states=3, n_alphabet=6)

    #new_model2 = bwiter.bw_iter_log(translated_dice, initial_model, stop_condition=IterCondition(5))
    new_model = bwiter.bw_iter_old(translated_dice, initial_model, stop_condition=IteratorCondition(10))
    #print('Old\n',new_model)
    new_model = bwiter.bw_iter(translated_dice, initial_model, stop_condition=IteratorCondition(10))
    #print('New\n',new_model)

    bw_output1 = initial_model.forward_backward(translated_dice)
    bw_output2 = new_model.forward_backward(translated_dice)
    #bw_output3 = fbward.forward_backward(translated_dice, rand_model)

    print(new_model)
    #print(rand_model)
    #print('----')
    #print(bw_output1.model_p)
    #print(bw_output2.model_p)
    #print(bw_output3.model_p)

    def showPlot():
        fig, ax = plt.subplots()
        unfair_regions = []
        unfair_begin = -1
        for i, x in enumerate(realDice):
            if x == 2 and unfair_begin == -1:
                unfair_begin = i
            if x == 1 and unfair_begin != -1:
                unfair_regions.append((unfair_begin, i))
                unfair_begin = -1

        plt.plot(bw_output1.state_p[:, 1], '-r')
        #plt.plot(bw_output2.state_p[:, 1]*np.exp(bw_output1.model_p)/np.exp(bw_output2.model_p), '-b')
        plt.plot(bw_output2.state_p[:, 1], '-b')
        max_p = max(bw_output1.state_p[:, 1])
        for s, t in unfair_regions:
            p = Polygon([(s, 0), (s, max_p), (t, max_p), (t, 0)], facecolor='0.9', edgecolor='0.5')
            ax.add_patch(p)

        plt.show()
    #showPlot()

def logTest():
    n_tiles = 5000
    fair = True
    dice = []
    realDice = []
    for i in range(0, n_tiles):
        if fair:
            fair = np.random.randint(1, 101) <= 95
        else:
            fair = np.random.randint(1, 101) >= 90

        realDice.append(1 if fair else 2)
        if fair:
            dice.append(np.random.randint(1, 7))
        else:
            #in unfair dice 1/10 for 1-5 and 9/10 for 6
            dice.append(6 if np.random.randint(0, 2) < 1 else np.random.randint(1, 6))

    #state to state
    a_zero = 1e-300
    state_transition = np.array([
        [a_zero, 1, a_zero],
        [a_zero, 0.75, 0.25],
        [a_zero, 0.1, 0.9]
    ])

    #satet to outputs
    beginEmission = np.ones(6)

    fairEmission = np.ones(6) / 6  # all equal
    #fairEmission=[0.1, 0.2, 0.1, 0.2, 0.2, 0.2]
    unfairEmission = np.ones(6) / 10  # all equal to 1/10 and 6 to 0.5
    unfairEmission[5] = 0.5

    emission = np.array([beginEmission, fairEmission, unfairEmission])
    initial_model = HMMModel(state_transition, emission)
    translated_dice = np.array(dice)  # translate to emission alphabet
    translated_dice -= 1
    #rand_model = bwiter.bw_iter(translated_dice, n_states=3, n_alphabet=6)

    #from scipy.stats import pearsonr
    #print(pearsonr(np.exp(fbward.forward_backward_log(translated_dice,initial_model).state_p[:, 1]),fbward.forward_backward(translated_dice, initial_model).state_p[:, 1]))
    #print(np.exp(fbward.forward_backward_log(translated_dice,initial_model).state_p[0:5, 1]))
    #print(fbward.forward_backward(translated_dice, initial_model).state_p[0:5, 1])
    new_model2 = bwiter.bw_iter_log(translated_dice, initial_model, stop_condition=IteratorCondition(10))
    new_model = bwiter.bw_iter(translated_dice, initial_model, stop_condition=IteratorCondition(1))
    print('Correct:\n', new_model.emission[1:] )
    print('Log:\n', new_model2.emission[1:] )

#cProfile.run('simpleBwIter()')
simpleBwIter()
#logTest()