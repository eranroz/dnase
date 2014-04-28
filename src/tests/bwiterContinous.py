from hmm.bwiter import IteratorCondition
from hmm.multivariatenormal import MultivariateNormal

__author__ = 'eran'
"""
Test for Baum-Welch, or iterative backward forward
"""

from hmm import bwiter
from hmm.HMMModel import _ContinuousEmission, ContinuousHMM, GaussianHMM
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon

__author__ = 'eran'


def showPlot(realDice, bw_output):
    fig, ax = plt.subplots()
    unfair_regions = []
    unfair_begin = -1
    for i, x in enumerate(realDice):
        if x == 2 and unfair_begin == -1:
            unfair_begin = i
        if x == 1 and unfair_begin != -1:
            unfair_regions.append((unfair_begin, i))
            unfair_begin = -1

    plt.plot(bw_output.state_p[:, 0], '-r')
    max_p = max(bw_output.state_p[:, 0])
    for s, t in unfair_regions:
        p = Polygon([(s, 0), (s, max_p), (t, max_p), (t, 0)], facecolor='0.9', edgecolor='0.5')
        ax.add_patch(p)
    plt.show()


def simple_continuous():
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
            dice.append(np.random.normal(3, 1, 1)[0])
        else:
            #in unfair dice 1/10 for 1-5 and 9/10 for 6
            dice.append(np.random.normal(6, 2, 1)[0])

    #state to state
    a_zero = 1e-300
    state_transition = np.array([
        [a_zero, 1, a_zero],
        [a_zero, 0.95, 0.05],
        [a_zero, 0.1, 0.9]
    ])

    #satet to outputs
    beginEmission = np.ones(6)

    fairEmission = np.ones(6) / 6  # all equal
    #fairEmission=[0.1, 0.2, 0.1, 0.2, 0.2, 0.2]
    unfairEmission = np.ones(6) / 10  # all equal to 1/10 and 6 to 0.5
    unfairEmission[5] = 0.5

    emission = np.array([beginEmission, fairEmission, unfairEmission])
    con_emission = _ContinuousEmission(np.array([
        [0, 1],
        [2, 2],  # fair
        [5, 1]  # unfair tends to 6
    ]))

    initial_model = ContinuousHMM(state_transition, np.array([
        [0, 1],
        [2, 2],  # fair
        [5, 1]  # unfair tends to 6
    ]))  #
    intial_gaussian_model = GaussianHMM(state_transition, [
        (2, 2),  # fair
        (5, 1)  # unfair tends to 6
    ])
    translated_dice = np.array(dice)  # translate to emission alphabet

    emis1 = initial_model.emission[1:, translated_dice]
    print(emis1[0, 0:5])
    emis2 = intial_gaussian_model.emission[1:, translated_dice]
    n_iter = 10  # 15
    import time

    b1 = time.time()
    new_model2, p = bwiter.bw_iter(translated_dice, intial_gaussian_model, IteratorCondition(n_iter))
    b2 = time.time()
    new_model, p = bwiter.bw_iter(translated_dice, initial_model, IteratorCondition(n_iter))
    b3 = time.time()

    print(b2 - b1)  # new mixture
    print(b3 - b2)  # old
    # is it the cache?!
    print(new_model)
    print(new_model2)


def multi_dim_continous():
    n_tiles = 35000
    d = 2
    fair = True
    dice = np.zeros((d, n_tiles))
    realDice = []
    for i in range(0, n_tiles):
        if fair:
            fair = np.random.randint(1, 101) <= 95
        else:
            fair = np.random.randint(1, 101) >= 90

        realDice.append(1 if fair else 2)
        if fair:
            dice[0, i] = np.random.normal(3, 1, 1)[0]
            dice[1, i] = np.random.normal(3.2, 1, 1)[0]
        else:
            dice[0, i] = np.random.normal(6.3, 2, 1)[0]
            dice[1, i] = np.random.normal(6, 2, 1)[0]

    #state to state
    a_zero = 1e-300
    state_transition = np.array([
        [a_zero, 1, a_zero],
        [a_zero, 0.95, 0.05],
        [a_zero, 0.1, 0.9]
    ])

    #satet to outputs #
    intial_gaussian_model = GaussianHMM(state_transition, [
        (
            [2.1, 1.8], [[1, 0], [0, 1]]
        ),  # fair
        (
            [5, 5.1], [[1, 0], [0, 1]]
        )  # unfair tends to 6
    ])
    translated_dice = np.array(dice)  # translate to emission alphabet

    emis2 = intial_gaussian_model.emission[1:, translated_dice]
    n_iter = 20  # 15
    import time

    b1 = time.time()
    new_model2, p = bwiter.bw_iter(translated_dice, intial_gaussian_model, IteratorCondition(n_iter))
    b2 = time.time()

    print(b2 - b1)  # new mixture

    print(new_model2)


def gaussian_mixture():
    n_tiles = 1500
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
            dice.append(np.random.normal(3, 1, 1)[0] if np.random.rand() < 0.7 else np.random.normal(7, 2, 1)[0])
        else:
            #in unfair dice 1/10 for 1-5 and 9/10 for 6
            dice.append(np.random.normal(6, 2, 1)[0])

    #state to state
    a_zero = 1e-300
    state_transition = np.array([
        [a_zero, 1, a_zero],
        [a_zero, 0.95, 0.05],
        [a_zero, 0.1, 0.9]
    ])

    #satet to outputs
    beginEmission = np.ones(6)

    fairEmission = np.ones(6) / 6  # all equal
    #fairEmission=[0.1, 0.2, 0.1, 0.2, 0.2, 0.2]
    unfairEmission = np.ones(6) / 10  # all equal to 1/10 and 6 to 0.5
    unfairEmission[5] = 0.5

    initial_model = ContinuousHMM(state_transition, ([
                                                         (0, 1),
                                                         [(2, 2), (8, 1)],  # fair
                                                         (5, 1)  # unfair tends to 6
                                                     ]), mixture_coef=[1, [0.5, 0.5], 1])

    translated_dice = np.array(dice)  # translate to emission alphabet

    new_model = bwiter.bw_iter(translated_dice, initial_model, IteratorCondition(10))
    print(new_model)


#cProfile.run('runTest()')
#simple_continuous()
# #gaussian_mixture()
multi_dim_continous()
