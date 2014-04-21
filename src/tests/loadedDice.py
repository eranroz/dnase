import cProfile
from hmm.HMMModel import HMMModel
import numpy as np

__author__ = 'eran'


def test_dice():
    """
    Occasionally dishonest casino part 1 [Durbin p.55]
    This test the viterbi algorithm - result should be more or less as in Durbin p. 58
    """
    n_tiles = 35000
    fair = True
    dice = []
    real_dice = []
    for i in range(0, n_tiles):
        if fair:
            fair = np.random.randint(1, 101) <= 95
        else:
            fair = np.random.randint(1, 101) >= 90

        real_dice.append(1 if fair else 2)
        if fair:
            dice.append(np.random.randint(1, 7))
        else:
            #in unfair dice 1/10 for 1-5 and 9/10 for 6
            dice.append(6 if np.random.randint(0, 2) < 1 else np.random.randint(1, 6))

    #state to state
    a_zero = 0.00000001
    state_transition = np.array([
        [a_zero, 1, a_zero],
        [a_zero, 0.95, 0.05],
        [a_zero, 0.1, 0.9]
    ])

    #satet to outputs

    begin_emission = np.ones(6)

    fair_emission = np.ones(6) / 6  # all equal
    unfair_emission = np.ones(6) / 10  # all equal to 1/10 and 6 to 0.5
    unfair_emission[5] = 0.5

    emission = np.array([begin_emission, fair_emission, unfair_emission])
    model = HMMModel(state_transition, emission)
    translated_dice = np.array(dice)  # translate to emission alphabet
    translated_dice -= 1
    import time

    b = time.time()
    guess_dice = model.viterbi(translated_dice.flatten())
    b1 = time.time()
    guess_dice2 = model.viterbi_old(translated_dice.flatten())
    b2 = time.time()
    print(np.all(guess_dice == guess_dice2))
    print((b1 - b), (b2 - b1))
    #print(''.join(map(str, realDice)))
    #print(''.join(map(str, guess_dice+1)))


test_dice()
#cProfile.run('test_dice()')