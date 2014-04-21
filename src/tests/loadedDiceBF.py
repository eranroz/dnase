__author__ = 'eran'
"""
Test for backward & forward algorithm
This test is based on Durbin p. 61
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from hmm.HMMModel import HMMModel


__author__ = 'eran'

n_tiles = 5000
fair = True
dice = []
realDice = []
for i in range(0, n_tiles):
    if fair:
        fair = np.random.randint(1, 101) <= 95
    else:
        fair = np.random.randint(1, 101) > 90

    realDice.append(1 if fair else 2)
    if fair:
        dice.append(np.random.randint(1, 7))
    else:
        #in unfair dice 1/10 for 1-5 and 9/10 for 6
        dice.append(6 if np.random.randint(0, 2) < 1 else np.random.randint(1, 6))

#state to state
a_zero = 1e-300
state_transition = np.array([
    [a_zero, 0.99, 0.01],
    [a_zero, 0.95, 0.05],
    [a_zero, 0.1, 0.9]
])

#satet to outputs

beginEmission = np.ones(6)

fairEmission = np.ones(6) / 6  # all equal
unfairEmission = np.ones(6) / 10  # all equal to 1/10 and 6 to 0.5
unfairEmission[5] = 0.5

emission = np.array([beginEmission, fairEmission, unfairEmission])
model = HMMModel(state_transition, emission)
translated_dice = np.array(dice)  # translate to emission alphabet
translated_dice -= 1
import time

start_1 = time.time()
p_states = model.forward_backward(translated_dice.flatten())
start_2 = time.time()
p_states2 = model.forward_backward_log(translated_dice.flatten())
end = time.time()

print(p_states.model_p)
print(p_states2.model_p)
#print(np.exp(p_states2.forward)/np.sum(np.exp(p_states2.forward-p_states2.model_p[:, None]),1)[0:5])
#print(np.column_stack((np.log(p_states.forward[1]*p_states.scales[1,None]), p_states2.forward[1])))
#print(np.column_stack((np.log(p_states.state_p), p_states2.state_p)))

print('Time1:', start_2 - start_1)
print('Time2:', end - start_2)

state_p1 = np.log(p_states.state_p[:, 0])
#print(np.row_stack((np.log(p_states.state_p[0:10]), p_states2.state_p[0:10])))
# print(np.column_stack((p_states.forward[0:10], p_states2.forward[0:10])))
#raise Exception()
fig, ax = plt.subplots()
plt.plot(state_p1, '-r')
plt.plot(p_states2.state_p[:, 0], '-b')
from scipy.stats.stats import pearsonr

print("Pearson", pearsonr(state_p1, p_states2.state_p[:, 0]))
ff = p_states2.forward[1:, 0] + p_states2.backward[:-1, 0]
#plt.plot(p_states2.forward[1:, 0]+p_states2.backward[:-1, 0]-p_states2.model_p, '-g')

unfair_regions = []
unfair_begin = -1
for i, x in enumerate(realDice):
    if x == 2 and unfair_begin == -1:
        unfair_begin = i
    if x == 1 and unfair_begin != -1:
        unfair_regions.append((unfair_begin, i))
        unfair_begin = -1

max_p = np.max(state_p1)
for s, t in unfair_regions:
    p = Polygon([(s, 0), (s, max_p), (t, max_p), (t, 0)], facecolor='0.9', edgecolor='0.5')
    ax.add_patch(p)

# print the likelihood of the model
print('log likelihood', p_states.model_p)
print('log likelihood2', p_states2.model_p)

plt.show()




