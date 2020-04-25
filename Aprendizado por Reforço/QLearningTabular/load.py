import pickle
import numpy as np
from objects import Environment

def discretize(s):
    return tuple(round(i/10) for i in s)

def load_table(file):
	with open(file, 'rb') as pickle_in:
		Q = pickle.load(pickle_in)
	return Q

env = Environment()
Q = load_table('model.pickle')

NUMBER_OF_EPISODES = 1

for i in range(NUMBER_OF_EPISODES):
	done = False
	s = env.reset()
	s = discretize(s)
	while not done:
		action = np.argmax(Q[s])
		s2, reward, done, _ = env.step(action)
		s2 = discretize(s2)
		env.render()
		s = s2

