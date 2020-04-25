import pickle
import matplotlib.pyplot as plt
from numpy import mean

def load_table(file):
	with open(file, 'rb') as pickle_in:
		Q = pickle.load(pickle_in)
	return Q

times = load_table('times.pickle')

plt.style.use('seaborn')
plt.figure(figsize=(16,16),dpi=80)
#plt.plot(range(len(times)),times)
plt.plot(range(len(times)),[mean(times[max(0,t-50):t]) for t in range(len(times))],
         color = 'r')
plt.show()