import numpy as np
from objects import Environment
import pickle
import matplotlib.pyplot as plt

a = 0.05 #learning rate
e_min = 0.01
e = 0.7 # epsilon
gamma = 0.9  # fator de desconto
decay = 0.9999999 # decaímento do epsilon
N_EPISODES = 1000
times = []
Q = {} # keys: estados; values: valor atribuido à cada ação

def discretize(s):
    return tuple(round(i/10) for i in s)

def save_model(Q, name = 'model.pickle'):
    with open(name,'wb') as pickle_out:
        pickle.dump(Q, pickle_out)

def choose_action(s, e):
    if np.random.random() < e:
        action = np.random.choice([0,1,2])
    else:
        action = np.argmax(Q[s])
    e *= decay
    return action, max(e, e_min)

def train(state, action, reward, next_state):
    # para cada estado ainda não descoberto, iniciamos seu valor como nulo
    if s not in Q.keys(): Q[s] = [0,0,0] 
    if s2 not in Q.keys(): Q[s2] = [0,0,0]      

    # equação de Bellman
    Q[s][action] = Q[s][action] + a*(r + gamma*np.max(Q[s2]) - Q[s][action])


env = Environment()
rewards = []
for i_episode in range(1,N_EPISODES+1):
    
    s = env.reset()
    s = discretize(s)
    if s not in Q.keys(): Q[s] = [0,0,0]
    
    done = False
    t = 0
    total_reward = 0
    
    # main loop
    while not done:
        # politica
        action, e = choose_action(s, e)
        # A ação é tomada e os valores novos são coletados
        # O novo estado é salvo numa nova variavel
        s2, r, done, info = env.step(action)
        s2 = discretize(s2)
        total_reward += r
        
        train(s, action, r, s2)
        

        s = s2
        t += 1
    
    rewards.append(total_reward)
    if i_episode%10 == 0:
        save_model(Q)
    if i_episode%50 == 0:
        save_model(times, 'times.pickle')
    times.append(t)
    print(f'{i_episode} durou {t}, recompensa {total_reward:.2f}, recompensa média {np.mean(rewards[-min(len(rewards),50):]):.2f}, score {env.score[0]}x{env.score[1]}, epsilon: {e:.2f}, tamanho da tabela: {len(Q)}')


plt.plot(range(len(times)),[np.mean(times[max(0,t-50):t+1]) for t in range(len(times))], color = 'g')
plt.show()

