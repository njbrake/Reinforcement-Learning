import hiive.mdptoolbox as mdptoolbox
from  hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import PolicyIterationModified, ValueIteration, PolicyIteration, QLearning
import matplotlib.pyplot as plt
import numpy as np
import argparse

from froz import FrozenLakeEnv
# Got help with creating the frozen lake env via 
# https://github.com/luclement/ml-assignment4/blob/master/frozen_lake.ipynb
def getTransitionAndReward(env):
    nA, nS = env.nA, env.nS
    T = np.zeros([nA, nS, nS])
    R = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for p_trans, next_s, reward, _ in transitions:
                T[a,s,next_s] += p_trans
                R[s,a] = reward
            T[a,s,:] /= np.sum(T[a,s,:])
    return T, R


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning!')
    # parser.add_argument('--clustering', action='store_true',
    #                     help='If true, run clustering')
    parser.add_argument('--part', help='The part of the assignment you\'re doing')
    args = parser.parse_args()


    envs = []
    envs.append(getTransitionAndReward(FrozenLakeEnv(penalty=0.0, prob=0.3)))
    # S is the number of states. You have a 1-p probability of moving to the next state (getting closer to the big reward once it's fully grown) and a p probability of moving back to the youngest state.
    # You get 0 reward if you cut down the tree when it's in its youngest state, 1 reward if cut when it's not it's youngest state, and r2 reward if you cut it when its at its oldest state. You get 0 
    # reward for waiting unless you wait at the oldest state, then you get r1
    envs.append(forest(S=5000, r1=10000, r2=7000, p=0.001))

    for (P,R), name in zip(envs, ['frozen', 'foreset']):

        algs = []
        if args.part == '1':
            algs.append((ValueIteration(P, R, 0.99, max_iter=20000), "VI"))
            algs.append((PolicyIteration(P, R, 0.99, max_iter=20000), "PI"))
        else:
            print("running q learning")
            algs.append((QLearning(P, R, 0.96, alpha=0.2, epsilon=0.01, epsilon_decay=0.9999, n_iter=100000000, run_stat_frequency=1000), "QL"))
        for alg, name in algs:
            runs = alg.run()
            maxR  = runs[-1]['Max V']
            print(f"maxR for {name} is {maxR}")
            print(f"took {alg.time} seconds")
            xs = []
            ys = []
            for i,run in enumerate(runs):
                ys.append(run['Max V'])
                xs.append(i)
            plt.plot(xs,ys)
            plt.savefig(f"out/{name}.png",bbox_inches='tight')
            plt.clf()