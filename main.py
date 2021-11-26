import hiive.mdptoolbox as mdptoolbox
from  hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import PolicyIterationModified, ValueIteration, PolicyIteration, QLearning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import gym

from plotting import plot_lake, plot_forest

# suppress pandas warning
pd.options.mode.chained_assignment = None

# set seed
np.random.seed(28)

def get_policy(env,stateValue, lmbda=0.9):
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + lmbda * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning!')
    # parser.add_argument('--clustering', action='store_true',
    #                     help='If true, run clustering')
    parser.add_argument('--part', help='The part of the assignment you\'re doing')
    args = parser.parse_args()


    envs = []
    # Setup 4x4
    # P = 0.3 of going in correct direction
    env = gym.make('FrozenLake-v1').unwrapped

    env.max_episode_steps=500

    # Create transition and reward matrices from OpenAI P matrix
    rows = env.nrow
    cols = env.ncol
    T = np.zeros((4, rows*cols, rows*cols))
    R = np.zeros((4, rows*cols, rows*cols))

    old_state = np.inf

    for square in env.P:
        for action in env.P[square]:
            available_action = env.P[square][action]
            for i in range(len(available_action)):
                new_state = available_action[i][1]
                if new_state == old_state:
                    T[action][square][available_action[i][1]] = T[action][square][old_state] + available_action[i][0]
                    R[action][square][available_action[i][1]] = R[action][square][old_state] + available_action[i][2]
                else:
                    T[action][square][available_action[i][1]] = available_action[i][0]
                    R[action][square][available_action[i][1]] = available_action[i][2]
                old_state = available_action[i][1]
                
    plot_lake(env)
    envs.append((T,R))
    # S is the number of states. You have a 1-p probability of moving to the next state (getting closer to the big reward once it's fully grown) and a p probability of moving back to the youngest state.
    # You get 0 reward if you cut down the tree when it's in its youngest state, 1 reward if cut when it's not it's youngest state, and r2 reward if you cut it when its at its oldest state. You get 0 
    # reward for waiting unless you wait at the oldest state, then you get r1
    forest_env = forest(S=2500, r1=100, r2=50, p=0.1)
    envs.append(forest_env)

    for index, (P,R), prob_name in zip([0,1],envs, ['frozen', 'forest']):
        algs = []
        if index == 0:
            continue
        if args.part == '1':
            algs.append((ValueIteration(P, R, 0.99, epsilon=0.0001, max_iter=1000), "VI"))
            algs.append((PolicyIteration(P, R, 0.99, max_iter=1000), "PI"))
        else:
            print("running q learning")
            def check_if_new_episode(old_s, action, new_s):
                if new_s == 'G' or new_s == 'H':
                    return True
                else:
                    return False
            algs.append((QLearning(P, R, 0.9999, alpha=0.05, alpha_decay=0.9999, epsilon=0.99999, epsilon_decay=0.999, n_iter=80000000), "QL"))
        for alg, name in algs:
            runs = alg.run()
            maxR = runs[-1]['Max V']
            iters = runs[-1]['Iteration']
         
            print(f"maxR for {name} is {maxR}")

            print(f"took {alg.time} seconds")
            print(f"Took {iters} iterations")
            xs = []
            ys = []
            for run in runs:
                ys.append(run['Max V'])
                xs.append(run['Iteration'])
            plt.plot(xs,ys)
            if prob_name == 'frozen':
                delta = 0.05
            else:
                delta = 10
            ymin = min(ys)-delta
            ymax = max(ys)+delta
            plt.ylim(ymin, ymax)
            plt.savefig(f"out/{name}_{prob_name}.png",bbox_inches='tight')
            plt.clf()
            if prob_name == 'frozen':
                plot_lake(env, alg.policy, f"mapped_{name}_{prob_name}")
            else:
                plot_forest(alg.policy, f"mapped_{name}_{prob_name}")