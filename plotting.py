
import hiive.mdptoolbox as mdptoolbox
from  hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import PolicyIterationModified, ValueIteration, PolicyIteration, QLearning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import gym


# Stole from https://github.com/reedipher/CS-7641-reinforcement_learning/blob/master/code/forest.ipynb
def plot_forest(policy, title='Forest Management'):
    colors = {
    0: 'g',
    1: 'k'
}

    labels = {
        0: 'W',
        1: 'C',
    }
    rows = 50
    cols = 50
    
    # reshape policy array to be 2-D - assumes 500 states...
    policy = np.array(list(policy)).reshape(rows,cols)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, xlim=(-.01, cols+0.01), ylim = (-.01, rows+0.01))
    plt.title(title, fontsize=16, weight='bold', y=1.01)
    
    for i in range(rows):
        for j in range(rows):
            y = rows - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            p.set_facecolor(colors[policy[i,j]])
            ax.add_patch(p)
            
            text = ax.text(x+0.5, y+0.5, labels[policy[i, j]],
                           horizontalalignment='center', size=10, verticalalignment='center', color='w')
    
    plt.axis('off')
    plt.savefig('./out/' + title + '.png', dpi=400, bbox_inches='tight')
    plt.clf()



# Got help with creating the frozen lake env via 
# https://github.com/reedipher/CS-7641-reinforcement_learning/blob/master/code/frozen.ipynb
def plot_lake(env, policy=None, title='Frozen Lake'):
    colors = {
        b'S': 'y',
        b'F': 'w',
        b'H': 'k',
        b'G': 'g'
    }

    directions = {
                0: '←',
                1: '↓',
                2: '→',
                3: '↑'
    }
    squares = env.nrow
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, xlim=(-.01, squares+0.01), ylim=(-.01, squares+0.01))
    plt.title(title, fontsize=16, weight='bold', y=1.01)
    count = 0
    for i in range(squares):
        for j in range(squares):
            y = squares - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1, linewidth=1, edgecolor='k')
            p.set_facecolor(colors[env.desc[i,j]])
            ax.add_patch(p)
            
            if policy is not None:
                text = ax.text(x+0.5, y+0.5, directions[policy[count]],
                               horizontalalignment='center', size=25, verticalalignment='center',
                               color='k')
                count += 1
            
    plt.axis('off')
    plt.savefig('./out/' + title + '.png', dpi=400, bbox_inches='tight')
    plt.clf()
