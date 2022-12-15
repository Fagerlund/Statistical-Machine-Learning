import blackjack_extended as bjk #The extended Black-Jack environment
import blackjack_base as bjk_base #The standard sum-based Black-Jack environment
from math import inf
import RL as rl
import sys
import os
import os.path

import plotting as pl
import time
import matplotlib
import csv
from csv import DictReader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


for decks in [1,2,6,8,inf]:
    # Create zero-matrices
    actionsTrue = np.zeros((32,27))
    actionsFalse = np.zeros((32,27))

    # Read sumQ files
    with open('sumQ_{}.csv'.format(str(decks)), 'r') as data:
        dict_reader = csv.DictReader(data, fieldnames=('psum','dsum','uace','action'))
        for dict in dict_reader:
            # Clean up the data
            dict['psum'] = int(dict['psum'].replace('(',''))
            dict['dsum'] = int(dict['dsum'])
            dict['uace'] = dict['uace'].replace(')','')
            dict['uace'] = dict['uace'].replace(' ','')
            dict['uace'] = eval(dict['uace'])
            dict['action'] = dict['action'].replace('[','')
            dict['action'] = dict['action'].replace(']','')
            actionlist = list(dict['action'].split(' '))
            actionlist = [actionlist[0], actionlist[1]]
            dict['action'] = [eval(i) for i in actionlist]

            # Set action to respective player and dealer sum
            if dict['uace']: # Usable ace
                # Assign the action with highest expected value
                actionsTrue[dict['psum'], dict['dsum']] = np.argmax(dict['action'])
            else: # No usable ace
                actionsFalse[dict['psum'], dict['dsum']] = np.argmax(dict['action'])

    # Only keep the relevant player and dealer sums
    actionsTrue = np.delete(actionsTrue,[*range(21,32)],0)
    actionsTrue = np.delete(actionsTrue,[*range(12)],0)
    actionsTrue = np.delete(actionsTrue,[*range(12,27)],1)
    actionsTrue = np.delete(actionsTrue,[*range(2)],1)

    actionsFalse = np.delete(actionsFalse,[*range(21,32)],0)
    actionsFalse = np.delete(actionsFalse,[*range(10)],0)
    actionsFalse = np.delete(actionsFalse,[*range(12,27)],1)
    actionsFalse = np.delete(actionsFalse,[*range(2)],1)


    # Create heatmap for usable ace case
    plt.figure()
    ax = plt.axes()
    dfactionsTrue = pd.DataFrame(actionsTrue, index=[*range(12,21)], columns=[*range(2,12)])
    hmapTrue = sns.heatmap(dfactionsTrue, cbar=False, linewidth=.5, ax=ax)
    
    ax.set_title('With usable ace, number of decks: {}'.format(str(decks)))
    hmapTrue.set(xlabel="Dealer", ylabel="Player")

    figTrue = hmapTrue.get_figure()
    figTrue.savefig('heatmapTrue_{}.png'.format(str(decks)))


    # Create heatmap for non-usable ace case
    plt.figure()
    ax = plt.axes()
    dfactionsFalse = pd.DataFrame(actionsFalse, index=[*range(10,21)], columns=[*range(2,12)])
    hmapFalse = sns.heatmap(dfactionsFalse, cbar=False, linewidth=.5, ax=ax)
    
    ax.set_title('Without usable ace, number of decks: {}'.format(str(decks)))
    hmapFalse.set(xlabel="Dealer", ylabel="Player")

    figFalse = hmapFalse.get_figure()
    figFalse.savefig('heatmapFalse_{}.png'.format(str(decks)))