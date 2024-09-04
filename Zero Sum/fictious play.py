#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:14:11 2024

@author: msayed
"""

import numpy as np
import nashpy as nash

# Given reward matrix
reward = np.array([
    [-187., -187., -110., -110.],
    [-187., -187., -184.,   16.],
    [ -93.,  -93., -184., -184.],
    [-187., -187., -138., -138.],
    [-187., -187.,   16., -184.],
    [-187., -187., -136., -136.],
    [-157., -157., -184., -184.],
    [-187.,   13., -184., -184.],
    [-137., -137., -184., -184.],
    [  13., -187., -184., -184.]
])


game1 = nash.Game(reward)  
iterations1=10000
np.random.seed(0)
play_counts = tuple(game1.fictitious_play(iterations=iterations1))
equilibria1 = play_counts[-1]
def_mixed_strategy = equilibria1[0].copy()/iterations1
att_mixed_strategy = equilibria1[1].copy()/iterations1
   
def_random = [1/len(def_mixed_strategy) for _ in range(len(def_mixed_strategy))]
# def_random_good=[0, 1/2, 0, 1/2]
#print("def_mixed_strategy,att_mixed_strategy", def_mixed_strategy, att_mixed_strategy)
print('rew_NE', game1[def_mixed_strategy, att_mixed_strategy]) 

#print("def_random,att_mixed_strategy", def_random, att_mixed_strategy)
print('rew_Random', game1[def_random,att_mixed_strategy])
# print('rew_Random_good', game1[def_random_good,att_mixed_strategy])
def_MCN= [0 for _ in range(len(def_mixed_strategy))]


def_MCN[-2]=1/2
def_MCN[-1]=1/2
print('rew_greedy', game1[def_MCN,att_mixed_strategy])
