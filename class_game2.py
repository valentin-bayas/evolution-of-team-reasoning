#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 15:43:56 2022

@author: valentinbayas
"""
# nashpy

import numpy as np
import pandas as pd
from numpy.random import binomial


class game:
    '''
    holds for two player, normal form symmetric games with two strategies
    '''
    def __init__(self, name: str, strat: list, payoffs):
        '''give payoffs as ndarrays as in matrix'''
        self.name = name
        self.strat = strat
        self.payoffs = payoffs
        
    def probs(w, a, b, c, d):
        return w*w*a + w*(1-w)*b + (1-w)*w*c + ((1-w)**2)*d

    def reason(self, w: float):
        '''
        w = probability of team reasoning, assumes positive payoffs that sum to more than 0
        '''
        # calculate EU of all possible profiles for team and for indiv reasoner
        A = self.strat[0]
        B = self.strat[1]
        a = self.payoffs[0][0]
        b = self.payoffs[1][1]
        x = self.payoffs[1][0] # good one
        y = self.payoffs[0][1] # sucker
        m = (x+y)/2
        EUteam = {}
        EUind = {}
        # clarify utilities for cases where team playes two different acts - is the utility if both team reason really m?
        # calculate EUs for individual players
        EUind[f'{A},{A},{A}|{A}'] = a
        EUind[f'{B},{A},{A}|{A}'] = game.probs(w, a, a, x, x)
        EUind[f'{B},{B},{A}|{A}'] = game.probs(w, a, y, x, b)
        EUind[f'{A},{B},{A}|{A}'] = game.probs(w, a, y, a, y)
        EUind[f'{A},{A},{A}|{B}'] = game.probs(w, m, m, a, a)
        EUind[f'{B},{A},{A}|{B}'] = game.probs(w, m, 0.5*(a+x), 0.5*(x+b), x)
        EUind[f'{B},{B},{A}|{B}'] = game.probs(w, m, m, b, b)
        EUind[f'{A},{B},{A}|{B}'] = game.probs(w, m, 0.5*(y+b), 0.5*(a+y), y)
        EUind[f'{A},{A},{B}|{B}'] = game.probs(w, b, x, y, a)
        EUind[f'{B},{A},{B}|{B}'] = game.probs(w, b, x, b, x)
        EUind[f'{B},{B},{B}|{B}'] = b
        EUind[f'{A},{B},{B}|{B}'] = game.probs(w, b, b, y, y)
        # calculate EUs for team - since game is symmetric cases where both individual reasoners play different strategies can be escluded
        EUteam[f'{A},{A},{A}|{A}'] = a
        EUteam[f'{B},{B},{A}|{A}'] = game.probs(w, a, m, m, b)
        EUteam[f'{A},{A},{A}|{B}'] = game.probs(w, m, m, a, a)
        EUteam[f'{B},{B},{A}|{B}'] = game.probs(w, m, m, b, b)
        EUteam[f'{A},{A},{B}|{B}'] = game.probs(w, b, m, m, a)
        EUteam[f'{B},{B},{B}|{B}'] = b
        # look for best responses
        def bresponse(competing_profiles: list, dictname: dict):
            best = max([dictname[s] for s in competing_profiles])
            return [s for s in competing_profiles if dictname[s] == best]
        bestteam = bresponse([f'{A},{A},{A}|{A}', f'{A},{A},{A}|{B}', f'{A},{A},{B}|{B}'], EUteam) + bresponse([f'{B},{B},{A}|{A}', f'{B},{B},{A}|{B}', f'{B},{B},{B}|{B}'], EUteam)
        bestind = []
        rec = []
        for i in range(len(EUind.keys())):
            rec.append(list(EUind.keys())[i])
            if i % 2 != 0:
                bestind += bresponse(rec, EUind)
                rec = []  
        bestind = [i for i in bestind if i in EUteam.keys()]
        # give equilibria (= profiles where both players play their best response)
        equilibria = []
        for i in EUind.keys():
            if i in bestind and i in bestteam:
                equilibria.append(i)
        return equilibria
    
    def decomp(self, s: str):
        return ([s.split(',')[0], s.split(',')[1]] + s.split(',')[2].split('|'))
    
    def actstoStrats(self, acts: list):
        strats = []
        for act in acts:
            for i in range(2):
                if act == self.strat[i]:
                    strats.append(i)
        return strats
    
    def act(self, est1: float, est2: float, type1, type2, equibreak1, equibreak2):
        """takes the equilibrium and players types and returns the acts they play and their fitness gains/losses 
        equibreak=probability with which players will play strat 2 over 1 if there are 2 equilibria"""
        # unpack equilibria
        eq1 = [self.decomp(i) for i in self.reason(est1)]          
        eq2 = [self.decomp(i) for i in self.reason(est2)]
        # print(eq1)
        # print(eq2)
        # find out strategies
        if type1 == 0:
            acts1 = [i[0] for i in eq1]
        else:
            acts1 = [i[2] for i in eq1]
        if type2 == 0:
            acts2 = [i[1] for i in eq2]
        else:
            acts2 = [i[3] for i in eq2]
        if len(acts1) > 1:
            acts1 = acts1[binomial(1, 1-equibreak1)]
        else:
            acts1 = acts1[0]
        if len(acts2) > 1:
            acts2 = acts2[binomial(1, 1-equibreak2)]
        else:
            acts2 = acts2[0]
        # calculate fitnessgain
        strats = self.actstoStrats((acts1, acts2))
        fitnessgain = (self.payoffs[strats[0]][strats[1]], self.payoffs[strats[1]][strats[0]])
        # calculate information gain (whether from the point of the other player there is a unique profile and team reasoners and individual reasoners play different actions in it)
        isTeam2 = None
        if len(eq1) == 1:
            if eq1[0][0] != eq1[0][2]:
                if acts2 == eq1[0][0]:
                    isTeam2 = False
                elif acts2 == eq1[0][2]:
                    isTeam2 = True
        isTeam1 = None
        if len(eq2) == 1:
            if eq2[0][0] != eq2[0][2]:
                if acts1 == eq2[0][0]:
                    isTeam1 = False
                elif acts1 == eq2[0][2]:
                    isTeam1 = True
        return acts1, acts2, fitnessgain, isTeam2, isTeam1
        
       
    
if __name__ == '__main__':
    sPD = ['C', 'D']
    sHiLo = ['Hi', 'Lo']
    p1 = np.array([[2, 0],[3, 1.6]])
    stag = np.array([[4, 0],[3, 3]])
    h = np.array([[2, 0],[0, 1]])
    PD = game('PD', sPD, p1)
    print('PD team reasoning equilibrium =', PD.reason(0.33334))
    HiLo = game('HiLo', sHiLo, h)
    print('HiLo team reasoning equilibrium =', HiLo.reason(0.66667))
    Stag = game('Stag Hunt', ['Stag', 'Hare'], stag)
    print('Stag Hunt team reasoning equilibrium =', Stag.reason(0.8))
    print('\n\n')
    for i in range(1):
        print(PD.act(0.5, 0.1, 1, 1, 0.5, 0.5))
    
    
    
    
    
    
