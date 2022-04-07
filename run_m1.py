#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:36:43 2022

@author: valentinbayas
"""
from mesa import Agent, Model
from class_game2 import game
from mesa.time import RandomActivation
from model_mesa_introspect_mutate import PopulationModel
import numpy as np
import pandas as pd


if __name__ == '__main__':   
    game1 = game('PD',['C', 'D'], np.array([[2, 0],[3, 1.6]]))
    game2 = game('Hi Lo', ['Hi', 'Lo'], np.array([[2, 0],[0, 1]]))
    # simulate populations with different values of w in both games in 0.1 intervals
    count= 0
    steps = 25
    for w1 in range(1,steps):
        for w2 in range(1,steps):
            count += 1
            practice = PopulationModel(6, game1, game2, w1/steps, w2/steps, True, 1000)
            for i in range(900):
                practice.step()
            model_vars = practice.datacollector.get_model_vars_dataframe()
            agent_vars = practice.datacollector.get_agent_vars_dataframe()
            if count == 1:
                resultsM = model_vars
                resultsA = agent_vars
            else:
                resultsM = pd.concat([resultsM, model_vars])
                resultsA = pd.concat([resultsA, agent_vars])
            # print(teamprob.loc[0,:])
            
            print(f'....{count}/576')
    resultsM = resultsM.round(2)
    resultsA = resultsA.round(2)
            # print('\n\n\n',resultsM)

    # --> also track actions and perceived probs
    
    with open('/Users/valentinbayas/Desktop/m1_introspect_Model.csv', 'w') as file:
        resultsM.to_csv(file, sep=',', encoding='utf-9')
    with open('/Users/valentinbayas/Desktop/m1_introspect_Agents.csv', 'w') as file:
        resultsA.to_csv(file, sep=',', encoding='utf-9')