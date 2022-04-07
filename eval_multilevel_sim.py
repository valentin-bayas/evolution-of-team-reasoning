#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:17:39 2022

@author: valentinbayas
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir('/Users/valentinbayas/Desktop/BA/Results/multilevel_sim_several_runs')


level_ratios = [2,4,8]
mutation_rates = [3,9,21]
learn = [False, True]

count = 0
for learning in learn:     # let learning run introspect or shallow
    for ratio in level_ratios:
        for mutation in mutation_rates:
            data = f'/Users/valentinbayas/Desktop/multilevel_sim_introspect=={learning},level_ratio=={ratio},mutation_rate=={mutation}.csv'
            sim = pd.read_csv(data)
            # sim = sim.dropna()
            # print(sim1.head)
            
            last = sim.loc[sim.loc[:,'Step']==599, :]
            # print(last.head)
            # print(last.mean(axis=0))
            # print('\ndoes not terminate:\n', len(last.loc[last.loc[:,'fix_gens']==2000, :]))
            
            
            #print('\nmean of ProbPD > 0.3334 (Team reasoners play C)\n', (last.loc[last.loc[:,'wPD']>=0.3334, :]).mean(axis=0))
            #print('\nmean of ProbPD < 0.3334 (Team reasoners play D)\n', (last.loc[last.loc[:,'wPD']<0.3334, :]).mean(axis=0))
            
            # print('\nmean of ProbPD > 0.8 (Team reasoners play C)\n', (last.loc[last.loc[:,'wPD']>=0.8, :]).mean(axis=0))
            # print('\nmean of ProbPD < 0.8 (Team reasoners play D)\n', (last.loc[last.loc[:,'wPD']<0.8, :]).mean(axis=0))
            print(sim.columns)
            name = f'introspect={learning}, level ratio={ratio}, mutation rate={mutation}'
            sim = sim.rename(columns={'teamProbabilityPD': 'probPD', 'teamProbabilityHi': 'probHi', 'extimatedProbPD':'estimated probPD', 'extimatedProbHi': 'estimated probHi'})
            obs = [['probPD', 'probHi'], ['estimated probPD', 'estimated probHi'], ['cooperation', 'Hi-play']]
            run_means = sim.groupby('Step').mean()
            for o in obs:
                run_means.loc[:,o].plot.line(title=f'{name}, all runs', subplots = True)
                plt.savefig(f'{name} all runs {o[0]} {o[1]}.png')
            
            
            # print('\nbelief in team reasoning in Hi Lo decreased\n', (last.loc[last.loc[:,'extimatedProbHi']>last.loc[:,'extimatedProbHi'], :]).mean(axis=0))
            # print((last.loc[last.loc[:,'extimatedProbHi']>last.loc[:,'extimatedProbHi'], :]).count())
            # look at the different runs
            #get the averages of all the values
            runs = {}
            for i in range(5):
                # split up by run and take the averages over steps (averages of the hole run per time)
                temp = sim.loc[sim.loc[:,'run']==i+1, :]
                runs[f'run{i+1}'] = temp.groupby('Step').mean()
                c = runs[f'run{i+1}']
                # get only 10 % of data points
                # c = c.loc[c.loc[:,'Step']%10==0]
                for o in obs:
                    c.loc[:,o].plot.line(title=f'{name}, run{i+1}', subplots = True)
                    plt.savefig(f'{name} run{i+1} {o[0]} {o[1]}.png')
            
            
            """
            probPD = []
            cooperate = []
            belief_in_team = []
            bins = []
            a = 10
            for i in range(a):
                bins.append(f'{(i+1)/a}')
                probPD.append((sim.loc[(i/a<= sim.loc[:,'wPD']) & (sim.loc[:,'wPD']<(i+1)/a), 'teamProbabilityPD']).mean(axis=0))
                cooperate.append((sim.loc[(i/a<= sim.loc[:,'wPD']) & (sim.loc[:,'wPD']<(i+1)/a), 'cooperation']).mean(axis=0))
                belief_in_team.append((sim.loc[(i/a<= sim.loc[:,'wPD']) & (sim.loc[:,'wPD']<(i+1)/a), 'extimatedProbPD']).mean(axis=0))
            
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            X = np.arange(len(cooperate))
            # ax.bar(bins,cooperate)
            plt.xticks(X, bins)
            ax.bar(X - 0.60, probPD, color = 'b', width = 0.25)
            ax.bar(X - 0.35, cooperate, color = 'g', width = 0.25)
            ax.bar(X - 0.10, belief_in_team, color = 'r', width = 0.25)
            ax.legend(labels=['probPD', 'cooperation', 'estimated probPD'])
            plt.ylabel("observed variables in percent")
            plt.xlabel("wPD")
            plt.show()
            
            probHi = []
            Hi = []
            belief_in_teamHi = []
            bins = []
            a = 10
            for i in range(a):
                bins.append(f'{(i+1)/a}')
                probHi.append((sim.loc[(i/a<= sim.loc[:,'wHi']) & (sim.loc[:,'wHi']<(i+1)/a), 'teamProbabilityHi']).mean(axis=0))
                Hi.append((sim.loc[(i/a<= sim.loc[:,'wHi']) & (sim.loc[:,'wHi']<(i+1)/a), 'Hi-play']).mean(axis=0))
                belief_in_teamHi.append((sim.loc[(i/a<= sim.loc[:,'wHi']) & (sim.loc[:,'wHi']<(i+1)/a), 'extimatedProbHi']).mean(axis=0))
            
            
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            X = np.arange(len(cooperate))
            # ax.bar(bins,cooperate)
            plt.xticks(X, bins)
            ax.bar(X - 0.60, probHi, color = 'b', width = 0.25)
            ax.bar(X - 0.35, Hi, color = 'g', width = 0.25)
            ax.bar(X - 0.10, belief_in_teamHi, color = 'r', width = 0.25)
            ax.legend(labels=['probHi', 'Hi-play', 'estimated probHi'])
            plt.ylabel("observed variables in percent")
            plt.xlabel("wHi")
            plt.show()
            
            """
            
            
            # Number of games that is Hi Lo does not vary here from game to game
            # observe learning?





