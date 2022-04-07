#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:29:39 2022

@author: valentinbayas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde

level_ratios = [2,4,8]
mutation_rates = [3,9,21]
learn = [False, True]

count = 0
for learning in learn:     # let learning run introspect or shallow
    for ratio in level_ratios:
        for mutation in mutation_rates:
            data = f'/Users/valentinbayas/Desktop/multilevel_sim_introspect=={learning},level_ratio=={ratio},mutation_rate=={mutation}.csv'
            sim = pd.read_csv(data)
            sim = sim.loc[sim.loc[:,'Step']>=1,:]
            
            runs = sim.groupby('run')           
            print(sim.columns)
            for name, run in runs:
                print(f'\n{learning}, level_ratio = {ratio}, mutation_rate = {mutation}', name)
                firstgencop = run.loc[run.loc[:,'Step']==1,'cooperation']
                print('cooperation in first generation', (firstgencop).sum()/firstgencop.size)
                nextgencop = 0
                nex = 10
                for i in range(nex):
                    cop = run.loc[run.loc[:,'Step']==i+2,'cooperation']
                    nextgencop += cop.sum()/cop.size
                print(f'cooperation in next {nex} generations', nextgencop/nex)
                
                
                sns.kdeplot(data = run, x="Step", y="cooperation", cmap="Reds", shade=True, clip=((0,600),(0,1)))
                plt.show()