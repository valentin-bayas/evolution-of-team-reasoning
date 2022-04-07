#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:24:11 2022

@author: valentinbayas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


level_ratios = [2,4,8]
mutation_rates = [3,9,21]
learn = [False, True]

count = 0
for learning in learn:     # let learning run introspect or shallow
    earlycopchange = []
    firsthighcopcount = 0
    nexhighcopcount = 0
    firstcop = []
    firstcopmax = []
    nexcopmax = []
    for ratio in level_ratios:
        for mutation in mutation_rates:
            data = f'/Users/valentinbayas/Desktop/multilevel_sim_introspect=={learning},level_ratio=={ratio},mutation_rate=={mutation}.csv'
            sim = pd.read_csv(data)
            sim = sim.loc[sim.loc[:,'Step']>=1,:]
            
            # print(f'\n{learning}, level_ratio = {ratio}, mutation_rate = {mutation}')
            firstgencop = sim.loc[sim.loc[:,'Step']==1,'cooperation']
            # print('cooperation in first generation', (firstgencop).sum()/firstgencop.size)
            firstcop.append(firstgencop.sum()/firstgencop.size)
            firstcopmax.append(firstgencop.max())
            nextgencop = 0
            # number of populations with average cooperation above 95%
            firsthighcopcount += firstgencop.loc[firstgencop.loc[:]>=0.95].count()
            
            nex = 4
            nexgencop = sim.loc[sim.loc[:,'Step']==nex+1,'cooperation']
            nexcopmax.append(nexgencop.max())
            nexhighcopcount += nexgencop.loc[nexgencop.loc[:]>=0.95].count()
            for i in range(nex):
                cop = sim.loc[sim.loc[:,'Step']==i+2,'cooperation']
                nextgencop += cop.sum()/cop.size
            # print(f'cooperation in next {nex} generations', nextgencop/nex)
            earlycopchange.append(nextgencop/nex - firstgencop.sum()/firstgencop.size)
    print(f"average cooperation in first generation: {sum(firstcop)/len(firstcop)}")
    print(f"maximum cooperation in first generation: {sum(firstcopmax)/len(firstcopmax)}")
    print('number of populations with cooperation rates higher than 95%', firsthighcopcount)
    print(f"difference in average cooperation between first and next {nex} rounds: {round(sum(earlycopchange)/len(earlycopchange), 3)}")
    print(f"maximum cooperation in {nex+1}th generation: {sum(nexcopmax)/len(nexcopmax)}")
    print(f'number of populations with cooperation rates higher than 95% in gen {nex+1}\n', nexhighcopcount)
    
    
    