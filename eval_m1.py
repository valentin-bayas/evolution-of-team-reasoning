#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 12:27:49 2022

@author: valentinbayas
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = '/Users/valentinbayas/Desktop/m1_introspect_Model.csv'
sim1 = pd.read_csv(data)
sim1 = sim1.rename(columns={'Unnamed: 0': 'Step'}, errors="raise")
# print(sim1.iloc[1,:])
first = sim1.loc[sim1.loc[:,'Step']==0, :]
last = sim1.loc[sim1.loc[:,'Step']==899, :]

print('mean min w1 >= 0.8\n', first.loc[first.loc[:,'w1']>=0.8,'teamProbG1min'].mean(), '\n')
print('mean max w1 <= 0.3\n', first.loc[first.loc[:,'w1']<=0.3,'teamProbG1max'].mean(), '\n')

print(sim1.index)
print(last.mean(axis=0))
# print('\ndoes not terminate:\n', len(sim.loc[sim.loc[:,'fix_gens']==2000, :]))


print('\nmean of ProbPD > 0.3334 (Team reasoners play C)\n', (last.loc[last.loc[:,'w1']>=0.3334, :]).mean(axis=0))
print('\nmean of ProbPD < 0.3334 (Team reasoners play D)\n', (last.loc[last.loc[:,'w1']<0.3334, :]).mean(axis=0))

print('\nmean of ProbPD > 0.8 (Team reasoners play C)\n', (last.loc[last.loc[:,'w1']>=0.8, :]).mean(axis=0))
print('\nmean of ProbPD < 0.8 (Team reasoners play D)\n', (last.loc[last.loc[:,'w1']<0.8, :]).mean(axis=0))

# coop = first.loc[first.loc[:,'w1']>=0.8, :]
# print('\nnumber of populations where wPD >= 0.8 and probPD < 0.85\n', (coop.loc[coop.loc[:,'teamprob1']<0.85, :]).shape[0])

# print('\nbelief in team reasoning in Hi Lo decreased\n', (last1.loc[last.loc[:,'perceivedProbability2']>last.loc[:,'perceivedProbability2'], :]).mean(axis=0))
# print((last.loc[last.loc[:,'BperceivedProbability2']>last.loc[:,'perceivedProbability2'], :]).count())

# does the type with the lowest initial team reasoning probability spread in a population? compare min in gen0 with mean in last gen
idx = list(range(576))
mins = first.loc[:,'teamProbG1min'].copy()     # first.loc[:,'w1']>=0.3334
mins.index = idx
means = last.loc[:, 'teamprob1'].copy()
means.index = idx
teamPDsuccess = means.subtract(mins, fill_value=np.nan)
print('mean difference between min in gen0 and mean in gen899 is ', round(teamPDsuccess.mean(),2))

# print(teamPDsuccess[teamPDsuccess > 0.5])

# only the populations with wPD > 1/3
mins = first.copy()     # first.loc[:,'w1']>=0.3334
mins.index = idx
means = last.copy()
means.index = idx
teamPDsuccess = means['teamprob1'].subtract(mins['teamProbG1min'], fill_value=np.nan)
# print('\n', teamPDsuccess, '\n')
mins['teamPDsuccess'] = teamPDsuccess
print('\nmean difference between min in gen0 and mean in gen899 for values of wPD > 1/3 is ', round((mins.loc[mins.loc[:,'w1']>=0.3334,'teamPDsuccess']).mean(),2))
print('\nnumber of individuals where mean is more than 10% greater that min ', (mins.loc[mins.loc[:,'teamPDsuccess']>0.1,'teamPDsuccess']).count())



probPD = []
probPDmins = []
cooperate = []
belief_in_team = []
bins = []
a = 10
for i in range(a):
    bins.append(f'{(i+1)/a}')
    probPD.append((last.loc[(i/a<= last.loc[:,'w1']) & (last.loc[:,'w1']<(i+1)/a), 'teamprob1']).mean(axis=0))
   # probPDmins.append((last.loc[(i/a<= last.loc[:,'w1']) & (last.loc[:,'w1']<(i+1)/a), 'teamprob1']).min(axis=0))
    cooperate.append((last.loc[(i/a<= last.loc[:,'w1']) & (last.loc[:,'w1']<(i+1)/a), 'actG1']).mean(axis=0))
    belief_in_team.append((last.loc[(i/a<= last.loc[:,'w1']) & (last.loc[:,'w1']<(i+1)/a), 'perceivedProbability1']).mean(axis=0))

# probPD = [round(x,2) for x in probPD]
# print('\nmeans:', probPD, '\nmins:', probPDmins, '\nmeans - mins:', [x1 - x2 for (x1, x2) in zip(probPD, probPDmins)])

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
plt.xlabel("w1")
plt.show()



probHi = []
Hi = []
belief_in_teamHi = []
bins = []
a = 10
for i in range(a):
    bins.append(f'{(i+1)/a}')
    probHi.append((last.loc[(i/a<= last.loc[:,'w2']) & (last.loc[:,'w2']<(i+1)/a), 'teamprob2']).mean(axis=0))
    Hi.append((last.loc[(i/a<= last.loc[:,'w2']) & (last.loc[:,'w2']<(i+1)/a), 'actG2']).mean(axis=0))
    belief_in_teamHi.append((last.loc[(i/a<= last.loc[:,'w2']) & (last.loc[:,'w2']<(i+1)/a), 'perceivedProbability2']).mean(axis=0))


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
plt.xlabel("w2")
plt.show()

# is probPD always as low as it can be in a bin


"""
print('\nx <= 0.5 (Team reasoners play C)\n', (last.loc[(last.loc[:,'x']<=0.5), :]).mean(axis=0))

print('\n0.7 <= x <= 0.9\n', (last.loc[(last.loc[:,'x']<=0.9) & (last.loc[:,'x']>=0.7), :]).mean(axis=0))

print('\nh == 0\n', (last.loc[last.loc[:,'h']==0, :]).mean(axis=0))
print('\nh == 1\n', (last.loc[last.loc[:,'h']==1, :]).mean(axis=0))



print('\nmean of ProbPD > 0.66 and x <= 0.5 (Team reasoners play C)\n', (last.loc[(last.loc[:,'ProbPD']>=0.7) & (last.loc[:,'x']<=0.5), :]).mean(axis=0))
    
    
print('\nmean of ProbPD < 0.66 and x <= 0.5 (Team reasoners play C)\n', (last.loc[(last.loc[:,'ProbPD']<0.7) & (last.loc[:,'x']<=0.5), :]).mean(axis=0))
"""
