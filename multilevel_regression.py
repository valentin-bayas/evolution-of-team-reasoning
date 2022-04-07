#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:52:51 2022

@author: valentinbayas
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = [[1,2,3], [1,2,9], [1,2,21], [1,4,3], [1,4,9], [1,4,21], [1,8,3], [1,8,9], [1,8,21],
     [0,2,3], [0,2,9], [0,2,21], [0,4,3], [0,4,9], [0,4,21], [0,8,3], [0,8,9], [0,8,21]]

y = [0.58, 0.11, 0.19, 0.60, 0.55, 0.02, 0.23, 0.37, 0.26, 0.02, 0.00, 0.00, 0.01, 0.03, 0.00, 0.19, 0.01, 0.00]

x, y = np.array(x), np.array(y)
introspectX = x[0:9]
introspectXlevel = [x[1] for x in introspectX]
introspectXmutate = [x[2] for x in introspectX]
introspectY = y[0:9]
shallowX = x[9:]
shallowXlevel = [x[1] for x in shallowX]
shallowXmutate = [x[2] for x in shallowX]
shallowY = y[9:]

def reg(x,y):
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_, '\n')

print('all values:')
reg(x,y)
print('introspect:')
reg(introspectX, introspectY)
print('shallow:')
reg(shallowX, shallowY)


"""
plt.plot(shallowXlevel, shallowY,'ro', markersize=5.0)
plt.plot(introspectXlevel, introspectY,'bo')
plt.axis([0, 22, -0.01, 1.0])
plt.show()

plt.plot(shallowXmutate, shallowY,'ro', markersize=5.0)
plt.plot(introspectXmutate, introspectY,'bo')
plt.axis([0, 22, -0.01, 1.0])
plt.show()

fig = plt.figure()
"""

# syntax for 3-D projection
ax = plt.axes(projection ='3d')
wid =  [1]*9
dep = [2]*9
introspectXmutate = [x-2 for x in introspectXmutate]

z = [0]*9
# print(shallowXlevel,'\n', shallowXmutate,'\n', z,'\n', wid, '\n',dep,'\n', shallowY)
# plotting
ax.set_ylim(22, 0)
ax.bar3d(shallowXlevel, shallowXmutate, z, wid, dep, shallowY, 'r')
ax.bar3d(introspectXlevel, introspectXmutate, z, wid, dep, introspectY, 'b')
ax.set_title('introspect = blue, shallow = red')
ax.set_xlabel('level ratio')
ax.set_ylabel('mutation rate')
ax.set_zlabel('cooperation')
plt.show()


"""
level_ratios = [2,4,8]
mutation_rates = [3,9,21]
introspect = [True, False]


count = 0
for learning in introspect:     # let learning run introspect or shallow
    for ratio in level_ratios:
        for mutation in mutation_rates:
            count += 1
            data = f'/Users/valentinbayas/Desktop/multilevel_sim_introspect=={learning},level_ratio=={ratio},mutation_rate=={mutation}.csv'
            sim = pd.read_csv(data)
            last = sim.loc[sim.loc[:,'Step']==599, :]
            if count == 1:
                df = last.mean(axis=0)
                introspect = last.mean(axis=0)
            else:
               df = pd.concat([df, last.mean(axis=0)], axis=1)
               if learning == True:
                   introspect = pd.concat([introspect, last.mean(axis=0)], axis=1)
               if learning == False:
                   if count == 10:
                       shallow = last.mean(axis=0)
                   else:
                       shallow = pd.concat([shallow, last.mean(axis=0)], axis=1)
                       
df, introspect, shallow = df.T, introspect.T, shallow.T
print('\n')
print('whole sim\n', df.mean(axis=0), '\n')
print('introspect\n',introspect.mean(axis=0), '\n')
print('shallow\n',shallow.mean(axis=0), '\n')
     
print()          
count = 0
for learning in introspect:     # let learning run introspect or shallow
    for ratio in level_ratios:
        for mutation in mutation_rates:
            count += 1
            data = f'/Users/valentinbayas/Desktop/multilevel_sim_introspect=={learning},level_ratio=={ratio},mutation_rate=={mutation}.csv'
            sim = pd.read_csv(data)
            first = sim.loc[sim.loc[:,'Step']==599, :]
            if count == 1:
                print(first.loc[first.loc[:,'teamProbabilityPD']>0.95, ['wPD', 'run','AgentID', 'extimatedProbPD']])

# check whether populations with high wPDs can dominate the group selection process if they occur


""" 


            