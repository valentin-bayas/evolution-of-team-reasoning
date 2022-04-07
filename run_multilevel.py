#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 19:51:01 2022

@author: valentinbayas
"""
# make a population model of different populations in mesa (agent = populationmodel)
from class_game2 import game
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from numpy.random import binomial
import random
import scipy.stats as stats
from model_mesa_introspect_mutate import PopulationModel, typecount, teamProbability1, teamProbability2, perceivedProbability1, perceivedProbability2, actG1, actG2 


def poptypecount(model):
    types = [pop.unique_id for pop in model.populations]
    temp = []
    for i in types:
        if i not in temp:
            temp.append(i)
    return len(temp)

def typecountR(model):
    return typecount(model.population)
def wPD(model):
    return round(model.population.w1, 3)
def wHi(model):
    return round(model.population.w2, 3)
def teamProbability1R(model):
    return round(teamProbability1(model.population),3)
def teamProbability2R(model):
    return round(teamProbability2(model.population),3)
def perceivedProbability1R(model):
    return round(perceivedProbability1(model.population), 3)
def perceivedProbability2R(model):
    return round(perceivedProbability2(model.population), 3)
def actG1R(model):
    return round(actG1(model.population), 3)
def actG2R(model):
    return round(actG2(model.population), 3)
def fitness(model):
    return round(model.popFitness, 3)


class EvenError(Exception):
    """Only use even numbers!"""
    pass

# evaluation metrics
def multilevelFitness(model):
   return sum([pop.popFitness for pop in model.populations])/len(model.populations)

# agent in our model
class Population(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.popFitness = 0
        # self.population = None
        
    def step(self):
        # reset fitness after x rounds
        # if self.model.counter % 5 == 0:
          #  self.fitness = 0
        for i in range(self.model.levelRatio):
            self.population.step()
        self.popFitness = self.population.avFitness()     
        
  
class MultilevelModel(Model):
    def __init__(self, popNumber: int, levelRatio: int, Nsqrt: int, game1: game, game2: game, introspect: bool, mutationRate: int):
        # mutationRate specifies after how many generations one individual mutates should be given in relation to population
        # level ratio is the number of generations the population go through between one generation at the group level
        self.popNumber = popNumber
        self.levelRatio = levelRatio
        if Nsqrt % 2 != 0:
            raise EvenError("Only use even numbers for sqrt of N!")
        self.game1 = game1
        self.game2 = game2
        self.Nsqrt = Nsqrt
        self.introspect = introspect
        self.mutationRate = mutationRate
        self.populations = []
        self.schedule = RandomActivation(self)
        for i in range(self.popNumber):
            a = Population(i, self)
            self.schedule.add(a)
            self.populations.append(a)
            w1 = random.random()
            w2 = random.random()
            a.population = PopulationModel(self.Nsqrt, self.game1, self.game2, w1, w2, self.introspect, self.mutationRate)


        self.datacollector = DataCollector(
             model_reporters={'poptypecount': poptypecount}, agent_reporters={
                 'fitness': fitness, 'typecount': typecountR, 'wPD': wPD, 'wHi': wHi,
                 'teamProbabilityPD': teamProbability1R, 'teamProbabilityHi': teamProbability2R, 'estimatedProbPD': perceivedProbability1R, 
                 'estimatedProbHi': perceivedProbability2R, 'cooperation': actG1R, 'Hi-play': actG2R}
        )
        
    def replaceAgent(self, dead, replicating):
        keys = vars(replicating).keys()
        for elem in keys:
            vars(dead)[f'{elem}'] = vars(replicating)[f'{elem}']
        
    def roulette_wheel_selection(self):
        """selects an individual to be replicated based on fitness"""
        max_value = sum(pop.popFitness for pop in self.populations)   # calculates sum of fitness in pop
        pick = random.uniform(0, max_value)
        current = 0
        for pop in self.populations:
            current += pop.popFitness
            if current >= pick:
                return pop
        
    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()
        a = random.choice([pop for pop in self.populations])
        b = self.roulette_wheel_selection()
        self.replaceAgent(a, b)

            
if __name__ == '__main__':   
    game1 = game('PD',['C', 'D'], np.array([[2, 0],[3, 1.6]]))
    game2 = game('Hi Lo', ['Hi', 'Lo'], np.array([[2, 0],[0, 1]]))
    groupLgens = 600      
    pops = 40
    runs = 5
    level_ratios = [2,4,8]
    mutation_rates = [3,9,21]
    introspect = [True, False]
    
    for learning in introspect:     # let learning run introspect or shallow
        for ratio in level_ratios:
            for mutation in mutation_rates:
                
                for r in range(runs):
                    practice = MultilevelModel(pops, ratio, 4, game1, game2, learning, mutation)
                    for i in range(groupLgens):
                        practice.step()
                        print(f'{r+1}/{runs}.......{i+1}/{groupLgens}')
                        # get the agent reporters: typecount, w1, w2, averageProbPD, averageProbHi, averageEstProbPD, averageEstProbHi, actG1, actG2, fitness, variation?
                    agent_vars = practice.datacollector.get_agent_vars_dataframe()
                    model_vars = practice.datacollector.get_model_vars_dataframe()
                    Poptypecount = []
                    for count in model_vars.loc[:,'poptypecount']:
                        temp = [count]*pops
                        Poptypecount += temp
                    agent_vars['poptypecount'] = Poptypecount
                    #print(model_vars)
                    run = [r+1]*groupLgens*pops
                    agent_vars['run'] = run
                    # run = np.full((1,groupLgens),r+1)
                    # agent_vars['run'] = run.tolist()
                    if r == 0:
                        results = agent_vars.copy()
                    else:
                        results = pd.concat([results, agent_vars])
                    
                with open(f'/Users/valentinbayas/Desktop/multilevel_sim_introspect=={learning}, level_ratio=={ratio}, mutation_rate=={mutation}.csv', 'w') as file:
                    results.to_csv(file, sep=',', encoding='utf-9')
    
    # assess the effects of different level speeds. observe the same data as in model 1 but more process oriented
    # at 4 individuals and 7 populations a level ratio of 3 is appropriate for similar fixation speed
    # first run (multilevel_sim) with values (100, 3, 6, game1, game2, False, 10), 2000 gens
    # second run (multilevel_sim) with values (20, 4, 4, game1, game2, False, 10), 500 gens
    # third run (fast) with values (20, 2, 4, game1, game2, False, 10), 500 gens
    # it seems that some degree of cooperation at the beginning leads to the success of populations, but their cooperation deteriorates over time
    # --> lower levelRatio/ shorter simulation will give cooperation less time to deteriorate --> 
    # check whether cooperative populations really have a higher fitness in the beginning
    # making individuals introspect or raising the mutation rate could increase the probability of cooperative behavior spreading through multilevel selection
    # does mutation just give copies of individuals new properties, or does it create new individuals
    
    
    
    
    