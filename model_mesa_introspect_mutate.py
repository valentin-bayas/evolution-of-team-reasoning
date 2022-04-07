#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 17:28:14 2022

@author: valentinbayas
"""
# with possible introspection and mutation

from class_game2 import game
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import binomial
import random
import scipy.stats as stats

class EvenError(Exception):
    """Only use even numbers!"""
    pass
    

class Beta():
    """Used to update beliefs about probability of team reasoning"""
    def __init__(self):
        self.b = 1
        self.a = 1

    def point(self):
        return self.a/(self.a + self.b)
        
    def update(self,x: bool):
        "x is wether team reasoning was observed"
        if x == None:
            return
        if x == True:
            x = 1
        else:
            x = 0
        self.a += x
        self.b += 1 - x        
        
    def plot(self):
        span = np.linspace(0,1,200)
        density = stats.beta.pdf(span,self.a,self.b)
        plt.plot(span,density) 

# evaluation metrics
def estG1(agent):
    return agent.betaG1.point()
def estG2(agent):
    return agent.betaG2.point()  
def teamProbG1max(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return max([i.probG1 for i in agents])
def teamProbG2max(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return max([i.probG2 for i in agents])
def teamProbG1min(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return min([i.probG1 for i in agents])
def teamProbG2min(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return min([i.probG2 for i in agents])
def teamProbability1(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return sum([i.probG1 for i in agents])/len(agents)
def teamProbability2(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return sum([i.probG2 for i in agents])/len(agents)
def typecount(model):
    types = [cell[0].unique_id for cell in model.grid.coord_iter()]
    temp = []
    for i in types:
        if i not in temp:
            temp.append(i)
    return len(temp)
def actG1(model):
    """How many agents play C in PD"""
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return len([i for i in agents if i.G1lastAct == model.game1.strat[0]])/len(agents)
def actG2(model):
    """How many agents play C in PD"""
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return len([i for i in agents if i.G2lastAct == model.game2.strat[0]])/len(agents)
def perceivedProbability1(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return sum([i.betaG1.point() for i in agents])/len(agents)
def perceivedProbability2(model):
    agents = [cell[0] for cell in model.grid.coord_iter()]
    return sum([i.betaG2.point() for i in agents])/len(agents)

# agent in our model
class Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # individual probability of team reasoning in HiLo and PD, dependent on w
        self.probG1 = False
        self.probG2 = False
        self.fitness = 0
        # estimation of w
        self.betaG1 = Beta()
        self.betaG2 = Beta()
        # probability of choosing action 1 in a game with two team reasoning equilibria
        self.G1equibreak = False
        self.G2equibreak = False
        self.G1lastAct = False
        self.G2lastAct = False
        # def move(self):
           # should agents move around in the model? or maybe rather be reshuffled?
        
    def play(self, opponent: Agent):
        # get the action of the agent
        G1type1 = binomial(1, self.probG1)
        G1type2 = binomial(1, opponent.probG1)
        G2type1 = binomial(1, self.probG2)
        G2type2 = binomial(1, opponent.probG2)
        G1act1, G1act2, G1fitnessgain, G1oppisTeam, G1selfisTeam = self.model.game1.act(self.betaG1.point(), opponent.betaG1.point(), G1type1, G1type2, self.G1equibreak, opponent.G1equibreak)
        G2act1, G2act2, G2fitnessgain, G2oppisTeam, G2selfisTeam = self.model.game2.act(self.betaG2.point(), opponent.betaG2.point(), G2type1, G2type2, self.G2equibreak, opponent.G2equibreak)
        # add fitnessgain
        self.fitness += G1fitnessgain[0] + G2fitnessgain[0]
        opponent.fitness += G1fitnessgain[1] + G2fitnessgain[1]
        self.G1lastAct = G1act1
        opponent.G1lastAct = G1act2             #############################
        self.G2lastAct = G2act1
        opponent.G2lastAct = G2act2
        # possible introspection: update beliefs with own type in last round
        if self.model.introspect == True:
            self.betaG1.update(G1type1)
            opponent.betaG1.update(G1type2)
            self.betaG2.update(G2type1)
            opponent.betaG2.update(G2type2)
        self.betaG1.update(G1oppisTeam)
        opponent.betaG1.update(G1selfisTeam)
        self.betaG2.update(G2oppisTeam)
        opponent.betaG2.update(G2selfisTeam)
        
        
    def get_right(self):
        agent = self
        return self.model.get_right(agent)
        
    def step(self):
        # reset fitness after x rounds
        if self.model.counter % 5 == 0:
            self.fitness = 0
        if self.pos[1] % 2 == 0:
            self.play(self.get_right())          
        
  
class PopulationModel(Model):
    def __init__(self, Nsqrt: int, game1: game, game2: game, w1: float, w2: float, introspect: bool, mutationRate: int):
        # mutationRate specifies after how many generations one individual mutates should be given in relation to population
        if Nsqrt % 2 != 0:
            raise EvenError("Only use even numbers for sqrt of N!")
        self.game1 = game1
        self.game2 = game2
        self.num_agents = Nsqrt**2
        self.w1 = w1
        self.w2 = w2
        self.introspect = introspect
        self.mutationRate = mutationRate
        self.grid = SingleGrid(Nsqrt, Nsqrt, True)
        self.counter = 0
        self.id = 0
        self.schedule = RandomActivation(self)      # do the agents play in a random order after another or simultaneously?
        # find parameters of betavariate functions that generates values for w with given means
        self.beta1 = 1/w1 - 1
        self.beta2 = 1/w2 - 1
        betaEquibreak = 0.5
        self.betabreak = 1/betaEquibreak-1
        for i in range(self.num_agents):
            a = self.add_agent()
            # Add the agent to a random grid cell
            self.grid.position_agent(a, 'random', 'random')
        self.datacollector = DataCollector(
            model_reporters={"teamprob1": teamProbability1, "teamprob2": teamProbability2, "w1": "w1", "w2": "w2", "typecount": typecount,
                             "actG1": actG1, "actG2": actG2,"perceivedProbability1": perceivedProbability1, "perceivedProbability2": perceivedProbability2,
                             'teamProbG1max': teamProbG1max, 'teamProbG2max': teamProbG2max, 'teamProbG1min': teamProbG1min, 'teamProbG2min': teamProbG2min},
            agent_reporters={"Fitness": "fitness", 'probPD': 'probG1', 'probHi': 'probG2', 'estPD': estG1, 'estHi': estG2}
        )
        
    def add_agent(self):    
        a = Agent(self.id, self)
        self.id += 1
        self.schedule.add(a)
        a.probG1 = random.betavariate(1, self.beta1)
        mean = random.betavariate(1, self.beta1)
        a.betaG1.b = 1/mean - 1
        a.probG2 = random.betavariate(1, self.beta2)
        mean = random.betavariate(1, self.beta2)
        a.betaG2.b = 1/mean - 1
        a.G1equibreak = random.betavariate(1, self.betabreak)   # every player is breaking an equilibrium with a random probability
        a.G2equibreak = random.betavariate(1, self.betabreak)
        return(a)
            
    def replaceAgent(self, dead, replicating):
        pos = dead.pos
        keys = vars(replicating).keys()
        for elem in keys:
            vars(dead)[f'{elem}'] = vars(replicating)[f'{elem}']
        dead.pos = pos
        
    def roulette_wheel_selection(self):
        """selects an individual to be replicated based on fitness"""
        max_value = sum(cell[0].fitness for cell in self.grid.coord_iter())   # calculates sum of fitness in pop
        pick = random.uniform(0, max_value)
        current = 0
        for cell in self.grid.coord_iter():
            current += cell[0].fitness
            if current >= pick:
                return cell[0]
        
    def swap_pos(self, a1, a2):
        """makes two agents swap positions"""
        pos1 = a1.pos
        pos2 = a2.pos
        self.grid.remove_agent(a1)
        self.grid.move_agent(a2, pos1)
        self.grid.position_agent(a1, pos2)
        
    def get_right(self, agent):
        """returns the agent right of another agent"""
        p = agent.pos
        list = [a for a in self.grid.get_neighbors(p, False, False)]
        for i in list:
            if (i.pos[0]==p[0] and i.pos[1]==p[1]+1):
                return(i)
     
    def avFitness(self):  # helper metric for multilevel model
        agents = [cell[0] for cell in self.grid.coord_iter()]
        return sum([i.fitness for i in agents])/len(agents)
    
    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.counter += 1
        self.schedule.step()
        #print()
        #for cell in practice.grid.coord_iter():
         #   print(cell[0].unique_id, cell[0].fitness)
        a = random.choice([cell[0] for cell in self.grid.coord_iter()])
        b = self.roulette_wheel_selection()
        self.replaceAgent(a, b)
        # mutation
        if self.counter % self.mutationRate == 0:
            a = self.add_agent()
            x = random.sample([cell[0] for cell in self.grid.coord_iter()], 1)[0]
            # print('\nvalues of mutating agent before', x.unique_id,', ', round(x.probG1, 2),', ', round(x.probG2, 2))
            # ide = x.unique_id
            x.unique_id, x.probG1, x.probG2, x.betaG1, x.betaG2 = a.unique_id, a.probG1, a.probG2, a.betaG1, a.betaG2
            self.schedule.remove(a)
            # print('values of mutating agent after', x.unique_id, ', ', round(x.probG1, 2),', ', round(x.probG2, 2), '\n')
            # self.schedule.remove(x)
            # self.grid.remove_agent(x)
            # self.grid.position_agent(a, 'random', 'random')
            # print(f'\nmutate: {ide} to {a.unique_id}\n')
            # problems: fitness of mutant is reset to 0, use actual mutation instead of new indiv
            
        # reshuffle the population, specify how often based on pop_size (loop)
        for i in range(int(self.num_agents/2)):
            (x, y) = random.sample([cell[0] for cell in self.grid.coord_iter()], 2)
            self.swap_pos(x, y)

# only input positive values as payoffs (roulette wheel selection)
if __name__ == '__main__':   
    game1 = game('PD',['C', 'D'], np.array([[2, 0],[3, 1.6]]))
    game2 = game('Hi Lo', ['Hi', 'Lo'], np.array([[2, 0],[0, 1]]))
    
    practice = PopulationModel(2, game1, game2, 0.75, 0.6, True, 3)
    for cell in practice.grid.coord_iter():
        print(cell[0].unique_id, cell[0].fitness)
    for i in range(25):
        practice.step()
        print()
        for cell in practice.grid.coord_iter():
            print(cell[0].unique_id, cell[0].fitness, cell[0].G1lastAct, cell[0].G2lastAct, cell[0].pos)
        print("cooperation:", actG1(practice))
     
# cell[0].betaG1.point(), cell[0].betaG2.point()
# print(actG2(practice))        
# agents = [cell[0] for cell in practice.grid.coord_iter()]
# print(sum([i.probG1 for i in agents])/len(agents), sum([i.probG2 for i in agents])/len(agents))


       