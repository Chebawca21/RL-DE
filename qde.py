import numpy as np
from itertools import product
from de import DifferentialEvolution
from qlearning import QLearning
from SO_BO.CEC2022 import cec2022_func

class QDE(DifferentialEvolution):
    def __init__(self, dimension, func_num, population_size, mutation_type='best'):
        self.D = dimension
        self.func_num = func_num
        self.cec = cec2022_func(self.func_num)
        self.population_size = population_size
        self.F_pool = [0.5, 0.8, 1.0]
        self.cr_pool = [0.0, 0.5, 1.0]
        actions = list(product(self.F_pool, self.cr_pool))
        states = [0]
        self.state = 0
        self.qlearning = QLearning(states, actions)
        self.mutation_type = mutation_type
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()
    
    def mutation(self, current, F):
        idxs = np.random.randint(0, self.population_size, 3)
        best_id = np.argmin([self.scores[idxs[0]], self.scores[idxs[1]], self.scores[idxs[2]]])
        mutant = self.difference(self.population[idxs[best_id]], self.population[idxs[1]], self.population[idxs[2]], F)
        return mutant

    def step(self):
        new_population = []
        new_scores = []

        for i in range(self.population_size):
            F, cr = self.qlearning.get_action(self.state)
            mutant = self.mutation(i, F)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                self.qlearning.update_qtable(self.state, self.state, (F, cr), 1)
                new_population.append(candidate)
                new_scores.append(candidate_score)
            else:
                self.qlearning.update_qtable(self.state, self.state, (F, cr), -0.5)
                new_population.append(self.population[i])
                new_scores.append(self.scores[i])

        self.population = new_population
        self.scores = np.array(new_scores)
        self.update_best_score()
