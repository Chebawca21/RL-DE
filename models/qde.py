import numpy as np
from itertools import product
from models.de import DifferentialEvolution
from qlearning import QLearning

class QDE(DifferentialEvolution):
    def __init__(self, dimension, func, population_size, mutation_type='rand', p=0.1, selection_strategy='boltzmann', actions='cde', archive_size=None):
        self.D = dimension
        self.func = func
        self.population_size = population_size
        self.rank_greediness_factor = 3
        if actions == 'qlde':
            actions = [(0.4, 0.7), (0.6, 0.7), (0.8, 0.7), (0.4, 0.9), (0.6, 0.9), (0.8, 0.9), (0.9, 0.9)]
        else:
            self.F_pool = [0.5, 0.8, 1.0]
            self.cr_pool = [0.0, 0.5, 1.0]
            actions = list(product(self.F_pool, self.cr_pool))
        states = [0]
        self.state = 0
        self.qlearning = QLearning(states, actions, selection_strategy=selection_strategy)
        self.mutation_type = mutation_type
        self.p = int(p * population_size)
        if archive_size is None:
            self.archive_size = population_size
        else:
            self.archive_size = archive_size
        self.func_evals = 0
        self.best_individual = None
        self.best_score = np.inf
        self.population = self.initializePopulation()
        self.archive = []
        self.evaluate_population()

    def step(self):
        new_population = np.zeros_like(self.population)
        new_scores = np.zeros(self.population_size)
        prs = self.get_rank_probabilities()

        for i in range(self.population_size):
            F, cr = self.qlearning.get_action(self.state)
            mutant = self.mutation(F, self.mutation_type, current=i, p=self.p, prs=prs)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                self.qlearning.update_qtable(self.state, self.state, (F, cr), 1)
                new_population[i] = candidate
                new_scores[i] = candidate_score
                self.archive.append(self.population[i])
            else:
                self.qlearning.update_qtable(self.state, self.state, (F, cr), 0)
                new_population[i] = self.population[i]
                new_scores[i] = self.scores[i]

        self.resize_archive()
        self.population = new_population
        self.scores = new_scores
        self.update_best_score()