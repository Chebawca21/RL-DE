import numpy as np
from itertools import product
from de import DifferentialEvolution
from qlearning import QLearning

class QDE(DifferentialEvolution):
    def __init__(self, dimension, func, population_size, mutation_type='best', p=0.1, archive_size=None):
        self.D = dimension
        self.func = func
        self.population_size = population_size
        self.F_pool = [0.5, 0.8, 1.0]
        self.cr_pool = [0.0, 0.5, 1.0]
        actions = list(product(self.F_pool, self.cr_pool))
        states = [0]
        self.state = 0
        self.qlearning = QLearning(states, actions)
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
        new_population = []
        new_scores = []

        for i in range(self.population_size):
            F, cr = self.qlearning.get_action(self.state)
            if self.mutation_type == 'current-to-best' or self.mutation_type == 'current-to-rand':
                mutant = self.mutation(F, self.mutation_type, i)
            elif self.mutation_type == 'current-to-pbest':
                mutant = self.mutation(F, self.mutation_type, self.p)
            else:
                mutant = self.mutation(F, self.mutation_type)
            candidate = self.binary_crossover(mutant, self.population[i], cr)
            candidate_score = self.evaluate(candidate)
            if candidate_score <= self.scores[i]:
                self.qlearning.update_qtable(self.state, self.state, (F, cr), 1)
                new_population.append(candidate)
                new_scores.append(candidate_score)
                self.archive.append(self.population[i])
            else:
                self.qlearning.update_qtable(self.state, self.state, (F, cr), -0.5)
                new_population.append(self.population[i])
                new_scores.append(self.scores[i])

        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]
        self.population = new_population
        self.scores = np.array(new_scores)
        self.update_best_score()
