import random
import copy
import numpy as np

class particleDE:
    def __init__(self, dimention, lower, upper):
        self.solution = np.array([random.uniform(lower[i], upper[i]) for i in range(dimention)])
        self.value = float('inf')

class algorithmDE:
    def __init__(self, dimention, populationSize, lower, upper, function,
                 maxIter, breakCriteria=None, F=0.8, C=0.9, seed=None):
        self.dimention = dimention
        self.populationSize = populationSize
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.function = function
        self.maxIter = maxIter
        self.breakCriteria = breakCriteria
        self.F = F
        self.C = C
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.particles = [particleDE(dimention, lower, upper) for _ in range(populationSize)]
        self.bestParticle = copy.deepcopy(self.particles[0])

    def run(self):
        for p in self.particles:
            p.value = self.function(p.solution)

        for iter in range(self.maxIter):
            for i in range(self.populationSize):
                r1, r2, r3 = random.sample(range(self.populationSize), 3)
                mutant = self.particles[r1].solution + self.F * (
                    self.particles[r2].solution - self.particles[r3].solution)

                mutant = np.clip(mutant, self.lower, self.upper)

                trial = np.copy(self.particles[i].solution)
                for j in range(self.dimention):
                    if random.random() < self.C:
                        trial[j] = mutant[j]

                f_trial = self.function(trial)
                if f_trial < self.particles[i].value:
                    self.particles[i].solution = trial
                    self.particles[i].value = f_trial

            self.bestParticle = min(self.particles, key=lambda p: p.value)

            if self.breakCriteria is not None and self.bestParticle.value < self.breakCriteria:
                break

        return self.bestParticle.solution, self.bestParticle.value
