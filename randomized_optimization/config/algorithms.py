import mlrose_hiive as mlrose

from functools import partial

algorithms = {
    "Randomized Hill Climbing": partial(mlrose.random_hill_climb, restarts=3, curve=True),
    "Simulated Annealing": partial(mlrose.simulated_annealing, curve=True),
    "Genetic Algorithm": partial(mlrose.genetic_alg, pop_size=50, curve=True),
    "MIMIC": partial(mlrose.mimic, curve=True),
}