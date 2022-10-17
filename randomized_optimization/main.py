import mlrose_hiive as mlrose
import numpy as np
import logging

from . import algorithms, functions, args, sizes, optimi, networks
from .config.fitness import functions
from sklearn.metrics import f1_score
from itertools import product
from pathlib import Path
from json import dump
from time import time


def fit(algorithm, size):
    # Read data from disk.
    X_train = np.load(Path(__file__).parents[1] / "data" / "X_train.npy")
    X_test = np.load(Path(__file__).parents[1] / "data" / "X_test.npy")
    y_train = np.load(Path(__file__).parents[1] / "data" / "y_train.npy")
    y_test = np.load(Path(__file__).parents[1] / "data" / "y_test.npy")

    # Initialize network.
    clf = mlrose.NeuralNetwork(
        hidden_nodes=[8, 8, 8],
        activation="relu",
        max_iters=size,
        clip_max=1,
        curve=True,
        **algorithm,
    )

    # Fit network.
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    return {
        "Training F1-Score": f1_score(y_train, clf.predict(X_train).flatten()),
        "Testing F1-Score": f1_score(y_test, clf.predict(X_test).flatten()),
        "Fitness Curve": clf.fitness_curve.tolist(),
        "Wall Clock Time": end - start,
    }


def main():
    for (
        (algorithm_id, algorithm),
        (function_id, function),
    ) in product(algorithms.items(), functions.items()):
        logging.info(f"Using {algorithm_id} on {function_id}...")

        for size in sizes[function_id]:
            logging.info(f"\tString length {size}...")

            # Define optimization problem.
            problem = mlrose.DiscreteOpt(size, function(**args[function_id](size)))

            # Optimize.
            start = time()
            (best_state, best_fitness, fitness_curve) = algorithm(problem)
            end = time()

            logging.info(f"\t\tFitness Percentage: {best_fitness / optimi[function_id](size):.2f}")

            # Write results.
            fp = Path(__file__).parents[1] / "results" / algorithm_id / function_id / f"{size}"
            fp.mkdir(exist_ok=True, parents=True)

            with open(fp / "out.json", "w+") as f:
                out = {
                    "Best State": best_state.tolist(),
                    "Best Fitness": best_fitness,
                    "Fitness Percentage": best_fitness / optimi[function_id](size),
                    "Fitness Curve": fitness_curve.tolist(),
                    "Wall Clock Time": end - start,
                }

                dump(out, f)

    for algorithm_id, algorithm in networks.items():
        logging.info(f"Using {algorithm_id} on Neural Network...")

        for size in sizes["Neural Network"]:
            logging.info(f"\tMax iterations {size}...")

            # Optimize.
            outs = {
                "Training F1-Score": [],
                "Testing F1-Score": [],
                "Fitness Curve": [],
                "Wall Clock Time": [],
            }

            for _ in range(10):
                out = fit(algorithm, size)

                for k, v in out.items():
                    outs[k].append(v)

            # Write results.
            fp = Path(__file__).parents[1] / "results" / algorithm_id / "Neural Network" / f"{size}"
            fp.mkdir(exist_ok=True, parents=True)

            with open(fp / "out.json", "w+") as f:
                dump(outs, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    main()
