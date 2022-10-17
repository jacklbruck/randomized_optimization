import mlrose_hiive as mlrose
import numpy as np

def random_edges(n):
    # Initialize graph.
    nodes = list(range(n))
    edges = []

    # Add edges.
    for _ in range(int(1 / 4 * n ** 2)):
        found = False

        while not found:
            u, v = sorted(np.random.choice(nodes, size=(2,), replace=False))

            if (u, v) not in edges:
                edges.append((u, v))

                found = True

    return edges

optimi = {
    "Flip Flop": lambda n: n - 1,
    "Four Peaks": lambda n: n + int((1 - 0.15) * n),
    "Max-K Color": lambda n: int(1 / 4 * n ** 2),
}

args = {
    "Flip Flop": lambda n: {},
    "Four Peaks": lambda n: {"t_pct": 0.15},
    "Max-K Color": lambda n: {"edges": random_edges(n)},
}

functions = {
    "Flip Flop": mlrose.FlipFlop,
    "Four Peaks": mlrose.FourPeaks,
    "Max-K Color": mlrose.MaxKColor,
}