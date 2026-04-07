import time
from modern_biojazz.grounding import GroundingEngine
from modern_biojazz.site_graph import ReactionNetwork, Protein

def run_benchmark():
    engine = GroundingEngine()
    # Create a large network
    network = ReactionNetwork()
    for i in range(1000000):
        network.proteins[f"Protein_{i}"] = Protein(name=f"Protein_{i}")

    constraints = {f"Protein_{i}": [f"Real_{i}"] for i in range(100)}
    real_interactions = []

    start_time = time.time()
    for _ in range(100):
        # original
        abstract_nodes = [n for n in network.proteins.keys() if n in constraints]
    end_time = time.time()
    print(f"Baseline Time taken: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    for _ in range(100):
        # new
        abstract_nodes = [n for n in network.proteins if n in constraints]
    end_time = time.time()
    print(f"Improved Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
