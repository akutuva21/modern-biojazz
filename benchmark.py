import time
from modern_biojazz.evolution import LLMEvolutionEngine, EvolutionConfig, RandomProposer
from modern_biojazz.site_graph import ReactionNetwork, Protein, Rule
from modern_biojazz.simulation import SimulationBackend, FitnessScorer
from typing import List, Dict, Any

class MockBackend(SimulationBackend):
    def simulate(self, network: ReactionNetwork, t_end: float, dt: float, solver: str) -> Dict[str, Any]:
        time.sleep(0.01) # Simulate expensive simulation
        return {"trajectory": [], "stats": {}, "solver": solver}

class MockFitness(FitnessScorer):
    def score(self, backend: SimulationBackend, network: ReactionNetwork, t_end: float, dt: float, solver: str) -> float:
        # call simulate to trigger sleep
        backend.simulate(network, t_end, dt, solver)
        return float(len(network.rules))

network = ReactionNetwork()
network.proteins["A"] = Protein(name="A", sites=[])
network.proteins["B"] = Protein(name="B", sites=[])
network.rules.append(Rule(name="rule1", rule_type="bind", reactants=["A"], products=["B"], rate=1.0))

engine = LLMEvolutionEngine(
    simulation_backend=MockBackend(),
    fitness_evaluator=MockFitness(),
    proposer=RandomProposer()
)

config = EvolutionConfig(
    population_size=10,
    generations=2,
    islands=2,
    sim_t_end=1.0,
    sim_dt=0.1
)

start = time.time()
engine.run(network, config)
end = time.time()

print(f"Time taken: {end - start:.4f} seconds")
