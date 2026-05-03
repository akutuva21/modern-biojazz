import time
import timeit

setup = """
from modern_biojazz.site_graph import ReactionNetwork, Protein
proteins = {f"P{i}": Protein(name=f"P{i}", sites={}) for i in range(100000)}
network = ReactionNetwork(proteins=proteins, rules=[])
constraints = {"P10": ["A"], "P5000": ["B"]}
"""

code_keys = """
abstract_nodes = [n for n in network.proteins.keys() if n in constraints]
"""

code_no_keys = """
abstract_nodes = [n for n in network.proteins if n in constraints]
"""

t_keys = timeit.timeit(code_keys, setup=setup, number=1000)
t_no_keys = timeit.timeit(code_no_keys, setup=setup, number=1000)

print(f"Time with .keys(): {t_keys:.4f} seconds")
print(f"Time without .keys(): {t_no_keys:.4f} seconds")
