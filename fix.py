import re

with open("tests/test_mutation_unit.py", "r") as f:
    content = f.read()

fixed = re.sub(r'<<<<<<< Updated upstream.*?\n(.*?)\n=======\n(.*?)>>>>>>> Stashed changes', r'\1\n\2', content, flags=re.DOTALL)

with open("tests/test_mutation_unit.py", "w") as f:
    f.write(fixed)
