# Quantum Machine Learning Tensor Network States
This is the follow-up code for the article "Quantum Machine Learning Tensor Network States",	arXiv:1804.02398 

The program generates a given number of random unitary matrices of dimensionality *2^n*, where *n* is a given number of qubits, and, for each unitary, tries to approximate one of its eigenstates by using the Matrix Product State ansatz with successive increasing the number of ebits it can support (see the original paper for details).


The program is represented in the two versions: by the source code *learning-mps.py* and by the Jupyter notebook *learning-mps.ipynb*.

The program outputs the four files:
- fun_values_flat.txt -- the values of the cost function for each iteration;
- overlaps_flat.txt -- overlaps with the closest eigenstate for each iteration;
- entanglements_flat.txt -- entanglement entropies of the found eigenstate for each iteration;
- plot-*.pdf -- the plot of above-mentioned values.

The Qiskit package is required for execution.
