import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

symbols = ["H", "H"]
coordinates = np.array([[-0.673, 0, 0], [0.673, 0, 0]])

H, qubits = qchem.molecular_hamiltonian(symbols, coordinates)
print(qubits)
print(H)

num_wires = qubits
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def exp_energy(state):
    qml.BasisState(np.array(state), wires=range(num_wires))
    return qml.expval(H)

print(exp_energy([1, 0, 1, 0]))

hf = qchem.hf_state(electrons=2, orbitals=4)
print(exp_energy(hf))
