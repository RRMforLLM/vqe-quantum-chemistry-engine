import pennylane as qml
from pennylane import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from dotenv import load_dotenv
import os

load_dotenv()

# --- LAYERS 1 & 2: Featurization and Active Space Mapping ---
symbols = ["H", "O", "H"]
geometry = np.array([[-0.0399, -0.0038, 0], [1.5780, 0.8540, 0], [2.7909, -0.5159, 0]])

# 1. Define the molecule
molecule = qml.qchem.Molecule(symbols, geometry, basis_name="6-31g")

# 2. Define the Active Space
# We tell the software to freeze the lowest energy orbital (the Oxygen 1s core)
# Total spatial orbitals = 7. Freezing 1 leaves an active space of 6 spatial orbitals.
# Total electrons = 10. Freezing 2 leaves an active space of 8 valence electrons.
active_electrons = 4
active_orbitals = 4

# 3. Generate the Hamiltonian using only the active space
H, qubits = qml.qchem.molecular_hamiltonian(
    molecule, 
    active_electrons=active_electrons, 
    active_orbitals=active_orbitals
)

print(f"Reduced Qubits: {qubits}")

# --- LAYER 3: The Quantum Circuit ---
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.getenv("IBM_API_KEY"))
real_backend = service.backend("ibm_kingston")

print(f"Fetching real-time noise profile from {real_backend.name}...")

noisy_sim = AerSimulator.from_backend(real_backend)

dev = qml.device("qiskit.aer", wires=qubits, backend=noisy_sim, shots=10000)

electrons = active_electrons
singles, doubles = qml.qchem.excitations(electrons, qubits)

@qml.qnode(dev)
def quantum_circuit(params):
    for i in range(qubits):
        qml.RY(params[0][i], wires=i)
    
    qml.SimplifiedTwoDesign(initial_layer_weights=params[0], weights=params[1], wires=range(qubits))

    return qml.expval(H)

# --- LAYER 4: The Classical Optimizer Loop (CORRECTED) ---
num_params = len(singles) + len(doubles)
params = np.zeros(num_params, requires_grad=True)

initial_energy = quantum_circuit(params)
print(f"Initial Hartree-Fock energy: {initial_energy:.8f} Ha")

opt = qml.SPSAOptimizer(maxiter=100)

max_iterations = 1

prev_energy = initial_energy

print("\n--- Starting VQE Optimization ---")
print(f"Targeting QPU: {noisy_sim.backend_name}")
print("Compiling circuits and entering IBM queue...")

for n in range(max_iterations):
    params, current_energy = opt.step_and_cost(quantum_circuit, params)

    energy_change = np.abs(prev_energy - current_energy)

    print(f"Step {n:3d} | Real Hardware Energy: {current_energy:.8f} Ha | Improvement: {energy_change:.8f}")

    prev_energy = current_energy

print(f"\nFinal Ground State Energy: {current_energy:.8f} Ha")
print(f"Optimized Parameters:      {params}")

print("Hardware execution complete.")
