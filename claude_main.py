"""
VQE Quantum Chemistry Engine
Ground-state energy via Variational Quantum Eigensolver
on a noise-injected IBM Digital Twin.

Architecture:
  Layer 1 & 2  - Classical featurization + Jordan-Wigner mapping
  Layer 3      - Hardware-Efficient Ansatz (HEA) on Aer digital twin
  Layer 4      - SPSA optimization loop (2 circuits/step, queue-safe)
  Layer 5      - PES scan + output
"""

import os
import warnings
import numpy as np
import pennylane as qml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_NAME     = "ibm_kingston"         # Target QPU for noise profile
SHOTS            = 10_000                 # Measurement shots per circuit
SPSA_MAXITER     = 300                    # Hard cap — patience will exit well before this
HEA_LAYERS       = 2                      # Depth of the HEA ansatz

# Layer 5 — PES scan
COMPARE_NOISE    = True                   # Plot noiseless vs noisy Digital Twin on Panel 2
PES_BOND_LENGTHS = np.linspace(0.7, 2.5, 10)  # O-H sweep range (Å)

# Molecule: Water (H2O) in Angstroms, 104.5° bent geometry
SYMBOLS  = ["H", "O", "H"]
GEOMETRY = np.array([
    [-0.0399, -0.0038, 0.0],
    [ 1.5780,  0.8540, 0.0],
    [ 2.7909, -0.5159, 0.0],
])

# Active space — freeze O 1s core, simulate valence electrons only
ACTIVE_ELECTRONS = 4
ACTIVE_ORBITALS  = 4

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 & 2: Featurization + Qubit Mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_hamiltonian(symbols, geometry, active_electrons, active_orbitals, basis="6-31g"):
    """
    Compute the molecular Hamiltonian in the Pauli-word basis.
    Returns (H, n_qubits).
    """
    molecule = qml.qchem.Molecule(symbols, geometry, basis_name=basis)
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    return H, n_qubits


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: Quantum Device — Digital Twin or Local Fallback
# ─────────────────────────────────────────────────────────────────────────────

def build_device(n_qubits, backend_name=BACKEND_NAME, shots=SHOTS):
    """
    Attempt to build a noise-injected AerSimulator digital twin of the real QPU.
    Falls back to pennylane's default.qubit if IBM credentials are unavailable.
    """
    ibm_token = os.getenv("IBM_API_KEY")

    if ibm_token:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_aer import AerSimulator

            service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=ibm_token,
            )
            real_backend = service.backend(backend_name)
            print(f"[Device]  Fetching noise profile from {real_backend.name}...")
            noisy_sim = AerSimulator.from_backend(real_backend)
            dev = qml.device("qiskit.aer", wires=n_qubits, backend=noisy_sim, shots=shots)
            print(f"[Device]  Digital twin ready — emulating {backend_name} locally.\n")
            return dev

        except Exception as exc:
            print(f"[Device]  IBM connection failed ({exc}). Falling back to default.qubit.\n")

    print("[Device]  No IBM credentials found. Using default.qubit (noiseless).\n")
    return qml.device("default.qubit", wires=n_qubits)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 (cont.): Hardware-Efficient Ansatz
# ─────────────────────────────────────────────────────────────────────────────

def hea_param_shape(n_qubits, n_layers):
    """
    SimplifiedTwoDesign parameter shapes:
      initial_layer_weights : (n_qubits,)          — one RY per qubit
      weights               : (n_layers, n_qubits-1, 2) — entangling layers
    Returns the two shapes as a tuple.
    """
    init_shape    = (n_qubits,)
    weights_shape = (n_layers, n_qubits - 1, 2)
    return init_shape, weights_shape


def make_circuit(H, n_qubits, n_layers, dev):
    """
    Build and return the VQE QNode.

    params is a flat 1-D array. Internally we split it into the two
    tensor shapes required by SimplifiedTwoDesign so SPSA can treat
    it as a single vector (2 circuit evaluations per step).
    """
    init_shape, weights_shape = hea_param_shape(n_qubits, n_layers)
    n_init    = int(np.prod(init_shape))
    n_weights = int(np.prod(weights_shape))
    n_total   = n_init + n_weights

    # Hartree-Fock reference state: fill the lowest `active_electrons` spin-orbitals
    hf_state = qml.qchem.hf_state(ACTIVE_ELECTRONS, n_qubits)

    @qml.qnode(dev)
    def circuit(params_flat):
        # ── State preparation (Hartree-Fock initialization) ──────────────────
        qml.BasisState(hf_state, wires=range(n_qubits))

        # ── Unpack flat params into HEA tensor shapes ─────────────────────────
        init_w    = params_flat[:n_init].reshape(init_shape)
        weights_w = params_flat[n_init:].reshape(weights_shape)

        # ── Hardware-Efficient Ansatz ──────────────────────────────────────────
        # SimplifiedTwoDesign: alternating RY layers + CZ entanglers.
        # Topology: nearest-neighbour only → survives Heavy-Hex routing.
        qml.SimplifiedTwoDesign(
            initial_layer_weights=init_w,
            weights=weights_w,
            wires=range(n_qubits),
        )

        return qml.expval(H)

    return circuit, n_total


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4: SPSA Optimization Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_vqe(circuit, n_params, spsa_maxiter=SPSA_MAXITER, verbose=True):
    """
    Minimize <H> using SPSA with Noisy Early Stopping (Patience).

    Tracks the absolute best energy seen. If no improvement of at least
    CONV_TOL is made for PATIENCE_LIMIT consecutive steps, the optimizer
    has hit the shot-noise floor and halts early. Returns the params from
    the best step, not the last.

    Returns (best_params, energy_history).
    """
    PATIENCE_LIMIT   = 20
    CONV_TOL         = 1e-4

    params           = np.zeros(n_params)
    best_energy      = float("inf")
    best_params      = params.copy()
    patience_counter = 0

    opt = qml.SPSAOptimizer(
        maxiter=spsa_maxiter,
        a=0.05,
        c=0.15,
        alpha=0.602,
        gamma=0.101,
    )
    energy_history = []

    if verbose:
        print("─" * 60)
        print(f"{'Step':>5}  {'Energy (Ha)':>16}  {'Best (Ha)':>16}  {'Patience':>8}")
        print("─" * 60)

    prev_energy = float(circuit(params))
    energy_history.append(prev_energy)
    best_energy = prev_energy
    best_params = params.copy()

    for step in range(spsa_maxiter):
        params, energy = opt.step_and_cost(circuit, params)
        energy         = float(energy)
        energy_history.append(energy)

        if energy < best_energy - CONV_TOL:
            best_energy      = energy
            best_params      = params.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (step % 10 == 0 or step == spsa_maxiter - 1):
            print(f"{step:>5}  {energy:>16.8f}  {best_energy:>16.8f}  {patience_counter:>8}/{PATIENCE_LIMIT}")

        if patience_counter >= PATIENCE_LIMIT:
            if verbose:
                print(f"\n  ↳ Convergence reached via early stopping at step {step}.")
                print(f"    No improvement > {CONV_TOL:.0e} Ha for {PATIENCE_LIMIT} consecutive steps.")
                print(f"    Best energy: {best_energy:.8f} Ha")
            break

    if verbose:
        print("─" * 60)

    return best_params, energy_history


def _geometry_from_bond_length(r):
    """Symmetric H2O geometry from O-H bond length r (Angstroms)."""
    import math
    angle_rad = math.radians(104.5 / 2)
    return np.array([
        [-r * math.sin(angle_rad),  r * math.cos(angle_rad), 0.0],   # H1
        [ 0.0,                       0.0,                     0.0],   # O
        [ r * math.sin(angle_rad),   r * math.cos(angle_rad), 0.0],   # H2
    ])


def pes_scan(bond_lengths, symbols, active_electrons, active_orbitals,
             n_layers=HEA_LAYERS, shots=SHOTS, use_noise=False, verbose=True):
    """
    Sweep over O-H bond lengths and record the VQE ground-state energy at each.
    H-O-H angle is held at 104.5° throughout.

    Args:
        use_noise: if True, build an ibm_kingston Digital Twin device;
                   if False, use noiseless default.qubit.

    Returns (bond_lengths_list, energies_list).
    """
    tag      = "noisy [ibm_kingston]" if use_noise else "noiseless [default.qubit]"
    energies = []

    for r in bond_lengths:
        if verbose:
            print(f"  [PES/{tag}]  r(O-H) = {r:.2f} Å", end="  →  ", flush=True)

        geometry          = _geometry_from_bond_length(r)
        H, n_qubits       = build_hamiltonian(symbols, geometry, active_electrons, active_orbitals)
        dev               = build_device(n_qubits, shots=shots) if use_noise \
                            else qml.device("default.qubit", wires=n_qubits)
        circuit, n_params = make_circuit(H, n_qubits, n_layers, dev)
        _, energy_history = run_vqe(circuit, n_params, verbose=False)
        final_e           = energy_history[-1]
        energies.append(final_e)

        if verbose:
            print(f"E = {final_e:.6f} Ha")

    return list(bond_lengths), energies


def export_dashboard(history, pes_data, output_path="vqe_dashboard.png"):
    """
    Save a dual-panel publication-quality PNG.

    Panel 1 — SPSA Convergence: energy vs optimisation step.
    Panel 2 — Potential Energy Surface: energy vs O-H bond length.
              If COMPARE_NOISE is True, both noiseless and noisy curves are shown.

    Args:
        history   : list of floats from run_vqe
        pes_data  : dict with keys "noiseless" and/or "noisy",
                    each mapping to (bond_lengths, energies)
        output_path: file path for the saved PNG
    """
    import matplotlib
    matplotlib.use("Agg")   # headless — no display required
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ── Palette ───────────────────────────────────────────────────────────────
    BG        = "#0d1117"   # GitHub-dark canvas
    PANEL     = "#161b22"   # slightly lighter panel fill
    ACCENT1   = "#58a6ff"   # IBM-blue  — noiseless / convergence line
    ACCENT2   = "#f78166"   # warm red  — noisy hardware line
    GRID      = "#21262d"
    TEXT      = "#e6edf3"
    SUBTEXT   = "#8b949e"
    CBS_LINE  = "#3fb950"   # green reference

    CBS_ENERGY = -76.0600   # Complete Basis Set limit for H2O (Ha)

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "grid.color":        GRID,
        "text.color":        TEXT,
        "font.family":       "monospace",
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID,
    })

    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    fig.suptitle(
        "VQE Quantum Chemistry Engine  —  H₂O Ground State",
        fontsize=15, fontweight="bold", color=TEXT, y=0.97,
    )

    gs     = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    ax_spsa = fig.add_subplot(gs[0])
    ax_pes  = fig.add_subplot(gs[1])

    # ── Panel 1: SPSA Convergence ─────────────────────────────────────────────
    steps = range(len(history))

    # Rolling 10-step smoothed line behind the raw trace
    window      = min(10, len(history) // 5 or 1)
    smoothed    = np.convolve(history, np.ones(window) / window, mode="valid")
    smooth_x    = range(window - 1, len(history))

    ax_spsa.plot(steps, history, color=ACCENT1, alpha=0.25, linewidth=0.8, label="_raw")
    ax_spsa.plot(smooth_x, smoothed, color=ACCENT1, linewidth=2.0, label="Energy (smoothed)")
    ax_spsa.axhline(history[0],  color=SUBTEXT,  linewidth=1.0, linestyle="--", label=f"HF baseline  {history[0]:.4f} Ha")
    ax_spsa.axhline(history[-1], color=CBS_LINE,  linewidth=1.0, linestyle=":",  label=f"Final  {history[-1]:.4f} Ha")

    ax_spsa.set_title("SPSA Convergence", fontsize=12, pad=10)
    ax_spsa.set_xlabel("Optimisation Step", fontsize=10)
    ax_spsa.set_ylabel("Energy (Hartree)", fontsize=10)
    ax_spsa.grid(True, linewidth=0.5)
    ax_spsa.legend(fontsize=8, loc="upper right")

    # Annotate the improvement arrow
    improvement = history[0] - history[-1]
    mid_step    = len(history) // 2
    ax_spsa.annotate(
        f"ΔE = {improvement:.4f} Ha",
        xy=(mid_step, history[-1]),
        xytext=(mid_step, (history[0] + history[-1]) / 2),
        arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.5),
        color=ACCENT2, fontsize=9, ha="center",
    )

    # ── Panel 2: Potential Energy Surface ────────────────────────────────────
    if "noiseless" in pes_data:
        r_nl, e_nl = pes_data["noiseless"]
        ax_pes.plot(r_nl, e_nl, "o-", color=ACCENT1, linewidth=2.0,
                    markersize=5, label="Noiseless (default.qubit)")
        eq_idx = int(np.argmin(e_nl))
        ax_pes.annotate(
            f"r* = {r_nl[eq_idx]:.2f} Å",
            xy=(r_nl[eq_idx], e_nl[eq_idx]),
            xytext=(r_nl[eq_idx] + 0.2, e_nl[eq_idx] + 0.3),
            arrowprops=dict(arrowstyle="->", color=ACCENT1, lw=1.2),
            color=ACCENT1, fontsize=9,
        )

    if "noisy" in pes_data:
        r_n, e_n = pes_data["noisy"]
        ax_pes.plot(r_n, e_n, "s--", color=ACCENT2, linewidth=1.8,
                    markersize=5, label=f"Noisy Digital Twin ({BACKEND_NAME})")

    ax_pes.axhline(CBS_ENERGY, color=CBS_LINE, linewidth=1.0, linestyle=":",
                   label=f"CBS limit  {CBS_ENERGY:.2f} Ha")

    ax_pes.set_title("Potential Energy Surface  —  O-H Stretch", fontsize=12, pad=10)
    ax_pes.set_xlabel("O-H Bond Length (Å)", fontsize=10)
    ax_pes.set_ylabel("Energy (Hartree)", fontsize=10)
    ax_pes.grid(True, linewidth=0.5)
    ax_pes.legend(fontsize=8, loc="upper right")

    # Hardware badge
    ax_pes.text(
        0.03, 0.04,
        f"QPU: {BACKEND_NAME} Digital Twin\nShots: {SHOTS:,}  |  SPSA iters: {SPSA_MAXITER}",
        transform=ax_pes.transAxes, fontsize=7.5, color=SUBTEXT,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=GRID, alpha=0.8),
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Dashboard]  Saved → {output_path}")


def export_json(molecule, backend, eq_bond_length, final_energy,
                steps_to_converge, output_path="vqe_results.json"):
    """
    Write a structured JSON payload for downstream UI consumption.

    Keys:
        molecule               : e.g. "H2O"
        backend                : QPU identifier string
        equilibrium_bond_length_A : float, Å
        final_energy_Ha        : float, Hartree
        steps_to_converge      : int
    """
    import json

    payload = {
        "molecule":                   molecule,
        "backend":                    backend,
        "equilibrium_bond_length_A":  round(float(eq_bond_length), 4),
        "final_energy_Ha":            round(float(final_energy), 8),
        "steps_to_converge":          int(steps_to_converge),
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[JSON]       Saved → {output_path}")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   VQE Quantum Chemistry Engine  —  H₂O           ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # ── Layers 1 & 2: Hamiltonian ─────────────────────────────────────────────
    print("[Hamiltonian]  Building H₂O Hamiltonian (active space 4e / 4o)...")
    H, n_qubits = build_hamiltonian(SYMBOLS, GEOMETRY, ACTIVE_ELECTRONS, ACTIVE_ORBITALS)
    n_terms = len(H.operands) if hasattr(H, "operands") else len(H.ops)
    print(f"[Hamiltonian]  Pauli terms: {n_terms}  |  Qubits required: {n_qubits}\n")

    # ── Layer 3: Device + Circuit ─────────────────────────────────────────────
    dev = build_device(n_qubits)
    circuit, n_params = make_circuit(H, n_qubits, HEA_LAYERS, dev)
    print(f"[Circuit]  HEA depth: {HEA_LAYERS} layers  |  Trainable params: {n_params}\n")

    # ── Layer 4: VQE Optimisation ─────────────────────────────────────────────
    print("[Optimizer]  SPSA — 2 circuit evaluations per step (queue-safe)\n")
    opt_params, history = run_vqe(circuit, n_params, spsa_maxiter=SPSA_MAXITER)

    final_energy       = history[-1]
    steps_to_converge  = len(history) - 1
    print(f"\n[Result]  Ground-state energy : {final_energy:.8f} Ha")
    print(f"[Result]  Energy improvement  : {history[0] - final_energy:.8f} Ha over {steps_to_converge} steps")
    print(f"[Result]  CBS reference (H₂O) : -76.0600 Ha\n")

    # ── Layer 5: PES Scan ─────────────────────────────────────────────────────
    pes_data = {}

    print("[PES]  Running noiseless sweep (default.qubit)...")
    r_vals, e_noiseless = pes_scan(
        PES_BOND_LENGTHS, SYMBOLS, ACTIVE_ELECTRONS, ACTIVE_ORBITALS,
        use_noise=False, verbose=True,
    )
    pes_data["noiseless"] = (r_vals, e_noiseless)

    if COMPARE_NOISE:
        print("\n[PES]  Running noisy sweep (ibm_kingston Digital Twin)...")
        _, e_noisy = pes_scan(
            PES_BOND_LENGTHS, SYMBOLS, ACTIVE_ELECTRONS, ACTIVE_ORBITALS,
            use_noise=True, verbose=True,
        )
        pes_data["noisy"] = (r_vals, e_noisy)

    # Equilibrium bond length = minimum of the noiseless PES
    eq_idx            = int(np.argmin(e_noiseless))
    eq_bond_length    = r_vals[eq_idx]
    print(f"\n[PES]  Equilibrium O-H bond length: {eq_bond_length:.2f} Å  "
          f"(E = {e_noiseless[eq_idx]:.6f} Ha)")

    # ── Layer 5: Dashboard PNG ────────────────────────────────────────────────
    print("\n[Dashboard]  Rendering dual-panel figure...")
    export_dashboard(history, pes_data, output_path="vqe_dashboard.png")

    # ── Layer 5: JSON Payload ─────────────────────────────────────────────────
    payload = export_json(
        molecule             = "H2O",
        backend              = BACKEND_NAME,
        eq_bond_length       = eq_bond_length,
        final_energy         = final_energy,
        steps_to_converge    = steps_to_converge,
        output_path          = "vqe_results.json",
    )

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   Run Complete                                    ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Final energy      : {final_energy:>12.6f} Ha          ║")
    print(f"║  Eq. bond length   : {eq_bond_length:>12.4f} Å           ║")
    print(f"║  Steps             : {steps_to_converge:>12d}              ║")
    print("║  Outputs           : vqe_dashboard.png           ║")
    print("║                    : vqe_results.json            ║")
    print("╚══════════════════════════════════════════════════╝\n")
