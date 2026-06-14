"""
VQE Telemetry Backend — FastAPI + Server-Sent Events
=====================================================
Wraps Layers 1-4 of vqe_engine.py in a streaming HTTP API.

Endpoints:
  POST /run-vqe   { bond_length: float }
    → SSE stream of { step, energy, best_energy, params, patience, status }

  GET  /health
    → { status: "ok" }

Run with:
  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import math
import os
import warnings

import numpy as np
import pennylane as qml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# App & CORS
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="VQE Telemetry API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to ["http://localhost:5173"] in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Engine config (mirrors vqe_engine.py constants)
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_NAME     = "ibm_kingston"
SHOTS            = 10_000
HEA_LAYERS       = 2
SPSA_MAXITER     = 300
PATIENCE_LIMIT   = 20
CONV_TOL         = 1e-4
ACTIVE_ELECTRONS = 4
ACTIVE_ORBITALS  = 4
SYMBOLS          = ["H", "O", "H"]


# ─────────────────────────────────────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────────────────────────────────────

class VQERequest(BaseModel):
    bond_length: float = Field(default=0.96, ge=0.5, le=3.0,
                               description="O-H bond length in Angstroms")


# ─────────────────────────────────────────────────────────────────────────────
# Engine helpers (self-contained — no import from vqe_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

def _geometry(r_angstrom: float) -> np.ndarray:
    """Symmetric H2O geometry from bond length r in Angstroms, converted to Bohr."""
    r_bohr = r_angstrom * 1.88973 
    
    angle = math.radians(104.5 / 2)
    return np.array([
        [-r_bohr * math.sin(angle),  r_bohr * math.cos(angle), 0.0],
        [ 0.0,                        0.0,                      0.0],
        [ r_bohr * math.sin(angle),   r_bohr * math.cos(angle), 0.0],
    ])


def _build_hamiltonian(geometry: np.ndarray):
    molecule = qml.qchem.Molecule(SYMBOLS, geometry, basis_name="6-31g")
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule,
        active_electrons=ACTIVE_ELECTRONS,
        active_orbitals=ACTIVE_ORBITALS,
    )
    return H, n_qubits


def _build_device(n_qubits: int):
    token = os.getenv("IBM_API_KEY")
    if token:
        try:
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime import QiskitRuntimeService
            svc     = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            backend = svc.backend(BACKEND_NAME)
            noisy   = AerSimulator.from_backend(backend)
            return qml.device("qiskit.aer", wires=n_qubits, backend=noisy, shots=SHOTS)
        except Exception:
            pass
    return qml.device("default.qubit", wires=n_qubits)


def _build_circuit(H, n_qubits: int, dev):
    hf_state      = qml.qchem.hf_state(ACTIVE_ELECTRONS, n_qubits)
    init_shape    = (n_qubits,)
    weights_shape = (HEA_LAYERS, n_qubits - 1, 2)
    n_init        = int(np.prod(init_shape))
    n_total       = n_init + int(np.prod(weights_shape))

    @qml.qnode(dev)
    def circuit(params_flat):
        qml.BasisState(hf_state, wires=range(n_qubits))
        init_w    = params_flat[:n_init].reshape(init_shape)
        weights_w = params_flat[n_init:].reshape(weights_shape)
        qml.SimplifiedTwoDesign(
            initial_layer_weights=init_w,
            weights=weights_w,
            wires=range(n_qubits),
        )
        return qml.expval(H)

    return circuit, n_total


# ─────────────────────────────────────────────────────────────────────────────
# Generator — yields SSE packets from the SPSA loop
# ─────────────────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def vqe_stream(bond_length: float):
    """
    Generator that runs the full VQE pipeline and yields one SSE packet
    per optimisation step.  Designed to be consumed by StreamingResponse.

    Packet schema:
      {
        step         : int,
        energy       : float,        # current step energy (Ha)
        best_energy  : float,        # best seen so far (Ha)
        params       : [float x 36], # current flat parameter vector
        patience     : int,          # patience counter
        status       : "running" | "converged" | "error",
        message      : str           # human-readable note (optional)
      }
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    yield _sse({"step": -1, "status": "running",
                "message": f"Building H₂O Hamiltonian at r={bond_length:.3f} Å…"})

    geometry          = _geometry(bond_length)
    H, n_qubits       = _build_hamiltonian(geometry)
    dev               = _build_device(n_qubits)
    circuit, n_params = _build_circuit(H, n_qubits, dev)

    yield _sse({"step": -1, "status": "running",
                "message": f"Circuit ready — {n_params} params, {n_qubits} qubits. Entering SPSA loop…"})

    # ── SPSA with patience early-stopping ─────────────────────────────────────
    params           = np.zeros(n_params)
    best_energy      = float("inf")
    best_params      = params.copy()
    patience_counter = 0

    opt = qml.SPSAOptimizer(
        maxiter=SPSA_MAXITER,
        a=0.05, c=0.15, alpha=0.602, gamma=0.101,
    )

    # Step 0: evaluate HF baseline before any update
    hf_energy   = float(circuit(params))
    best_energy = hf_energy
    best_params = params.copy()

    yield _sse({
        "step":        0,
        "energy":      hf_energy,
        "best_energy": best_energy,
        "params":      params.tolist(),
        "patience":    0,
        "status":      "running",
        "message":     f"HF baseline: {hf_energy:.6f} Ha",
    })

    for step in range(1, SPSA_MAXITER + 1):
        params, energy = opt.step_and_cost(circuit, params)
        energy         = float(energy)

        if energy < best_energy - CONV_TOL:
            best_energy      = energy
            best_params      = params.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        converged = patience_counter >= PATIENCE_LIMIT
        status    = "converged" if converged else "running"

        yield _sse({
            "step":        step,
            "energy":      energy,
            "best_energy": best_energy,
            "params":      params.tolist(),
            "patience":    patience_counter,
            "status":      status,
            "message":     (
                f"Converged at step {step} — best {best_energy:.6f} Ha"
                if converged else ""
            ),
        })

        if converged:
            break

    # Final summary packet if we hit the hard cap without triggering patience
    if patience_counter < PATIENCE_LIMIT:
        yield _sse({
            "step":        SPSA_MAXITER,
            "energy":      best_energy,
            "best_energy": best_energy,
            "params":      best_params.tolist(),
            "patience":    patience_counter,
            "status":      "converged",
            "message":     f"Max iterations reached — best {best_energy:.6f} Ha",
        })


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-vqe")
def run_vqe_endpoint(req: VQERequest):
    """
    Stream SPSA telemetry as Server-Sent Events.

    The generator runs synchronously inside FastAPI's thread pool
    (suitable for a single-user hackathon demo; swap to BackgroundTask +
    asyncio.Queue for multi-user production).
    """
    def event_stream():
        try:
            yield from vqe_stream(req.bond_length)
        except Exception as exc:
            yield _sse({"status": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",    # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )
