# CLAUDE.md — QuantumComputing Repository Guide

## Overview

This is a hands-on quantum computing exploration repository implemented with **Qiskit** and **qiskit-aer**. Each subdirectory contains a self-contained Python "Explorer" script that demonstrates a specific quantum algorithm or concept through structured experiments and matplotlib visualizations. The focus is educational: scripts print explanations, run circuits on a local simulator, and save PNG charts.

---

## Repository Structure

```
QuantumComputing/
├── MyFirstQuantumAlgoritherm.py          # Root-level entry point (Bell state demo)
├── README.md                              # Minimal requirements note
├── MyFirstQuantumCircuit/
│   └── MyFirstQuantumAlgoritherm.py      # Older version with IBM token hardcoded
├── Deutsch Jozsa Explorer/
│   └── DeutschJozsaExplorer.py           # Deutsch-Jozsa algorithm (1–3 qubit)
├── Grovers Algorithm Explorer/
│   └── GroversAlgorithmExplorer.py       # Grover's search (2–3 qubit)
├── Quantum Circuit Optimization Explorer/
│   └── QuantumCircuitOptimizationExplorer.py
├── Quantum Classification Explorer/
│   └── QuantumClassificationExplorer.py  # QNN / VQC / quantum kernels
├── Quantum Error Mitigation Explorer/
│   └── QuantumErrorMitigationExplorer.py # ZNE, MEM, PEC, symmetry, DD
├── Quantum Interference Explorer/
│   └── QuantumInterferenceExplorer.py    # Mach-Zehnder, phase sweep, 2Q interference
├── Quantum Phase Estimation Explorer/
│   └── QuantumPhaseEstimationExplorer.py # QPE + quantum counting
├── Quantum Superposition Explorer/
│   └── QuantumSuperpositionExplorer.py   # Hadamard, RY angles, multi-qubit
└── Variational Quantum Eigensolver/
    └── VQEExplorer.py                    # VQE for ground state energy (H₂, simple Hamiltonians)
```

Each explorer directory also contains:
- A `.docx` document with algorithm notes
- A `.txt` file summarizing experiments and key insights
- Several `.png` output files (pre-generated visualization results)

---

## Dependencies

```
qiskit
qiskit-aer
qiskit-ibm-runtime
matplotlib
numpy
scipy          # used by VQE and QEM explorers
```

Install with:
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy scipy
```

The `README.md` at repo root also records: `Packages: qiskit, matplot`.

---

## Running the Scripts

All scripts are standalone and run directly with Python:

```bash
python "Deutsch Jozsa Explorer/DeutschJozsaExplorer.py"
python "Grovers Algorithm Explorer/GroversAlgorithmExplorer.py"
# etc.
```

Scripts run against the local **AerSimulator** by default. PNG output files are written to the **current working directory**, so run each script from within its own folder if you want outputs to land alongside the existing PNGs:

```bash
cd "Deutsch Jozsa Explorer" && python DeutschJozsaExplorer.py
```

IBM Quantum hardware integration is commented out in all scripts. The root-level `MyFirstQuantumAlgoritherm.py` uses `QiskitRuntimeService()` which requires saved credentials; the pattern for saving them is shown (commented out) as:
```python
QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='...', overwrite=True, set_as_default=True)
```
**Never commit real IBM tokens.** The `MyFirstQuantumCircuit/` copy has a token placeholder (`*...`) but it should be treated as a legacy file.

---

## Code Patterns & Conventions

### Standard imports (used by every explorer)
```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import Statevector

import numpy as np
import matplotlib.pyplot as plt
```

### Canonical execution helper (copy-paste across all scripts)
```python
def run_circuit_and_get_counts(circuit, backend, shots=1000):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)
    result = job.result()
    return result[0].data.c.get_counts()
```
This pattern transpiles the circuit to an ISA (Instruction Set Architecture) form before running it, which is required by `SamplerV2`.

### Statevector inspection (without measurement)
```python
def get_statevector(circuit):
    return Statevector.from_instruction(circuit)
```
Call on a circuit with no `measure()` instructions to inspect amplitudes directly.

### Backend initialization
```python
backend = AerSimulator()
```
All active scripts use the local simulator. IBM hardware selection (`service.least_busy(...)`, `service.get_backend("ibm_fez")`) is present but commented out.

### Visualization pattern
- Each explorer produces 3–6 matplotlib figures
- Saved with `plt.savefig('<name>.png', dpi=150, bbox_inches='tight')`
- `plt.show()` is typically NOT called (scripts are batch-style, not interactive)
- Figures use `fig.suptitle(...)` for overarching titles and per-axis `ax.set_title(...)`

### Experiment structure
Scripts organize code into numbered, clearly delimited experiments:
```python
# ============================================================================
# EXPERIMENT N: Descriptive Title
# ============================================================================
```
Each experiment prints results to stdout with `print(f"...")` statements before generating visuals.

---

## Module-by-Module Summary

### MyFirstQuantumAlgoritherm.py (root)
Bell state demo (2-qubit entanglement). The production version — uses `AerSimulator`, IBM hardware lines are safely commented. Good reference for the minimal boilerplate pattern.

### Deutsch Jozsa Explorer
Determines if a function f:{0,1}^n → {0,1} is constant or balanced in one oracle call.
- Experiments: function properties, 1-qubit Deutsch, 2-qubit DJ, 3-qubit DJ, classical vs quantum complexity table, oracle construction explanation
- Output PNGs: `deutsch_1qubit_results.png`, `deutsch_2qubit_results.png`, `deutsch_3qubit_results.png`, `deutsch_complexity.png`, `deutsch_algorithm_flow.png`

### Grovers Algorithm Explorer
Quantum search with √N speedup over classical linear search.
- Experiments: classical vs quantum framing, 2-qubit search for |11⟩, iteration count effects, 3-qubit search for |101⟩, optimal iteration sweep, multiple-target search
- Key components: oracle (phase flip on target), diffusion operator (inversion about average)
- Output PNGs: 5 files covering amplitude growth, histograms, iteration sweeps, conceptual overview

### Quantum Interference Explorer
Demonstrates constructive vs destructive interference.
- Experiments: classical vs quantum probability paths, Mach-Zehnder interferometer (no-phase/π/2/π shifts), phase sweep (0→2π), 2-qubit interference with/without middle Hadamard, designed destructive interference
- Output PNGs: `interference_mzi.png`, `interference_phase_sweep.png`, `interference_two_qubit.png`, `interference_theory.png`

### Quantum Superposition Explorer
Single and multi-qubit superposition mechanics.
- Experiments: Hadamard gate (50/50), RY at variable angles (0°–90°), RX/RY/RZ comparison, basis rotation measurements, 3-qubit equal superposition
- Uses `plot_bloch_multivector` for Bloch sphere visualization
- Output PNGs: 3 files covering RY angles, bases, multi-qubit states

### Variational Quantum Eigensolver
Hybrid quantum-classical optimization for ground state energy.
- Algorithm: parameterized ansatz → Pauli Hamiltonian measurement → classical optimizer loop
- Key imports: `SparsePauliOp`, `scipy.optimize.minimize`, `scipy.optimize.differential_evolution`
- Covers H₂-like Hamiltonians, COBYLA/SPSA optimizers, ansatz design (hardware-efficient, UCCSD-like)
- Output PNGs: `vqe_ansatz_designs.png`, `vqe_workflow.png`, `vqe_optimization.png`, `vqe_pauli_measurements.png`, `vqe_applications.png`

### Quantum Error Mitigation Explorer
NISQ-era noise reduction without full error correction.
- Techniques implemented: Zero Noise Extrapolation (ZNE), Measurement Error Mitigation (MEM), Probabilistic Error Cancellation (PEC), symmetry verification, dynamical decoupling
- Key imports: `qiskit_aer.noise.NoiseModel`, `depolarizing_error`, `amplitude_damping_error`, `scipy.optimize.curve_fit`, `scipy.linalg.lstsq`
- Output PNGs: 5 files covering ZNE extrapolation, technique comparison, timeline, measurement error, noise impact

### Quantum Phase Estimation Explorer
Extracts eigenphase of a unitary operator with exponential precision.
- Experiments: 1-qubit, 3-qubit QPE; quantum counting application
- Key import: `from qiskit.circuit.library import QFT`
- Output PNGs: `qpe_1qubit_results.png`, `qpe_3qubit_results.png`, `qpe_algorithm_steps.png`, `qpe_precision.png`, `qpe_quantum_counting.png`

### Quantum Circuit Optimization Explorer
Techniques to reduce gate count, circuit depth, and hardware overhead.
- Covers: gate merging, cancellation, commutation, native gate optimization, 2Q gate reduction, qubit mapping, circuit synthesis, peephole optimization
- Output PNGs: `qco_optimization_basics.png`, `qco_optimization_levels.png`, `qco_optimization_impact.png`, `qco_hardware_profiles.png`, `qco_decision_tree.png`

### Quantum Classification Explorer
Quantum machine learning classification (QNN, VQC, quantum kernels).
- Concepts: feature maps (angle encoding), parameterized ansatz layers, measurement-to-class mapping, classical optimizer loop
- Output PNGs: `qc_feature_encodings.png`, `qc_decision_boundaries.png`, `qc_training_convergence.png`, `qc_quantum_vs_classical.png`, `qc_ml_readiness.png`

---

## Development Guidelines for AI Assistants

### Adding a new Explorer
1. Create a new directory: `New Concept Explorer/`
2. Add the main script: `NewConceptExplorer.py`
3. Follow the standard import block and `run_circuit_and_get_counts` helper exactly
4. Structure experiments with the `# EXPERIMENT N:` section header pattern
5. Print to stdout before showing visuals (batch-style, no `plt.show()` unless explicitly for interactive use)
6. Save PNGs with `dpi=150, bbox_inches='tight'`
7. Add a `Info.txt` or `ReadMe.txt` summarizing experiments and output files

### Modifying existing scripts
- Never hardcode IBM API tokens; keep them in comments or environment variables
- Do not call `plt.show()` — scripts are designed to run non-interactively
- Keep the `run_circuit_and_get_counts` helper signature stable; it's copy-pasted across all files
- `SamplerV2` (imported as `Sampler`) requires ISA circuits — always run `generate_preset_pass_manager` before `sampler.run()`

### IBM Quantum hardware
To switch from simulator to real hardware, replace:
```python
backend = AerSimulator()
```
with:
```python
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=<n>)
```
All scripts are already structured to support this swap without other changes.

### Git workflow
- Active development branch: `claude/claude-md-docs-p9l7zb`
- Main branch: `main`
- Commits have used "Add files via upload" messages historically; prefer descriptive messages for new work

---

## Key Quantum Concepts Covered

| Concept | Explorer |
|---|---|
| Superposition, Hadamard gate | Quantum Superposition Explorer |
| Constructive/destructive interference | Quantum Interference Explorer |
| Entanglement (Bell states) | MyFirstQuantumAlgoritherm.py |
| Deutsch-Jozsa algorithm | Deutsch Jozsa Explorer |
| Grover's search | Grovers Algorithm Explorer |
| Variational hybrid algorithms | Variational Quantum Eigensolver |
| Quantum Fourier Transform (via QPE) | Quantum Phase Estimation Explorer |
| NISQ noise and error mitigation | Quantum Error Mitigation Explorer |
| Circuit transpilation and optimization | Quantum Circuit Optimization Explorer |
| Quantum machine learning | Quantum Classification Explorer |
