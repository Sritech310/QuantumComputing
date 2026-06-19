"""
Quantum Superposition Deep Dive
================================
Explore how quantum superposition works by:
1. Creating superposition states with different angles
2. Measuring probability distributions
3. Visualizing how basis rotations affect outcomes
4. Understanding the relationship between unitary rotations and measurement probabilities

Key Concepts:
- Hadamard gate creates equal superposition (50% |0⟩, 50% |1⟩)
- Rotation gates (RX, RY, RZ) create superposition with different probabilities
- Measurement collapses superposition to a definite state
- Basis rotations change which basis we measure in
"""

# Core Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

import numpy as np
import matplotlib.pyplot as plt
import sys
import qiskit

print("Python:", sys.version.split()[0])
print("Qiskit:", qiskit.__version__)


def run_circuit_and_get_counts(circuit, backend, shots=1000):
    """
    Runs a quantum circuit on a specified backend and returns the measurement counts.

    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        backend: The backend to run the circuit on.
        shots (int): The number of times to run the circuit. Defaults to 1000.

    Returns:
        dict: The measurement counts.
    """
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)
    result = job.result()
    return result[0].data.c.get_counts()


# Initialize backend (using simulator for reproducibility)
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")


# ============================================================================
# EXPERIMENT 1: Basic Superposition with Hadamard
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Basic Superposition (Hadamard Gate)")
print("=" * 70)

"""
The Hadamard gate transforms:
|0⟩ → (|0⟩ + |1⟩)/√2   (equal superposition)
|1⟩ → (|0⟩ - |1⟩)/√2   (equal superposition with phase difference)

This creates a 50-50 probability distribution.
"""

circuit_hadamard = QuantumCircuit(1, 1, name="Hadamard Superposition")
circuit_hadamard.h(0)  # Apply Hadamard
circuit_hadamard.measure(0, 0)

counts_hadamard = run_circuit_and_get_counts(circuit_hadamard, backend, shots=1000)
print(f"Counts: {counts_hadamard}")
print("Expected: ~500 '0's and ~500 '1's (equal superposition)\n")


# ============================================================================
# EXPERIMENT 2: Superposition with RY rotation (different angles)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: Superposition with RY Gate (Variable Angles)")
print("=" * 70)

"""
The RY gate rotates around the Y-axis on the Bloch sphere.
RY(θ) creates a superposition where:
- Probability of |0⟩ = cos²(θ/2)
- Probability of |1⟩ = sin²(θ/2)

By varying θ, we can create different superposition ratios.
"""

angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
angle_labels = ["0°", "30°", "45°", "60°", "90°"]
results_ry = {}

for angle, label in zip(angles, angle_labels):
    circuit_ry = QuantumCircuit(1, 1, name=f"RY({label})")
    circuit_ry.ry(angle, 0)
    circuit_ry.measure(0, 0)

    counts = run_circuit_and_get_counts(circuit_ry, backend, shots=1000)
    results_ry[label] = counts

    # Calculate probabilities
    prob_0 = counts.get('0', 0) / 1000
    prob_1 = counts.get('1', 0) / 1000

    # Theoretical values
    theory_prob_0 = np.cos(angle/2) ** 2
    theory_prob_1 = np.sin(angle/2) ** 2

    print(f"Angle {label:>4}: P(|0⟩) = {prob_0:.3f} (theory: {theory_prob_0:.3f}), "
          f"P(|1⟩) = {prob_1:.3f} (theory: {theory_prob_1:.3f})")

print()


# ============================================================================
# EXPERIMENT 3: Different Rotation Axes (X, Y, Z)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: Superposition with Different Axes (RX, RY, RZ)")
print("=" * 70)

"""
RX, RY, and RZ rotate around different axes on the Bloch sphere.
At 90° rotation on any axis, they all produce similar superposition,
but the measurement in different bases reveals different probabilities.
"""

rotation_angle = np.pi / 2  # 90 degrees

# Create circuits for each rotation
circuit_rx = QuantumCircuit(1, 1, name="RX(90°)")
circuit_rx.rx(rotation_angle, 0)
circuit_rx.measure(0, 0)

circuit_ry = QuantumCircuit(1, 1, name="RY(90°)")
circuit_ry.ry(rotation_angle, 0)
circuit_ry.measure(0, 0)

circuit_rz = QuantumCircuit(1, 1, name="RZ(90°)")
circuit_rz.rz(rotation_angle, 0)
circuit_rz.measure(0, 0)

counts_rx = run_circuit_and_get_counts(circuit_rx, backend, shots=1000)
counts_ry = run_circuit_and_get_counts(circuit_ry, backend, shots=1000)
counts_rz = run_circuit_and_get_counts(circuit_rz, backend, shots=1000)

print(f"RX(90°) counts: {counts_rx}")
print(f"RY(90°) counts: {counts_ry}")
print(f"RZ(90°) counts: {counts_rz}")
print("Note: RZ(90°) produces no superposition (it's just a phase rotation)\n")


# ============================================================================
# EXPERIMENT 4: Basis Rotation - Measuring in Different Bases
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: Basis Rotation (Measuring in Different Bases)")
print("=" * 70)

"""
A superposition state can appear different depending on the basis we measure in.
By applying basis rotation gates before measurement, we can observe
the same state "projected" onto different basis.
"""

# Create a superposition in the Hadamard basis
circuit_basis_computational = QuantumCircuit(1, 1, name="Measure in Computational Basis")
circuit_basis_computational.h(0)
circuit_basis_computational.measure(0, 0)

# Measure the same superposition in the X basis (rotate by π/2 before measurement)
circuit_basis_x = QuantumCircuit(1, 1, name="Measure in X Basis")
circuit_basis_x.h(0)  # Create superposition
circuit_basis_x.ry(np.pi/2, 0)  # Rotate to X basis
circuit_basis_x.measure(0, 0)

# Measure the same superposition in the Y basis
circuit_basis_y = QuantumCircuit(1, 1, name="Measure in Y Basis")
circuit_basis_y.h(0)  # Create superposition
circuit_basis_y.rx(np.pi/2, 0)  # Rotate to Y basis
circuit_basis_y.measure(0, 0)

counts_comp = run_circuit_and_get_counts(circuit_basis_computational, backend, shots=1000)
counts_x = run_circuit_and_get_counts(circuit_basis_x, backend, shots=1000)
counts_y = run_circuit_and_get_counts(circuit_basis_y, backend, shots=1000)

print(f"Computational basis counts: {counts_comp}")
print(f"X basis counts: {counts_x}")
print(f"Y basis counts: {counts_y}")
print("Notice: Same state, different measurement bases → different distributions\n")


# ============================================================================
# EXPERIMENT 5: Multi-Qubit Superposition
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: Multi-Qubit Superposition")
print("=" * 70)

"""
When we apply superposition gates to multiple qubits,
we create a superposition across all possible states.
- 1 qubit with Hadamard: 2 possible states
- 2 qubits with Hadamard: 4 possible states (2²)
- 3 qubits with Hadamard: 8 possible states (2³)
Each with equal probability (1/2^n)
"""

circuit_multi_superposition = QuantumCircuit(3, 3, name="3-Qubit Superposition")
circuit_multi_superposition.h([0, 1, 2])  # Hadamard on all qubits
circuit_multi_superposition.measure([0, 1, 2], [0, 1, 2])

counts_multi = run_circuit_and_get_counts(circuit_multi_superposition, backend, shots=1000)
print(f"3-Qubit superposition counts: {counts_multi}")
print(f"Number of unique states: {len(counts_multi)}")
print(f"Expected: 8 states, each with ~125 counts (1000/8)\n")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Create a 2x2 subplot figure for RY angle variations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Quantum Superposition: RY Gate with Different Angles", fontsize=14, fontweight='bold')

for idx, (angle, label) in enumerate(zip(angles, angle_labels)):
    ax = axes[idx // 3, idx % 3]
    counts = results_ry[label]

    states = ['0', '1']
    values = [counts.get(state, 0) for state in states]

    ax.bar(states, values, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Counts')
    ax.set_title(f'RY({label})')
    ax.set_ylim([0, 1000])

    # Add theoretical line
    theory_0 = np.cos(angle/2) ** 2 * 1000
    ax.axhline(y=theory_0, color='r', linestyle='--', alpha=0.5, label='Theory')
    ax.legend()

# Hide the extra subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('superposition_ry_angles.png', dpi=150, bbox_inches='tight')
print("✓ Saved: superposition_ry_angles.png")


# Create comparison figure for different bases
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Measuring Hadamard Superposition in Different Bases", fontsize=14, fontweight='bold')

bases_data = [
    (counts_comp, "Computational"),
    (counts_x, "X Basis"),
    (counts_y, "Y Basis")
]

for idx, (counts, basis_name) in enumerate(bases_data):
    ax = axes[idx]
    states = ['0', '1']
    values = [counts.get(state, 0) for state in states]

    ax.bar(states, values, color=['#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Counts')
    ax.set_title(f'{basis_name}')
    ax.set_ylim([0, 1000])

plt.tight_layout()
plt.savefig('superposition_bases.png', dpi=150, bbox_inches='tight')
print("✓ Saved: superposition_bases.png")


# Create histogram for multi-qubit superposition
fig, ax = plt.subplots(figsize=(12, 6))
plot_histogram(counts_multi, ax=ax)
plt.title("3-Qubit Equal Superposition Distribution")
plt.savefig('superposition_multi_qubit.png', dpi=150, bbox_inches='tight')
print("✓ Saved: superposition_multi_qubit.png")

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE!")
print("=" * 70)
print("\nNext Steps:")
print("1. Modify the rotation angles and observe how probabilities change")
print("2. Try different combinations of rotation gates")
print("3. Explore how adding more qubits increases the superposition complexity")
print("4. Study how basis rotations reveal different properties of the same state")
print("5. Build a parameterized circuit where you sweep angles programmatically\n")
