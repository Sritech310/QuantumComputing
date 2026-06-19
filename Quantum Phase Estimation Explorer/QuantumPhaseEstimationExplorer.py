"""
Quantum Phase Estimation & Quantum Counting
=============================================
Explore Quantum Phase Estimation (QPE) - a fundamental algorithm for:
1. Estimating eigenvalues/phases of quantum operators
2. Counting marked items in databases (Quantum Counting)
3. Building blocks for Shor's and VQE algorithms

What is Phase Estimation?

The Problem:
Given a unitary operator U and an eigenstate |ψ⟩ such that:
    U|ψ⟩ = e^(2πiθ)|ψ⟩
Estimate the phase θ (where 0 ≤ θ < 1)

Classical Approach:
- Apply U repeatedly, measure relative phases
- Requires many experiments and classical post-processing
- Polynomial time complexity

Quantum Approach (Phase Estimation):
- Use n "counting" qubits + work qubits
- Apply controlled-U^(2^j) operations (parallel)
- Inverse QFT extracts the phase
- Single measurement gives high-precision estimate
- Exponentially better precision with linear resources

The Algorithm:
1. Initialize n counting qubits in superposition
2. Initialize work qubits in eigenstate |ψ⟩
3. Apply controlled-U^(2^0), controlled-U^(2^1), ..., controlled-U^(2^(n-1))
   (controlled by each counting qubit)
4. Apply inverse Quantum Fourier Transform
5. Measure counting qubits to get θ (as n-bit binary number)

Precision:
- With n counting qubits, phase estimated to precision 2^(-n)
- 3 qubits: precision 1/8 = 0.125
- 4 qubits: precision 1/16 = 0.0625
- 5 qubits: precision 1/32 = 0.03125

Applications:
- Quantum Counting: Count marked items in N items using Grover operator
- Shor's Algorithm: Find order of element (used in factoring)
- VQE: Estimate ground state energy of molecules
"""

# Core Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT

import numpy as np
import matplotlib.pyplot as plt
import sys
import qiskit

print("Python:", sys.version.split()[0])
print("Qiskit:", qiskit.__version__)


def run_circuit_and_get_counts(circuit, backend, shots=1000):
    """
    Runs a quantum circuit on a specified backend and returns the measurement counts.
    """
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)

    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)
    result = job.result()
    return result[0].data.c.get_counts()


def get_statevector(circuit):
    """Get the statevector without measurement."""
    return Statevector.from_instruction(circuit)


def phase_to_binary(phase, n_bits):
    """Convert phase (0 to 1) to n-bit binary representation"""
    int_val = int(phase * (2 ** n_bits))
    return format(int_val, f'0{n_bits}b')


# Initialize backend
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")


# ============================================================================
# EXPERIMENT 1: Understanding Eigenvalues and Phases
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Eigenvalues, Eigenstates, and Phases")
print("=" * 70)

print("\nFor a unitary operator U, we have:")
print("  U|ψ⟩ = λ|ψ⟩  (eigenvalue equation)")
print("  where λ = e^(2πiθ) is the eigenvalue (on unit circle)")
print("  and θ ∈ [0,1) is the phase we want to estimate")

print("\nExamples:")
print("  θ = 0.0   → λ = e^(2πi·0.0) = 1           (identity)")
print("  θ = 0.25  → λ = e^(2πi·0.25) = i          (90° rotation)")
print("  θ = 0.5   → λ = e^(2πi·0.5) = -1          (180° rotation)")
print("  θ = 0.75  → λ = e^(2πi·0.75) = -i         (270° rotation)")

print("\nCommon Unitaries:")
print("  Z gate: eigenvalues 1, -1 → phases 0, 0.5")
print("  S gate (T^2): eigenvalues 1, i → phases 0, 0.25")
print("  T gate: eigenvalues 1, e^(iπ/4) → phases 0, 0.125")
print()


# ============================================================================
# EXPERIMENT 2: Simple Phase Estimation (1 Counting Qubit)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: 1-Qubit Phase Estimation (Coarse Estimate)")
print("=" * 70)

"""
With 1 counting qubit, we can distinguish:
- θ ∈ [0, 0.5): measure 0
- θ ∈ [0.5, 1): measure 1

This gives us the most significant bit of θ.
"""

print("\nEstimating phase of Z^(1/2) gate (eigenvalue i, phase 0.25)\n")

print("Test 1: Eigenstate |0⟩ of Z gate (eigenvalue +1, phase 0.0)")
circuit_1q_z_plus = QuantumCircuit(2, 1, name="1Q-PE: Z|0⟩")
circuit_1q_z_plus.h(0)  # Counting qubit in superposition
# Controlled-Z^1 with control=0, target=1
# |0⟩ is +1 eigenstate of Z, so contributes 0 phase
circuit_1q_z_plus.cz(0, 1)
circuit_1q_z_plus.h(0)  # Inverse QFT (just H for 1 qubit)
circuit_1q_z_plus.measure(0, 0)

counts_z_plus = run_circuit_and_get_counts(circuit_1q_z_plus, backend, shots=1000)
print(f"Counts: {counts_z_plus}")
print(f"Expected: Mostly 0 (phase ≈ 0.0)")

print("\nTest 2: Eigenstate |1⟩ of Z gate (eigenvalue -1, phase 0.5)")
circuit_1q_z_minus = QuantumCircuit(2, 1, name="1Q-PE: Z|1⟩")
circuit_1q_z_minus.h(0)  # Counting qubit in superposition
circuit_1q_z_minus.x(1)  # Prepare |1⟩ eigenstate
# Controlled-Z^1 with control=0, target=1
# |1⟩ is -1 eigenstate of Z, so contributes 0.5 phase
circuit_1q_z_minus.cz(0, 1)
circuit_1q_z_minus.h(0)  # Inverse QFT
circuit_1q_z_minus.measure(0, 0)

counts_z_minus = run_circuit_and_get_counts(circuit_1q_z_minus, backend, shots=1000)
print(f"Counts: {counts_z_minus}")
print(f"Expected: Mostly 1 (phase ≈ 0.5)")
print()


# ============================================================================
# EXPERIMENT 3: 3-Qubit Phase Estimation (Higher Precision)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: 3-Qubit Phase Estimation (Precise Estimate)")
print("=" * 70)

"""
With 3 counting qubits, we can estimate phases to precision 1/8 = 0.125.
We'll estimate the phase of S gate (phase 0.25) and T gate (phase 0.125).
"""

print("\nEstimating phase of S gate (e^(iπ/2), eigenvalue i, phase 0.25)\n")

# S gate eigenstate: |+⟩ (superposition of eigenstates)
# But we'll use |0⟩ which is eigenstate with λ=1 (phase 0)
# And |1⟩ which gives more interesting phases

def phase_estimation_3q_s_gate_zero():
    """3-qubit phase estimation for S|0⟩"""
    qc = QuantumCircuit(4, 3, name="3Q-PE: S|0⟩")

    # Initialize counting qubits in superposition
    qc.h([0, 1, 2])

    # Controlled-S^(2^0) = Controlled-S
    qc.cu(0, 0, np.pi/2, 0, 0, 3)  # Controlled-S from qubit 0 to qubit 3

    # Controlled-S^(2^1) = Controlled-S^2 = Controlled-(-1)
    # S^2 = [[1,0],[0,-1]], eigenvalue -1 for |1⟩
    for _ in range(2):
        qc.cu(0, 0, np.pi/2, 0, 0, 3)

    # Controlled-S^(2^2) = Controlled-S^4 = Controlled-I (identity, no effect)
    # S^4 = I

    # Inverse QFT on counting qubits
    qft_inv = QFT(3, inverse=True, do_swaps=False)
    qc.append(qft_inv, [0, 1, 2])

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def phase_estimation_3q_t_gate():
    """3-qubit phase estimation for T gate (phase 0.125)"""
    qc = QuantumCircuit(4, 3, name="3Q-PE: T")

    # Initialize counting qubits in superposition
    qc.h([0, 1, 2])

    # T gate: diagonal [[1,0],[0,e^(iπ/4)]]
    # We'll approximate with phase rotation
    # Controlled-T from qubit 0 to qubit 3
    qc.cu(0, 0, np.pi/4, 0, 0, 3)

    # Controlled-T^2
    for _ in range(2):
        qc.cu(0, 0, np.pi/4, 0, 0, 3)

    # Controlled-T^4
    for _ in range(4):
        qc.cu(0, 0, np.pi/4, 0, 0, 3)

    # Inverse QFT
    qft_inv = QFT(3, inverse=True, do_swaps=False)
    qc.append(qft_inv, [0, 1, 2])

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


print("Test 1: Phase estimation of S gate")
circuit_3q_s = phase_estimation_3q_s_gate_zero()
counts_3q_s = run_circuit_and_get_counts(circuit_3q_s, backend, shots=1000)

print(f"Counts: {counts_3q_s}")
print(f"Expected: Phases near 0.0 (S|0⟩ eigenvalue 1, S|1⟩ eigenvalue i)")

# Find most likely measurement
if counts_3q_s:
    most_likely = max(counts_3q_s.items(), key=lambda x: x[1])[0]
    estimated_phase = int(most_likely, 2) / 8
    print(f"Most likely result: {most_likely} → phase ≈ {estimated_phase:.3f}")

print("\nTest 2: Phase estimation of T gate")
circuit_3q_t = phase_estimation_3q_t_gate()
counts_3q_t = run_circuit_and_get_counts(circuit_3q_t, backend, shots=1000)

print(f"Counts: {counts_3q_t}")
print(f"Expected: Phase 0.125 (T eigenvalue e^(iπ/4))")

if counts_3q_t:
    most_likely = max(counts_3q_t.items(), key=lambda x: x[1])[0]
    estimated_phase = int(most_likely, 2) / 8
    print(f"Most likely result: {most_likely} → phase ≈ {estimated_phase:.3f}")
print()


# ============================================================================
# EXPERIMENT 4: Precision Analysis (Varying Number of Qubits)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: Precision vs Number of Counting Qubits")
print("=" * 70)

"""
More counting qubits → better precision.
With n qubits: precision = 1/2^n
"""

print("\nPhase estimation precision for different numbers of counting qubits:\n")
print(f"{'Qubits':>8} | {'Precision':>12} | {'Distinguishable Phases':>25}")
print("-" * 50)

precisions = []
distinguishable = []

for n_qubits in range(1, 11):
    precision = 1 / (2 ** n_qubits)
    n_phases = 2 ** n_qubits
    precisions.append(precision)
    distinguishable.append(n_phases)

    print(f"{n_qubits:>8} | {precision:>12.6f} | {n_phases:>25}")

print()


# ============================================================================
# EXPERIMENT 5: Quantum Counting (Using Grover Operator)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: Quantum Counting - Count Marked Items")
print("=" * 70)

"""
Quantum Counting uses Phase Estimation with the Grover operator.
It counts how many items are marked in a database.

The Grover operator has eigenvalues related to the number of marked items.
By estimating the phase of Grover operator, we can determine the count.

For M marked items out of N total:
  Phase θ ≈ arcsin(√(M/N)) / π

From phase θ, we can recover: M ≈ N * sin²(πθ)
"""

print("\nQuantum Counting Setup:")
print("  Database size: N items")
print("  Marked items: M items")
print("  Grover operator G = DS (diffusion × oracle)")
print("  Phase of G relates to M/N ratio")
print()

print("Classical vs Quantum Counting:")
print("  Classical: Must check ~M items on average (or use other heuristics)")
print("  Quantum: Use phase estimation on Grover operator")
print("  Speedup: √N for exact count\n")

# For demonstration, we'll simulate counting 2 marked items out of 4
print("Example: Count marked items in 4-item database")
print("  Total items (N): 4")
print("  Marked items (M): 2")
print("  Ratio (M/N): 0.5")
print("  Theoretical phase: arcsin(√0.5) / π ≈ 0.207")
print("  (This is where Grover operator eigenvalue concentrates)\n")

# Create a simplified quantum counting circuit
def quantum_counting_2marked_4total():
    """Simplified: count 2 marked out of 4 using phase estimation"""
    # In real implementation, this would use controlled Grover operators
    # Here we simulate the result
    qc = QuantumCircuit(3, 3, name="Quantum Counting (2/4)")

    # Initialize counting qubits
    qc.h([0, 1, 2])

    # We'll use a controlled phase rotation to simulate Grover eigenvalue
    # Theoretical phase for 2 marked out of 4: ~0.207
    # In 3-qubit representation: 0.207 * 8 ≈ 1.66 → bit pattern ~001 or ~010

    # Simulate with phase rotation (simplified)
    target_phase = 0.207
    qc.p(2 * np.pi * target_phase, 0)  # Apply phase to simulate eigenvalue

    # Inverse QFT
    qft_inv = QFT(3, inverse=True, do_swaps=False)
    qc.append(qft_inv, [0, 1, 2])

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

circuit_qc = quantum_counting_2marked_4total()
counts_qc = run_circuit_and_get_counts(circuit_qc, backend, shots=1000)

print("Quantum Counting Result:")
print(f"Counts: {counts_qc}")

if counts_qc:
    # Find most likely measurement
    most_likely_bitstring = max(counts_qc.items(), key=lambda x: x[1])[0]
    estimated_phase = int(most_likely_bitstring, 2) / 8

    # Recover count: M ≈ N * sin²(πθ)
    N = 4
    estimated_M = N * (np.sin(np.pi * estimated_phase)) ** 2

    print(f"Most likely phase bits: {most_likely_bitstring} → phase ≈ {estimated_phase:.3f}")
    print(f"Estimated count: M ≈ {N} × sin²(π × {estimated_phase:.3f}) ≈ {estimated_M:.1f}")
    print(f"Actual count: 2")
    print(f"Error: {abs(estimated_M - 2):.2f}")
print()


# ============================================================================
# EXPERIMENT 6: Eigenvalue Estimation for Different Unitaries
# ============================================================================
print("=" * 70)
print("EXPERIMENT 6: Eigenvalue Estimation for Different Unitaries")
print("=" * 70)

print("\nEstimating eigenvalues of various quantum gates:\n")

# For |0⟩ eigenstate
unitaries_0 = [
    ("I (Identity)", 0.0),
    ("Z (Pauli-Z)", 0.0),  # |0⟩ is +1 eigenstate
    ("S (Phase)", 0.0),    # |0⟩ is +1 eigenstate
    ("T (π/4 phase)", 0.0), # |0⟩ is +1 eigenstate
]

# For |1⟩ eigenstate
unitaries_1 = [
    ("Z (Pauli-Z)", 0.5),      # |1⟩ is -1 eigenstate (phase 0.5)
    ("S (Phase)", 0.25),       # |1⟩ is i eigenstate (phase 0.25)
    ("T (π/4 phase)", 0.125),  # |1⟩ is e^(iπ/4) eigenstate (phase 0.125)
]

print("Eigenvalues for |0⟩ basis state:")
for gate, phase in unitaries_0:
    eigenvalue = np.exp(2j * np.pi * phase)
    print(f"  {gate:20} → phase = {phase:.3f} (eigenvalue = {eigenvalue:.3f})")

print("\nEigenvalues for |1⟩ basis state:")
for gate, phase in unitaries_1:
    eigenvalue = np.exp(2j * np.pi * phase)
    print(f"  {gate:20} → phase = {phase:.3f} (eigenvalue = {eigenvalue:.3f})")
print()


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: 1-Qubit Phase Estimation Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("1-Qubit Phase Estimation: Z Gate Eigenstates", fontsize=14, fontweight='bold')

# Z|0⟩ (phase 0)
ax1.bar(['0', '1'], [counts_z_plus.get('0', 0), counts_z_plus.get('1', 0)],
        color=['#2ca02c', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Counts', fontsize=12)
ax1.set_title('Z|0⟩ (phase ≈ 0.0)', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1000])
ax1.text(0.5, 0.95, 'Eigenvalue: +1', transform=ax1.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), fontsize=11)

# Z|1⟩ (phase 0.5)
ax2.bar(['0', '1'], [counts_z_minus.get('0', 0), counts_z_minus.get('1', 0)],
        color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Counts', fontsize=12)
ax2.set_title('Z|1⟩ (phase ≈ 0.5)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1000])
ax2.text(0.5, 0.95, 'Eigenvalue: -1', transform=ax2.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7), fontsize=11)

plt.tight_layout()
plt.savefig('qpe_1qubit_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qpe_1qubit_results.png")


# Figure 2: 3-Qubit Phase Estimation Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("3-Qubit Phase Estimation: Precision = 1/8", fontsize=14, fontweight='bold')

# S gate
states_3q = [format(i, '03b') for i in range(8)]
values_s = [counts_3q_s.get(state, 0) for state in states_3q]
colors_s = ['#2ca02c' if v > 50 else '#1f77b4' for v in values_s]

ax1.bar(range(8), values_s, color=colors_s, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(8))
ax1.set_xticklabels(states_3q, fontsize=10)
ax1.set_ylabel('Counts', fontsize=12)
ax1.set_title('S Gate Eigenvalue Estimation', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1000])

# T gate
values_t = [counts_3q_t.get(state, 0) for state in states_3q]
colors_t = ['#2ca02c' if v > 50 else '#1f77b4' for v in values_t]

ax2.bar(range(8), values_t, color=colors_t, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(8))
ax2.set_xticklabels(states_3q, fontsize=10)
ax2.set_ylabel('Counts', fontsize=12)
ax2.set_title('T Gate Eigenvalue Estimation (phase ≈ 0.125)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1000])

plt.tight_layout()
plt.savefig('qpe_3qubit_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qpe_3qubit_results.png")


# Figure 3: Precision vs Number of Qubits
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase Estimation Precision and Resolution", fontsize=14, fontweight='bold')

qubits_range = np.arange(1, 11)
precision_vals = [1 / (2**n) for n in qubits_range]
distinguishable_vals = [2**n for n in qubits_range]

# Panel 1: Precision
ax1.semilogy(qubits_range, precision_vals, 'o-', linewidth=2.5, markersize=10, color='#1f77b4')
ax1.set_xlabel('Number of Counting Qubits', fontsize=12)
ax1.set_ylabel('Phase Precision (log scale)', fontsize=12)
ax1.set_title('Precision Improvement')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(qubits_range)

# Panel 2: Distinguishable phases
ax2.semilogy(qubits_range, distinguishable_vals, 's-', linewidth=2.5, markersize=10, color='#ff7f0e')
ax2.set_xlabel('Number of Counting Qubits', fontsize=12)
ax2.set_ylabel('Number of Distinguishable Phases (log scale)', fontsize=12)
ax2.set_title('Resolution Improvement')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(qubits_range)

# Add annotations
ax1.text(8, 1e-3, 'Exponential\nimprovement', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax2.text(8, 1e2, 'Exponential\nimprovement', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('qpe_precision.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qpe_precision.png")


# Figure 4: Phase Estimation Circuit Diagram (Conceptual)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Quantum Phase Estimation: Algorithm Steps", fontsize=14, fontweight='bold')

# Step 1: Initialization
ax = axes[0, 0]
ax.text(0.5, 0.85, "Step 1: Initialize", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.1, 0.65, "Counting qubits:", fontsize=11, transform=ax.transAxes, fontweight='bold')
ax.text(0.15, 0.55, "Apply H gates", fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.48, "→ Superposition", fontsize=10, transform=ax.transAxes)

ax.text(0.1, 0.35, "Work qubits:", fontsize=11, transform=ax.transAxes, fontweight='bold')
ax.text(0.15, 0.25, "Initialize to |ψ⟩", fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.18, "(eigenstate of U)", fontsize=10, transform=ax.transAxes)

ax.text(0.5, 0.05, "State: |superposition⟩|ψ⟩", ha='center', fontsize=10,
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 2: Controlled Unitaries
ax = axes[0, 1]
ax.text(0.5, 0.85, "Step 2: Controlled U Gates", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.1, 0.7, "Apply controlled operations:", fontsize=11, transform=ax.transAxes)
ax.text(0.15, 0.6, "CU^(2^0) from counting qubit 0", fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.15, 0.52, "CU^(2^1) from counting qubit 1", fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.15, 0.44, "CU^(2^2) from counting qubit 2", fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.15, 0.36, "... (parallel operations)", fontsize=10, transform=ax.transAxes)

ax.text(0.5, 0.15, "Phase encoded in", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.05, "superposition of counting qubits", ha='center', fontsize=10,
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 3: Inverse QFT
ax = axes[1, 0]
ax.text(0.5, 0.85, "Step 3: Inverse QFT", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.1, 0.65, "Apply Inverse Quantum", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.55, "Fourier Transform:", fontsize=11, transform=ax.transAxes)
ax.text(0.15, 0.42, "• Extracts phase bits", fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.34, "• Converts phase info", fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.26, "  to basis states", fontsize=10, transform=ax.transAxes)

ax.text(0.5, 0.05, "Phase now in basis", ha='center', fontsize=10,
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 4: Measurement
ax = axes[1, 1]
ax.text(0.5, 0.85, "Step 4: Measure", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.1, 0.65, "Measure counting qubits", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.55, "Get binary string b", fontsize=11, transform=ax.transAxes)
ax.text(0.15, 0.42, "Phase estimate:", fontsize=10, transform=ax.transAxes, fontweight='bold')
ax.text(0.15, 0.32, "θ ≈ b / 2^n", fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.15, 0.22, "Precision: Δθ = 1/2^n", fontsize=10, transform=ax.transAxes, family='monospace')

ax.text(0.5, 0.05, "Result: Phase estimate", ha='center', fontsize=10,
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

plt.tight_layout()
plt.savefig('qpe_algorithm_steps.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qpe_algorithm_steps.png")


# Figure 5: Quantum Counting Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Quantum Counting: From Phase to Item Count", fontsize=14, fontweight='bold')

# Panel 1: Phase distribution
ax1.bar(range(8), [counts_qc.get(format(i, '03b'), 0) for i in range(8)],
        color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Phase Bits', fontsize=12)
ax1.set_ylabel('Counts', fontsize=12)
ax1.set_title('Measured Phase Bits')
ax1.set_xticks(range(8))
ax1.set_xticklabels([format(i, '03b') for i in range(8)], fontsize=10)
ax1.set_ylim([0, 1000])

# Panel 2: Relationship between phase and count
phases = np.linspace(0, 1, 100)
N = 4
counts_from_phase = N * np.sin(np.pi * phases) ** 2

ax2.plot(phases, counts_from_phase, linewidth=2.5, color='#ff7f0e')
ax2.axhline(y=2, color='r', linestyle='--', linewidth=2, alpha=0.5, label='True count (M=2)')
ax2.fill_between(phases, 0, counts_from_phase, alpha=0.3, color='#ff7f0e')

ax2.set_xlabel('Estimated Phase', fontsize=12)
ax2.set_ylabel('Item Count (M)', fontsize=12)
ax2.set_title('Phase → Count Mapping\nM = N × sin²(πθ)')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, N])

plt.tight_layout()
plt.savefig('qpe_quantum_counting.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qpe_quantum_counting.png")


print("\n" + "=" * 70)
print("QUANTUM PHASE ESTIMATION EXPLORATION COMPLETE!")
print("=" * 70)

print("\nKey Insights:")
print("✓ Phase estimation: Measure eigenvalues of quantum operators")
print("✓ Precision: n counting qubits → precision 1/2^n")
print("✓ Uses inverse QFT to extract phase information")
print("✓ Exponential precision improvement with linear qubit increase")
print("✓ Quantum Counting: Count marked items using Grover operator phase")

print("\nPrecision Table:")
print("  1 qubit: ±0.5   (distinguishes 2 phases)")
print("  2 qubits: ±0.25 (distinguishes 4 phases)")
print("  3 qubits: ±0.125 (distinguishes 8 phases)")
print("  n qubits: ±1/2^n (distinguishes 2^n phases)")

print("\nApplications:")
print("✓ Quantum Counting: Count marked items in O(1) oracle calls")
print("✓ Shor's Algorithm: Find order (used for factoring)")
print("✓ VQE: Estimate ground state energy")
print("✓ Quantum Simulation: Extract energy eigenvalues")

print("\nNext Steps:")
print("1. Understand Quantum Fourier Transform (QFT) deeper")
print("2. Implement with custom unitary operators")
print("3. Explore Grover operator eigenvalues")
print("4. Study VQE (hybrid quantum-classical algorithm)")
print("5. Build toward Shor's Algorithm for factoring\n")
