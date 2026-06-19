"""
Deutsch-Jozsa Algorithm - Function Property Determination
==========================================================
Explore the Deutsch-Jozsa Algorithm - a quantum algorithm that determines
function properties with unprecedented efficiency!

The Problem:
Given a function f: {0,1}^n → {0,1}, determine if f is:
  - CONSTANT: f(x) = c for all x (either always 0 or always 1)
  - BALANCED: f(x) = 0 for exactly half the inputs, 1 for the other half

Classical Approach:
  - Need to evaluate f up to 2^(n-1) + 1 times
  - For n=3: up to 5 evaluations
  - For n=100: up to 2^99 + 1 evaluations (impossible!)

Quantum Approach (Deutsch-Jozsa):
  - Need to evaluate f exactly ONCE
  - For any n: just 1 evaluation!
  - Exponential speedup!

The Algorithm:
1. Initialize n qubits in |0⟩ and 1 ancilla in |1⟩
2. Apply Hadamard to all qubits (create superposition + ancilla to |->)
3. Apply oracle U_f (encodes the function f)
4. Apply Hadamard to the first n qubits
5. Measure the first n qubits
   - All 0s → CONSTANT function
   - At least one 1 → BALANCED function

Why It Works:
  - Quantum interference amplifies the function property
  - The oracle phase-encodes the function output
  - Hadamard interference reveals global property (constant vs balanced)
  - Classical: Must check each case. Quantum: One interference pattern tells all!
"""

# Core Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import Statevector

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
        shots (int): The number of times to run the circuit.

    Returns:
        dict: The measurement counts.
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


# Initialize backend
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")


# ============================================================================
# EXPERIMENT 1: Understanding Function Properties
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Function Properties Explained")
print("=" * 70)

print("\nFor a function f: {0,1}^n → {0,1}")
print("\nCONSTANT functions (2 types):")
print("  f(0...0) = c, f(0...1) = c, f(1...0) = c, f(1...1) = c")
print("  Example (n=2): f(00)=0, f(01)=0, f(10)=0, f(11)=0")
print("  or            f(00)=1, f(01)=1, f(10)=1, f(11)=1")

print("\nBALANCED functions (many types):")
print("  Exactly 2^(n-1) inputs map to 0, and 2^(n-1) map to 1")
print("  Example (n=2): f(00)=0, f(01)=1, f(10)=0, f(11)=1")
print("  or            f(00)=0, f(01)=0, f(10)=1, f(11)=1")

print("\nCLASSICAL CHALLENGE:")
print("  Worst case: Must evaluate function 2^(n-1) + 1 times")
print("  n=1: 2 evaluations (half of 2 inputs)")
print("  n=2: 3 evaluations (half of 4 inputs + 1)")
print("  n=3: 5 evaluations (half of 8 inputs + 1)")
print("  n=20: 524,289 evaluations!")

print("\nQUANTUM (DEUTSCH-JOZSA) ADVANTAGE:")
print("  Always: Exactly 1 oracle evaluation!")
print("  Works for any n\n")


# ============================================================================
# EXPERIMENT 2: 1-Qubit Deutsch Algorithm
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: 1-Qubit Deutsch Algorithm (Simple Case)")
print("=" * 70)

"""
The 1-qubit Deutsch algorithm is a special case of Deutsch-Jozsa.
With 1 qubit, there are only 4 possible functions:
  f₀(x) = 0          (constant-0)
  f₁(x) = 1          (constant-1)
  f₂(x) = x          (balanced: identity)
  f₃(x) = NOT x      (balanced: negation)
"""

print("\nFor 1 qubit, there are 4 possible functions:")
print("  f₀ = constant-0 (always 0)")
print("  f₁ = constant-1 (always 1)")
print("  f₂ = identity   (balanced: f(0)=0, f(1)=1)")
print("  f₃ = NOT        (balanced: f(0)=1, f(1)=0)\n")


def deutsch_circuit_constant_0():
    """Deutsch circuit for f(x) = 0 (constant)"""
    qc = QuantumCircuit(2, 1, name="Deutsch f=constant-0")
    qc.x(1)  # Initialize ancilla to |1⟩
    qc.h([0, 1])  # Hadamard on both
    # Oracle: do nothing (identity)
    qc.h(0)  # Hadamard on input qubit
    qc.measure(0, 0)
    return qc


def deutsch_circuit_constant_1():
    """Deutsch circuit for f(x) = 1 (constant)"""
    qc = QuantumCircuit(2, 1, name="Deutsch f=constant-1")
    qc.x(1)  # Initialize ancilla to |1⟩
    qc.h([0, 1])  # Hadamard on both
    # Oracle: apply Z (phase flip) to ancilla
    qc.z(1)
    qc.h(0)  # Hadamard on input qubit
    qc.measure(0, 0)
    return qc


def deutsch_circuit_identity():
    """Deutsch circuit for f(x) = x (balanced: identity)"""
    qc = QuantumCircuit(2, 1, name="Deutsch f=identity")
    qc.x(1)  # Initialize ancilla to |1⟩
    qc.h([0, 1])  # Hadamard on both
    # Oracle: CNOT (XOR with input)
    qc.cx(0, 1)
    qc.h(0)  # Hadamard on input qubit
    qc.measure(0, 0)
    return qc


def deutsch_circuit_not():
    """Deutsch circuit for f(x) = NOT x (balanced: negation)"""
    qc = QuantumCircuit(2, 1, name="Deutsch f=NOT")
    qc.x(1)  # Initialize ancilla to |1⟩
    qc.h([0, 1])  # Hadamard on both
    # Oracle: X on ancilla, then CNOT, then X again (XOR with inverted input)
    qc.x(1)
    qc.cx(0, 1)
    qc.x(1)
    qc.h(0)  # Hadamard on input qubit
    qc.measure(0, 0)
    return qc


deutsch_circuits_1q = [
    (deutsch_circuit_constant_0(), "constant-0", "CONSTANT"),
    (deutsch_circuit_constant_1(), "constant-1", "CONSTANT"),
    (deutsch_circuit_identity(), "identity", "BALANCED"),
    (deutsch_circuit_not(), "NOT", "BALANCED")
]

print("Testing all 4 functions:\n")

deutsch_1q_results = []

for circuit, func_name, func_type in deutsch_circuits_1q:
    counts = run_circuit_and_get_counts(circuit, backend, shots=1000)

    # Result: 0 means constant, 1 means balanced
    measurement_0 = counts.get('0', 0)
    measurement_1 = counts.get('1', 0)

    is_constant = measurement_1 == 0
    predicted_type = "CONSTANT" if is_constant else "BALANCED"

    print(f"f(x) = {func_name:15} | Actual: {func_type:8} | Predicted: {predicted_type:8} | "
          f"Measured: {measurement_0} zeros, {measurement_1} ones")

    deutsch_1q_results.append({
        'function': func_name,
        'actual_type': func_type,
        'predicted_type': predicted_type,
        'counts': counts,
        'is_correct': func_type == predicted_type
    })

print()


# ============================================================================
# EXPERIMENT 3: 2-Qubit Deutsch-Jozsa Algorithm
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: 2-Qubit Deutsch-Jozsa Algorithm")
print("=" * 70)

"""
With 2 qubits, there are 2^2 = 4 possible inputs and 2^4 = 16 possible functions!
We'll test a few representative ones:
  - Constant-0: All inputs map to 0
  - Constant-1: All inputs map to 1
  - Balanced-1: (f(00), f(01), f(10), f(11)) = (0,0,1,1)
  - Balanced-2: (f(00), f(01), f(10), f(11)) = (0,1,0,1)
"""

print("\nTesting 2-qubit functions with inputs: 00, 01, 10, 11\n")


def dj_circuit_constant_0_2q():
    """Deutsch-Jozsa for f = constant-0 (2 qubits)"""
    qc = QuantumCircuit(3, 2, name="DJ f=0 (2q)")
    qc.x(2)  # Ancilla to |1⟩
    qc.h([0, 1, 2])  # Hadamard on all
    # Oracle: do nothing
    qc.h([0, 1])  # Hadamard on input qubits
    qc.measure([0, 1], [0, 1])
    return qc


def dj_circuit_constant_1_2q():
    """Deutsch-Jozsa for f = constant-1 (2 qubits)"""
    qc = QuantumCircuit(3, 2, name="DJ f=1 (2q)")
    qc.x(2)  # Ancilla to |1⟩
    qc.h([0, 1, 2])  # Hadamard on all
    # Oracle: phase flip on ancilla
    qc.z(2)
    qc.h([0, 1])  # Hadamard on input qubits
    qc.measure([0, 1], [0, 1])
    return qc


def dj_circuit_balanced_1_2q():
    """Deutsch-Jozsa for balanced function (2 qubits)
    f(00)=0, f(01)=0, f(10)=1, f(11)=1
    Implements: f(x1,x2) = x1 (depends only on first qubit)
    """
    qc = QuantumCircuit(3, 2, name="DJ balanced-1 (2q)")
    qc.x(2)  # Ancilla to |1⟩
    qc.h([0, 1, 2])  # Hadamard on all
    # Oracle: CNOT from qubit 0 to ancilla (f depends on first input)
    qc.cx(0, 2)
    qc.h([0, 1])  # Hadamard on input qubits
    qc.measure([0, 1], [0, 1])
    return qc


def dj_circuit_balanced_2_2q():
    """Deutsch-Jozsa for balanced function (2 qubits)
    f(00)=0, f(01)=1, f(10)=0, f(11)=1
    Implements: f(x1,x2) = x2 (depends only on second qubit)
    """
    qc = QuantumCircuit(3, 2, name="DJ balanced-2 (2q)")
    qc.x(2)  # Ancilla to |1⟩
    qc.h([0, 1, 2])  # Hadamard on all
    # Oracle: CNOT from qubit 1 to ancilla (f depends on second input)
    qc.cx(1, 2)
    qc.h([0, 1])  # Hadamard on input qubits
    qc.measure([0, 1], [0, 1])
    return qc


dj_circuits_2q = [
    (dj_circuit_constant_0_2q(), "constant-0", "CONSTANT"),
    (dj_circuit_constant_1_2q(), "constant-1", "CONSTANT"),
    (dj_circuit_balanced_1_2q(), "balanced-1: f=x₁", "BALANCED"),
    (dj_circuit_balanced_2_2q(), "balanced-2: f=x₂", "BALANCED")
]

print("Testing 2-qubit functions:\n")

dj_2q_results = []

for circuit, func_name, func_type in dj_circuits_2q:
    counts = run_circuit_and_get_counts(circuit, backend, shots=1000)

    # Check if result is all-zero (constant) or has at least one 1 (balanced)
    all_zero_count = counts.get('00', 0)
    other_counts = sum(v for k, v in counts.items() if k != '00')

    is_constant = other_counts == 0
    predicted_type = "CONSTANT" if is_constant else "BALANCED"

    print(f"f = {func_name:20} | Actual: {func_type:8} | Predicted: {predicted_type:8} | "
          f"Result: {counts}")

    dj_2q_results.append({
        'function': func_name,
        'actual_type': func_type,
        'predicted_type': predicted_type,
        'counts': counts,
        'is_constant': is_constant
    })

print()


# ============================================================================
# EXPERIMENT 4: 3-Qubit Deutsch-Jozsa Algorithm
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: 3-Qubit Deutsch-Jozsa Algorithm")
print("=" * 70)

"""
With 3 qubits, we have 8 possible inputs: 000, 001, 010, 011, 100, 101, 110, 111
There are 2^8 = 256 possible functions!

We'll test:
  - Constant-0: All 8 inputs → 0
  - Constant-1: All 8 inputs → 1
  - Balanced functions (4 inputs → 0, 4 inputs → 1)
"""

print("\nTesting 3-qubit functions with inputs: 000-111\n")


def dj_circuit_constant_0_3q():
    """DJ for f = constant-0 (3 qubits)"""
    qc = QuantumCircuit(4, 3, name="DJ f=0 (3q)")
    qc.x(3)
    qc.h([0, 1, 2, 3])
    # Oracle: do nothing
    qc.h([0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def dj_circuit_constant_1_3q():
    """DJ for f = constant-1 (3 qubits)"""
    qc = QuantumCircuit(4, 3, name="DJ f=1 (3q)")
    qc.x(3)
    qc.h([0, 1, 2, 3])
    # Oracle: phase flip
    qc.z(3)
    qc.h([0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def dj_circuit_balanced_odd_parity_3q():
    """DJ for balanced function with odd parity
    f(x) = x₀ XOR x₁ (balanced: 4 inputs give 0, 4 give 1)
    """
    qc = QuantumCircuit(4, 3, name="DJ balanced-parity (3q)")
    qc.x(3)
    qc.h([0, 1, 2, 3])
    # Oracle: XOR of qubits 0 and 1
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.h([0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def dj_circuit_balanced_linear_3q():
    """DJ for balanced linear function
    f(x) = x₀ (depends only on first qubit, balanced: 4 zeros, 4 ones)
    """
    qc = QuantumCircuit(4, 3, name="DJ balanced-linear (3q)")
    qc.x(3)
    qc.h([0, 1, 2, 3])
    # Oracle: CNOT from qubit 0
    qc.cx(0, 3)
    qc.h([0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


dj_circuits_3q = [
    (dj_circuit_constant_0_3q(), "constant-0", "CONSTANT"),
    (dj_circuit_constant_1_3q(), "constant-1", "CONSTANT"),
    (dj_circuit_balanced_linear_3q(), "balanced-linear: f=x₀", "BALANCED"),
    (dj_circuit_balanced_odd_parity_3q(), "balanced-parity: f=x₀⊕x₁", "BALANCED")
]

print("Testing 3-qubit functions:\n")

dj_3q_results = []

for circuit, func_name, func_type in dj_circuits_3q:
    counts = run_circuit_and_get_counts(circuit, backend, shots=1000)

    # Check if all measurements are 000 (constant) or has non-zero (balanced)
    all_zero_count = counts.get('000', 0)
    other_counts = sum(v for k, v in counts.items() if k != '000')

    is_constant = other_counts == 0
    predicted_type = "CONSTANT" if is_constant else "BALANCED"

    print(f"f = {func_name:25} | Actual: {func_type:8} | Predicted: {predicted_type:8}")
    print(f"    Result distribution: {counts}")

    dj_3q_results.append({
        'function': func_name,
        'actual_type': func_type,
        'predicted_type': predicted_type,
        'counts': counts,
        'is_constant': is_constant
    })

print()


# ============================================================================
# EXPERIMENT 5: Classical Complexity vs Quantum
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: Classical vs Quantum Complexity Comparison")
print("=" * 70)

print("\nFunction evaluations needed to determine constant vs balanced:\n")
print(f"{'Qubits':>8} | {'Inputs':>10} | {'Classical':>15} | {'Quantum':>10} | {'Speedup':>10}")
print("-" * 60)

for n_qubits in range(1, 11):
    n_inputs = 2 ** n_qubits
    classical_evals = n_inputs // 2 + 1  # Worst case
    quantum_evals = 1
    speedup = classical_evals / quantum_evals

    print(f"{n_qubits:>8} | {n_inputs:>10} | {classical_evals:>15} | {quantum_evals:>10} | "
          f"{speedup:>9.0f}x")

print("\nFor n=100: Classical needs 2^99 ≈ 6.3×10²⁹ evaluations")
print("           Quantum needs just 1 evaluation!")
print()


# ============================================================================
# EXPERIMENT 6: Understanding the Oracle
# ============================================================================
print("=" * 70)
print("EXPERIMENT 6: Oracle Construction and Verification")
print("=" * 70)

"""
The oracle U_f encodes the function f by applying controlled operations.
Different functions require different oracle implementations.

For a function f: {0,1}^n → {0,1}:
- Oracle phase-flips the ancilla when f(x) = 1
- This creates a phase-encoded version of the function
- The Hadamard interference then reveals if f is balanced
"""

print("\nOracle construction patterns:\n")

print("Constant-0 oracle:")
print("  Do nothing (identity operation)")
print("  → All paths have same phase")

print("\nConstant-1 oracle:")
print("  Apply Z gate to ancilla (global phase flip)")
print("  → All paths get inverted")

print("\nBalanced oracle (example: f = x₀):")
print("  Apply CNOT(q₀, ancilla)")
print("  → Paths where x₀=1 get phase-flipped")
print("  → 50% of paths get flipped, 50% don't")

print("\nLinear balanced oracle (example: f = x₀ ⊕ x₁):")
print("  Apply CNOT(q₀, ancilla), then CNOT(q₁, ancilla)")
print("  → Paths where parity is odd get flipped")
print("  → 50% get flipped, 50% don't")
print()


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: 1-Qubit Deutsch Results
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("1-Qubit Deutsch Algorithm: Function Classification", fontsize=14, fontweight='bold')

for idx, (result, ax) in enumerate(zip(deutsch_1q_results, axes.flat)):
    func_name = result['function']
    actual_type = result['actual_type']
    counts = result['counts']

    states = ['0', '1']
    values = [counts.get(state, 0) for state in states]

    color_0 = '#2ca02c' if actual_type == "CONSTANT" else '#1f77b4'
    color_1 = '#ff7f0e' if actual_type == "BALANCED" else '#1f77b4'
    colors = [color_0, color_1]

    ax.bar(states, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_xlabel('Measurement Result', fontsize=11)
    ax.set_title(f'f = {func_name}\nActual: {actual_type}', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1000])

    # Add interpretation text
    result_text = "CONSTANT (all 0s)" if values[1] == 0 else "BALANCED (has 1s)"
    ax.text(0.5, 0.95, result_text, transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

plt.tight_layout()
plt.savefig('deutsch_1qubit_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: deutsch_1qubit_results.png")


# Figure 2: 2-Qubit DJ Results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("2-Qubit Deutsch-Jozsa: Function Classification", fontsize=14, fontweight='bold')

for idx, (result, ax) in enumerate(zip(dj_2q_results, axes.flat)):
    func_name = result['function']
    actual_type = result['actual_type']
    counts = result['counts']

    states = sorted(counts.keys())
    values = [counts.get(state, 0) for state in states]

    # Color: constant functions have all weight on 00
    color = '#2ca02c' if actual_type == "CONSTANT" else '#1f77b4'
    colors = [color for _ in states]
    colors[0] = '#2ca02c' if actual_type == "CONSTANT" else '#ff7f0e'

    ax.bar(range(len(states)), values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=10)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title(f'f = {func_name}\nActual: {actual_type}', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1000])

    # Add interpretation
    is_all_zero = counts.get('00', 0) == 1000
    result_text = "CONSTANT" if is_all_zero else "BALANCED"
    ax.text(0.5, 0.95, result_text, transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

plt.tight_layout()
plt.savefig('deutsch_2qubit_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: deutsch_2qubit_results.png")


# Figure 3: 3-Qubit DJ Results
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("3-Qubit Deutsch-Jozsa: Function Classification", fontsize=14, fontweight='bold')

for idx, (result, ax) in enumerate(zip(dj_3q_results, axes.flat)):
    func_name = result['function']
    actual_type = result['actual_type']
    counts = result['counts']

    states = [format(i, '03b') for i in range(8)]
    values = [counts.get(state, 0) for state in states]

    # Color: constant has all at 000
    is_constant = counts.get('000', 0) > 900
    color = '#2ca02c' if is_constant else '#1f77b4'
    colors = [color for _ in states]
    colors[0] = '#2ca02c' if is_constant else '#ff7f0e'

    ax.bar(range(len(states)), values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=9)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title(f'f = {func_name}\nActual: {actual_type}', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1000])

    result_text = "CONSTANT" if is_constant else "BALANCED"
    ax.text(0.5, 0.95, result_text, transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

plt.tight_layout()
plt.savefig('deutsch_3qubit_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: deutsch_3qubit_results.png")


# Figure 4: Complexity Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Deutsch-Jozsa: Classical vs Quantum Complexity", fontsize=14, fontweight='bold')

# Panel 1: Linear scale
qubits = np.arange(1, 16)
classical_linear = 2 ** qubits / 2 + 1
quantum_linear = np.ones_like(qubits)

ax1.semilogy(qubits, classical_linear, 'o-', linewidth=2.5, markersize=8, color='#ff7f0e', label='Classical')
ax1.semilogy(qubits, quantum_linear, 's-', linewidth=2.5, markersize=8, color='#2ca02c', label='Quantum')
ax1.set_xlabel('Number of Qubits', fontsize=12)
ax1.set_ylabel('Function Evaluations (log scale)', fontsize=12)
ax1.set_title('Evaluations Needed')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xlim([0.5, 15.5])

# Panel 2: Speedup
speedup = classical_linear / quantum_linear
ax2.semilogy(qubits, speedup, 'o-', linewidth=2.5, markersize=8, color='#1f77b4')
ax2.set_xlabel('Number of Qubits', fontsize=12)
ax2.set_ylabel('Speedup Factor (log scale)', fontsize=12)
ax2.set_title('Quantum Speedup over Classical')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0.5, 15.5])

# Add annotations
ax2.text(10, 1e50, 'Exponential speedup!', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('deutsch_complexity.png', dpi=150, bbox_inches='tight')
print("✓ Saved: deutsch_complexity.png")


# Figure 5: Algorithm Flow
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Deutsch-Jozsa Algorithm: Step-by-Step", fontsize=14, fontweight='bold')

# Step 1: Initialization
ax = axes[0, 0]
ax.text(0.5, 0.9, "Step 1: Initialization", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.7, "• n input qubits: |0⟩⊗ⁿ", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.5, "• 1 ancilla qubit: |1⟩", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.3, "State: |0...0⟩|1⟩", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.axis('off')

# Step 2: Hadamard
ax = axes[0, 1]
ax.text(0.5, 0.9, "Step 2: Hadamard Gates", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.7, "• Apply H to all qubits", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.5, "• Create superposition", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.3, "State: Σ|x⟩|−⟩", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.axis('off')

# Step 3: Oracle
ax = axes[1, 0]
ax.text(0.5, 0.9, "Step 3: Apply Oracle U_f", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.7, "• Phase-flip ancilla", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.5, "  when f(x)=1", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.3, "State: Σ(-1)^f(x)|x⟩|−⟩", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
ax.axis('off')

# Step 4: Measurement
ax = axes[1, 1]
ax.text(0.5, 0.9, "Step 4: Hadamard + Measure", ha='center', fontsize=12, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.7, "• Apply H to input qubits", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.5, "• Measure all qubits", ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.25, "Constant → all 0s\nBalanced → ≥ one 1", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.axis('off')

plt.tight_layout()
plt.savefig('deutsch_algorithm_flow.png', dpi=150, bbox_inches='tight')
print("✓ Saved: deutsch_algorithm_flow.png")


print("\n" + "=" * 70)
print("DEUTSCH-JOZSA ALGORITHM EXPLORATION COMPLETE!")
print("=" * 70)

print("\nKey Insights:")
print("✓ Determines function property with ONE oracle call")
print("✓ Classical: needs up to 2^(n-1) + 1 calls")
print("✓ Quantum: always needs exactly 1 call")
print("✓ Exponential speedup for large n")
print("✓ Uses quantum interference to reveal global properties")
print("✓ Oracle encodes function via phase manipulation")

print("\nThe Power:")
print("  n=1:    Classical 2 vs Quantum 1 (2x speedup)")
print("  n=10:   Classical 513 vs Quantum 1 (500x speedup)")
print("  n=100:  Classical 2^99 vs Quantum 1 (EXPONENTIAL speedup!)")

print("\nNext Steps:")
print("1. Try custom oracle functions")
print("2. Implement more balanced functions")
print("3. Study why interference reveals constant vs balanced")
print("4. Explore Simon's Algorithm (similar principle, different problem)")
print("5. Build toward Shor's Algorithm for factoring\n")
