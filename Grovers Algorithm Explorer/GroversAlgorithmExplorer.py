"""
Grover's Algorithm - Quantum Search
====================================
Explore Grover's Algorithm - the first quantum algorithm showing speedup!

What is Grover's Algorithm?
- Finds a marked item in an unsorted database quadratically faster
- Classical: N/2 attempts on average (for N items)
- Quantum: √N attempts on average
- For N=1,000,000: Classical ~500k tries, Quantum ~1000 tries!

Key Components:
1. Initialization: Superposition of all N states
2. Oracle: Marks the solution by flipping its phase
3. Diffusion Operator: Amplifies marked state via interference
4. Iterate: Repeat oracle + diffusion ~√N times
5. Measure: Collapse to the marked state

The Magic:
- Uses constructive/destructive INTERFERENCE to amplify solution amplitude
- Classical path: 1/N probability of success
- Quantum path: ~1 probability of success (after √N iterations)
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


def get_amplitudes(circuit):
    """Get the amplitudes of all basis states."""
    sv = get_statevector(circuit)
    return sv.data


# Initialize backend
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")


# ============================================================================
# EXPERIMENT 1: Classical vs Quantum - The Problem
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Classical vs Quantum Search")
print("=" * 70)

print("\nThe Problem: Find a marked item in an unsorted list")
print("  Database size: N items")
print("  Classical approach: Random guessing")
print("    - Average attempts: N/2 (try roughly half the list)")
print("    - Worst case: N (try everything)")

print("\n  Quantum approach: Grover's Algorithm")
print("    - Attempts needed: ~√N (quadratic speedup!)")
print("    - For N=1,000,000: Classical ~500k, Quantum ~1000")
print("    - Time complexity: O(√N) vs O(N)\n")


# ============================================================================
# EXPERIMENT 2: 2-Qubit Grover (Simple)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: 2-Qubit Grover's Algorithm")
print("=" * 70)

"""
With 2 qubits, we have 4 possible states: |00⟩, |01⟩, |10⟩, |11⟩
We want to find a specific state (let's say |11⟩).

Steps:
1. H-gates: Create superposition (equal amplitude 1/2 for each state)
2. Oracle: Flip phase of target (|11⟩)
3. Diffusion: Flip amplitudes about the average (inversion about average)
4. Measure: Likely to measure |11⟩
"""

print("\nSearching for target state: |11⟩\n")

# Step 1: Initialization - superposition
print("Step 1: Create superposition (1/2 for each of 4 states)")
circuit_init = QuantumCircuit(2, name="Initialize")
circuit_init.h([0, 1])
amps = get_amplitudes(circuit_init)
print(f"Amplitudes after H-gates: {np.round(amps, 3)}")
print(f"Probabilities: {np.round(np.abs(amps)**2, 3)}\n")


# Step 2: Define Oracle for |11⟩
def oracle_2qubit_11(circuit, q0, q1):
    """Oracle that marks |11⟩ with a phase flip"""
    circuit.cz(q0, q1)  # CZ gate flips phase when both qubits are |1⟩


# Step 3: Define Diffusion Operator (Inversion about Average)
def diffusion_operator_2qubit(circuit, q0, q1):
    """
    Diffusion operator for 2-qubit Grover.
    Does: D = 2|s⟩⟨s| - I, where |s⟩ is the superposition state
    Achieves: Inversion about the average amplitude
    """
    circuit.h([q0, q1])  # H gates
    circuit.x([q0, q1])  # X gates (flip bits)
    circuit.cz(q0, q1)   # CZ gate (phase flip when both 1)
    circuit.x([q0, q1])  # X gates again
    circuit.h([q0, q1])  # H gates again


# Build Grover circuit for 2 qubits without iteration
print("Step 2-3: Apply Oracle and Diffusion")
circuit_grover_2q = QuantumCircuit(2, 2, name="Grover 2Q (1 iteration)")
circuit_grover_2q.h([0, 1])

print("Amplitudes before oracle:")
sv_before = get_statevector(circuit_grover_2q)
print(f"  {np.round(sv_before.data, 3)}")

oracle_2qubit_11(circuit_grover_2q, 0, 1)
print("Amplitudes after oracle (|11⟩ gets -1):")
sv_after_oracle = get_statevector(circuit_grover_2q)
print(f"  {np.round(sv_after_oracle.data, 3)}")

diffusion_operator_2qubit(circuit_grover_2q, 0, 1)
print("Amplitudes after diffusion:")
sv_after_diffusion = get_statevector(circuit_grover_2q)
print(f"  {np.round(sv_after_diffusion.data, 3)}")

circuit_grover_2q.measure([0, 1], [0, 1])
counts_grover_2q = run_circuit_and_get_counts(circuit_grover_2q, backend, shots=1000)
print(f"\nMeasurement results: {counts_grover_2q}")
print(f"Success! Probability of finding |11⟩: {counts_grover_2q.get('11', 0)/10:.1f}%\n")


# ============================================================================
# EXPERIMENT 3: 2-Qubit Grover with Multiple Iterations
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: Effect of Multiple Iterations")
print("=" * 70)

"""
One iteration isn't always optimal. For 2 qubits with 1 target,
we might need 0 or 1 iteration depending on the algorithm phase.
Let's see how probability changes with each iteration.
"""

print("\nIterations 0-3 (showing amplitude of |11⟩):\n")

iteration_results = []

for iteration in range(4):
    circuit = QuantumCircuit(2, 2, name=f"Grover 2Q ({iteration} iter)")
    circuit.h([0, 1])

    # Apply Oracle + Diffusion 'iteration' times
    for i in range(iteration):
        oracle_2qubit_11(circuit, 0, 1)
        diffusion_operator_2qubit(circuit, 0, 1)

    # Get amplitudes before measurement
    sv = get_statevector(circuit)
    target_amp = sv.data[3]  # |11⟩ is the 4th state (binary 11)
    target_prob = np.abs(target_amp) ** 2

    circuit.measure([0, 1], [0, 1])
    counts = run_circuit_and_get_counts(circuit, backend, shots=1000)
    measured_prob = counts.get('11', 0) / 1000

    print(f"Iteration {iteration}:")
    print(f"  Amplitude of |11⟩: {target_amp:.4f}")
    print(f"  Theoretical probability: {target_prob:.4f}")
    print(f"  Measured probability: {measured_prob:.4f}")

    iteration_results.append({
        'iteration': iteration,
        'amplitude': np.abs(target_amp),
        'theory_prob': target_prob,
        'measured_prob': measured_prob,
        'counts': counts
    })

print()


# ============================================================================
# EXPERIMENT 4: 3-Qubit Grover (8 states, find 1)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: 3-Qubit Grover's Algorithm")
print("=" * 70)

"""
With 3 qubits, we have 8 possible states: |000⟩ to |111⟩
We want to find |101⟩ (5 in decimal).

For 3 qubits with 1 target:
- Number of iterations needed: round(π/4 * √8) ≈ round(π/4 * 2.83) ≈ 2
"""

print("\nSearching for target state: |101⟩ (5 in decimal)")
print(f"Number of qubits: 3")
print(f"Total states: 8")
print(f"Target iterations: round(π/4 * √8) ≈ 2\n")


# Oracle for |101⟩
def oracle_3qubit_101(circuit, q0, q1, q2):
    """Oracle that marks |101⟩ with a phase flip"""
    # |101⟩ means q0=1, q1=0, q2=1
    # X on q1 to convert to all-ones, then MCZ, then X again
    circuit.x(q1)  # Flip q1 temporarily
    circuit.ccz(q0, q1, q2)  # Multi-controlled Z
    circuit.x(q1)  # Flip q1 back


# Diffusion for 3 qubits
def diffusion_operator_3qubit(circuit, q0, q1, q2):
    """Diffusion operator for 3-qubit Grover"""
    circuit.h([q0, q1, q2])
    circuit.x([q0, q1, q2])
    circuit.ccz(q0, q1, q2)
    circuit.x([q0, q1, q2])
    circuit.h([q0, q1, q2])


# Build 3-qubit Grover with optimal iterations
optimal_iterations_3q = 2
circuit_grover_3q = QuantumCircuit(3, 3, name="Grover 3Q")
circuit_grover_3q.h([0, 1, 2])

for i in range(optimal_iterations_3q):
    oracle_3qubit_101(circuit_grover_3q, 0, 1, 2)
    diffusion_operator_3qubit(circuit_grover_3q, 0, 1, 2)

circuit_grover_3q.measure([0, 1, 2], [0, 1, 2])
counts_grover_3q = run_circuit_and_get_counts(circuit_grover_3q, backend, shots=1000)

print(f"Results after {optimal_iterations_3q} iterations:")
print(f"Counts: {counts_grover_3q}")

prob_101 = counts_grover_3q.get('101', 0) / 1000
print(f"Probability of measuring |101⟩: {prob_101:.1%}")
print(f"Success!\n")


# ============================================================================
# EXPERIMENT 5: Iteration Sweep (Demonstrating Optimal Iterations)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: Iteration Sweep - Finding Optimal Iterations")
print("=" * 70)

"""
Different numbers of iterations give different success probabilities.
There's an optimal number where the target amplitude is maximized.
Too many iterations can actually decrease the probability (overshooting).
"""

print("\nSweeping iterations for 3-qubit search (target |101⟩):\n")

sweep_results = []

for num_iterations in range(0, 6):
    circuit = QuantumCircuit(3, 3, name=f"Grover 3Q sweep")
    circuit.h([0, 1, 2])

    for i in range(num_iterations):
        oracle_3qubit_101(circuit, 0, 1, 2)
        diffusion_operator_3qubit(circuit, 0, 1, 2)

    # Get theoretical amplitude
    sv = get_statevector(circuit)
    # |101⟩ is the 5th state in order: |000⟩=0, |001⟩=1, |010⟩=2, |011⟩=3, |100⟩=4, |101⟩=5
    target_amp = sv.data[5]
    target_prob = np.abs(target_amp) ** 2

    circuit.measure([0, 1, 2], [0, 1, 2])
    counts = run_circuit_and_get_counts(circuit, backend, shots=1000)
    measured_prob = counts.get('101', 0) / 1000

    print(f"Iterations: {num_iterations} | Theory: {target_prob:.4f} | Measured: {measured_prob:.4f}")
    sweep_results.append({
        'iterations': num_iterations,
        'theory_prob': target_prob,
        'measured_prob': measured_prob
    })

print()


# ============================================================================
# EXPERIMENT 6: Multiple Targets (Marked Items)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 6: Grover with Multiple Targets")
print("=" * 70)

"""
Grover's algorithm can find multiple marked items!
The oracle marks all targets, and diffusion amplifies all of them.
With M targets out of N items:
- Iterations needed: ~√(N/M)
- If half the items are marked: √(N/N/2) = √2 ≈ 1.4 iterations
"""

print("\nSearching for two target states: |00⟩ and |11⟩")
print("Oracle will mark both targets\n")


def oracle_2qubit_00_or_11(circuit, q0, q1):
    """Oracle that marks both |00⟩ and |11⟩"""
    # |11⟩ marked by CZ
    circuit.cz(q0, q1)
    # |00⟩ marked by X-CZ-X pattern
    circuit.x([q0, q1])
    circuit.cz(q0, q1)
    circuit.x([q0, q1])


circuit_multi_target = QuantumCircuit(2, 2, name="Grover 2Q Multi-Target")
circuit_multi_target.h([0, 1])

oracle_2qubit_00_or_11(circuit_multi_target, 0, 1)
diffusion_operator_2qubit(circuit_multi_target, 0, 1)

circuit_multi_target.measure([0, 1], [0, 1])
counts_multi = run_circuit_and_get_counts(circuit_multi_target, backend, shots=1000)

print(f"Results:")
print(f"Counts: {counts_multi}")
print(f"Probability of |00⟩: {counts_multi.get('00', 0)/10:.1f}%")
print(f"Probability of |11⟩: {counts_multi.get('11', 0)/10:.1f}%")
print(f"Both targets amplified!\n")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: 2-Qubit Iteration Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("2-Qubit Grover: Amplitude Amplification Over Iterations", fontsize=14, fontweight='bold')

iterations = [r['iteration'] for r in iteration_results]
amplitudes = [r['amplitude'] for r in iteration_results]
measured_probs = [r['measured_prob'] for r in iteration_results]
theory_probs = [r['theory_prob'] for r in iteration_results]

# Plot 1: Amplitude growth
ax1.plot(iterations, amplitudes, 'o-', linewidth=2, markersize=10, color='#1f77b4', label='|11⟩ Amplitude')
ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Maximum (1.0)')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Amplitude Magnitude', fontsize=12)
ax1.set_title('Amplitude Growth')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0, 1.1])

# Plot 2: Probability comparison
ax2.plot(iterations, theory_probs, 'o-', linewidth=2, markersize=10, color='#1f77b4', label='Theoretical')
ax2.plot(iterations, measured_probs, 's-', linewidth=2, markersize=10, color='#ff7f0e', label='Measured (1000 shots)')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Probability of |11⟩', fontsize=12)
ax2.set_title('Success Probability')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('grovers_2qubit_iterations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: grovers_2qubit_iterations.png")


# Figure 2: 2-Qubit Search Results
fig, ax = plt.subplots(figsize=(10, 6))

states = ['00', '01', '10', '11']
counts = counts_grover_2q
values = [counts.get(state, 0) for state in states]

colors = ['#1f77b4' if v < 100 else '#2ca02c' if state == '11' else '#1f77b4' for state, v in zip(states, values)]
bars = ax.bar(states, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

ax.set_ylabel('Counts', fontsize=12)
ax.set_xlabel('Basis State', fontsize=12)
ax.set_title('2-Qubit Grover Search (Target: |11⟩)', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1000])

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('grovers_2qubit_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: grovers_2qubit_results.png")


# Figure 3: 3-Qubit Iteration Sweep
fig, ax = plt.subplots(figsize=(12, 6))

iterations_sweep = [r['iterations'] for r in sweep_results]
theory_sweep = [r['theory_prob'] for r in sweep_results]
measured_sweep = [r['measured_prob'] for r in sweep_results]

ax.plot(iterations_sweep, theory_sweep, 'o-', linewidth=2.5, markersize=10,
        color='#1f77b4', label='Theoretical Probability', zorder=3)
ax.plot(iterations_sweep, measured_sweep, 's--', linewidth=2.5, markersize=10,
        color='#ff7f0e', label='Measured Probability (1000 shots)', zorder=3)

# Mark optimal iteration
optimal_idx = np.argmax(theory_sweep)
ax.axvline(x=iterations_sweep[optimal_idx], color='green', linestyle=':', linewidth=2, alpha=0.5, label='Optimal Iterations')

ax.set_xlabel('Number of Iterations', fontsize=12)
ax.set_ylabel('Probability of Finding |101⟩', fontsize=12)
ax.set_title("3-Qubit Grover: Optimal Iteration Count", fontsize=14, fontweight='bold')
ax.set_xticks(iterations_sweep)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('grovers_3qubit_sweep.png', dpi=150, bbox_inches='tight')
print("✓ Saved: grovers_3qubit_sweep.png")


# Figure 4: 3-Qubit Results Histogram
fig, ax = plt.subplots(figsize=(12, 6))

states_3q = [format(i, '03b') for i in range(8)]
counts_3q = counts_grover_3q
values_3q = [counts_3q.get(state, 0) for state in states_3q]

colors_3q = ['#2ca02c' if state == '101' else '#1f77b4' for state in states_3q]
bars_3q = ax.bar(range(len(states_3q)), values_3q, color=colors_3q, alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xticks(range(len(states_3q)))
ax.set_xticklabels(states_3q, fontsize=11)
ax.set_ylabel('Counts', fontsize=12)
ax.set_xlabel('Basis State', fontsize=12)
ax.set_title("3-Qubit Grover Search (Target: |101⟩, 2 iterations)", fontsize=14, fontweight='bold')
ax.set_ylim([0, 1000])

for bar, val in zip(bars_3q, values_3q):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('grovers_3qubit_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: grovers_3qubit_results.png")


# Figure 5: Conceptual Overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Grover's Algorithm: The Complete Picture", fontsize=14, fontweight='bold')

# Panel 1: Initial superposition
ax = axes[0, 0]
states_init = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
amps_init = [0.5] * 4
colors_init = ['#1f77b4'] * 4
ax.bar(range(4), amps_init, color=colors_init, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(4))
ax.set_xticklabels(states_init)
ax.set_ylabel('Amplitude')
ax.set_title('Step 1: Equal Superposition')
ax.set_ylim([0, 1])

# Panel 2: After oracle
ax = axes[0, 1]
amps_oracle = [0.5, 0.5, 0.5, -0.5]
colors_oracle = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e']
ax.bar(range(4), amps_oracle, color=colors_oracle, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(4))
ax.set_xticklabels(states_init)
ax.set_ylabel('Amplitude')
ax.set_title('Step 2: Oracle (|11⟩ phase flipped)')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_ylim([-1, 1])

# Panel 3: After diffusion
ax = axes[1, 0]
amps_diffusion = [-0.25, -0.25, -0.25, 0.75]  # Inverted about average
colors_diffusion = ['#1f77b4', '#1f77b4', '#1f77b4', '#2ca02c']
ax.bar(range(4), amps_diffusion, color=colors_diffusion, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(4))
ax.set_xticklabels(states_init)
ax.set_ylabel('Amplitude')
ax.set_title('Step 3: Diffusion (Amplification)')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_ylim([-1, 1])

# Panel 4: Probability comparison
ax = axes[1, 1]
x = np.array([0, 1, 2])
classical_probs = np.array([0.25, 0.25, 0.25])
quantum_probs = np.array([0.0625, 0.0625, 0.5625])
width = 0.35

ax.bar(x - width/2, classical_probs, width, label='Classical', color='gray', alpha=0.7)
ax.bar(x + width/2, quantum_probs, width, label='After Grover', color='#2ca02c', alpha=0.7)
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(['Other states', 'Other states', 'Target |11⟩'])
ax.set_title('Probability: Classical vs Quantum')
ax.legend()
ax.set_ylim([0, 0.7])

plt.tight_layout()
plt.savefig('grovers_conceptual.png', dpi=150, bbox_inches='tight')
print("✓ Saved: grovers_conceptual.png")


print("\n" + "=" * 70)
print("GROVER'S ALGORITHM EXPLORATION COMPLETE!")
print("=" * 70)

print("\nKey Insights:")
print("✓ Amplitude amplification is the key mechanism")
print("✓ Oracle marks solutions, diffusion amplifies marked states")
print("✓ Optimal iterations: ~π/4 * √N")
print("✓ Quadratic speedup: √N vs N")
print("✓ Works for single and multiple targets")

print("\nComparison:")
print("  Classical search (random): 4 tries for 4 items (25% per try)")
print("  Grover's algorithm: 1-2 tries for 4 items (90%+ per try)")
print("  For 1,000,000 items:")
print("    Classical: ~500,000 tries")
print("    Grover: ~1,000 tries  ← 500x speedup!")

print("\nNext Steps:")
print("1. Modify the oracle to search for different target states")
print("2. Try searching with multiple targets")
print("3. Increase qubits (4, 5, 6) and observe scaling")
print("4. Study how oracle design affects performance")
print("5. Combine with phase estimation for more complex algorithms\n")
