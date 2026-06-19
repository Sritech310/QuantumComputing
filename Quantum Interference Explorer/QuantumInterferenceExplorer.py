"""
Quantum Interference Deep Dive
==============================
Explore how quantum interference works by:
1. Understanding constructive and destructive interference
2. Comparing quantum vs classical probability paths
3. Implementing Mach-Zehnder interferometer
4. Demonstrating interference cancellation
5. Visualizing how phases affect outcomes

Key Concepts:
- Quantum amplitudes can be positive or negative (complex numbers)
- Paths with same amplitude add (constructive) or cancel (destructive)
- Classical: P(path1) + P(path2) always increases total probability
- Quantum: Amplitudes interfere before squaring for probabilities
- This is the essence of quantum advantage!
"""

# Core Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Additional imports for visualization
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


def get_statevector(circuit):
    """Get the statevector of a quantum circuit without measurement."""
    return Statevector.from_instruction(circuit)


# Initialize backend
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")


# ============================================================================
# EXPERIMENT 1: Classical vs Quantum Probability
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Classical vs Quantum Probability Paths")
print("=" * 70)

"""
Classical Probability:
- If path 1 gives 50% outcome A, path 2 gives 50% outcome A
- Total = 50% + 50% = 100% outcome A

Quantum Mechanics:
- Amplitudes can cancel (destructive interference)
- Amplitude A from path1 = a₁, from path2 = a₂
- Total amplitude = a₁ + a₂ (not squared!)
- Probability = |a₁ + a₂|² (can be less than |a₁|² + |a₂|²)
"""

print("\nClassical Model:")
print("  Path 1 → A (50%), Path 2 → A (50%)")
print("  Total probability for A: 50% + 50% = 100%")

print("\nQuantum Model:")
print("  Path 1 → A (amplitude = 1/√2), Path 2 → A (amplitude = 1/√2)")
print("  Total amplitude = 1/√2 + 1/√2 = 2/√2 = √2")
print("  Total probability = |√2|² = 2... but normalized: 100%")
print("  (This is CONSTRUCTIVE interference)")

print("\nBut if Path 2 has a PHASE SHIFT:")
print("  Path 1 → A (amplitude = 1/√2)")
print("  Path 2 → A (amplitude = -1/√2)  [negative phase]")
print("  Total amplitude = 1/√2 - 1/√2 = 0")
print("  Total probability = |0|² = 0%")
print("  (This is DESTRUCTIVE interference - paths cancel!)\n")


# ============================================================================
# EXPERIMENT 2: Mach-Zehnder Interferometer
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: Mach-Zehnder Interferometer")
print("=" * 70)

"""
The Mach-Zehnder interferometer is the quantum analog of the optical experiment.

Circuit structure:
1. Beam Splitter 1 (Hadamard): Split the beam into two paths
2. Phase Shifter: Apply different phases to each path (controlled by middle gate)
3. Beam Splitter 2 (Hadamard): Recombine the paths
4. Measurement: See which detector fires

Without phase shifter: Constructive interference → always get |1⟩
With phase shifter (π): Destructive interference → always get |0⟩
With phase shifter (π/2): Partial interference → 50-50 split
"""

# MZI without phase shift (constructive interference)
print("\n--- MZI without phase shift (should get |1⟩) ---")
circuit_mzi_no_phase = QuantumCircuit(1, 1, name="MZI No Phase")
circuit_mzi_no_phase.h(0)  # Beam splitter 1
# No phase shifter
circuit_mzi_no_phase.h(0)  # Beam splitter 2
circuit_mzi_no_phase.measure(0, 0)

counts_mzi_no_phase = run_circuit_and_get_counts(circuit_mzi_no_phase, backend, shots=1000)
print(f"Counts: {counts_mzi_no_phase}")
print("Expected: Mostly |1⟩ (constructive interference)")

# Get statevector to see amplitudes
sv_mzi_no_phase = get_statevector(circuit_mzi_no_phase.remove_final_measurements(inplace=False))
print(f"Statevector: {sv_mzi_no_phase}")


# MZI with π phase shift (destructive interference)
print("\n--- MZI with π phase shift (should get |0⟩) ---")
circuit_mzi_phase_pi = QuantumCircuit(1, 1, name="MZI Phase π")
circuit_mzi_phase_pi.h(0)  # Beam splitter 1
circuit_mzi_phase_pi.p(np.pi, 0)  # Phase shift π (relative phase of -1)
circuit_mzi_phase_pi.h(0)  # Beam splitter 2
circuit_mzi_phase_pi.measure(0, 0)

counts_mzi_phase_pi = run_circuit_and_get_counts(circuit_mzi_phase_pi, backend, shots=1000)
print(f"Counts: {counts_mzi_phase_pi}")
print("Expected: Mostly |0⟩ (destructive interference)")

sv_mzi_phase_pi = get_statevector(circuit_mzi_phase_pi.remove_final_measurements(inplace=False))
print(f"Statevector: {sv_mzi_phase_pi}")


# MZI with π/2 phase shift (partial interference)
print("\n--- MZI with π/2 phase shift (should get 50-50) ---")
circuit_mzi_phase_pi2 = QuantumCircuit(1, 1, name="MZI Phase π/2")
circuit_mzi_phase_pi2.h(0)  # Beam splitter 1
circuit_mzi_phase_pi2.p(np.pi/2, 0)  # Phase shift π/2
circuit_mzi_phase_pi2.h(0)  # Beam splitter 2
circuit_mzi_phase_pi2.measure(0, 0)

counts_mzi_phase_pi2 = run_circuit_and_get_counts(circuit_mzi_phase_pi2, backend, shots=1000)
print(f"Counts: {counts_mzi_phase_pi2}")
print("Expected: ~50% |0⟩ and ~50% |1⟩ (partial interference)")

sv_mzi_phase_pi2 = get_statevector(circuit_mzi_phase_pi2.remove_final_measurements(inplace=False))
print(f"Statevector: {sv_mzi_phase_pi2}\n")


# ============================================================================
# EXPERIMENT 3: Sweep Phase to Show Interference Pattern
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: Phase Sweep - Interference Pattern")
print("=" * 70)

"""
By sweeping the phase from 0 to 2π, we can visualize the interference pattern.
This is similar to the interference fringes in Young's double slit experiment.
"""

phases = np.linspace(0, 2*np.pi, 20)
prob_0_list = []
prob_1_list = []

print("Phase sweep (0 to 2π)...")

for phase in phases:
    circuit = QuantumCircuit(1, 1, name=f"MZI Phase {phase}")
    circuit.h(0)
    circuit.p(phase, 0)
    circuit.h(0)
    circuit.measure(0, 0)

    counts = run_circuit_and_get_counts(circuit, backend, shots=1000)
    prob_0 = counts.get('0', 0) / 1000
    prob_1 = counts.get('1', 0) / 1000

    prob_0_list.append(prob_0)
    prob_1_list.append(prob_1)

print("Phase sweep complete\n")


# ============================================================================
# EXPERIMENT 4: Two-Qubit Interference (CNOT-based)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: Two-Qubit Interference (Quantum Entanglement Effect)")
print("=" * 70)

"""
When multiple qubits interfere, the effects are more complex.
This circuit demonstrates how interference can create specific output patterns
when both qubits interact.
"""

# Circuit without middle Hadamard (direct path)
print("\n--- Two-qubit path WITHOUT Hadamard in middle ---")
circuit_2q_no_hadamard = QuantumCircuit(2, 2, name="2Q No Hadamard")
circuit_2q_no_hadamard.h(0)  # Create superposition on qubit 0
circuit_2q_no_hadamard.cx(0, 1)  # CNOT creates entanglement
# No middle Hadamard
circuit_2q_no_hadamard.measure([0, 1], [0, 1])

counts_2q_no_h = run_circuit_and_get_counts(circuit_2q_no_hadamard, backend, shots=1000)
print(f"Counts: {counts_2q_no_h}")
print("States created: |00⟩ and |11⟩ (Bell state)")


# Circuit with middle Hadamard (interference path)
print("\n--- Two-qubit path WITH Hadamard in middle (interference) ---")
circuit_2q_with_hadamard = QuantumCircuit(2, 2, name="2Q With Hadamard")
circuit_2q_with_hadamard.h(0)  # Create superposition
circuit_2q_with_hadamard.cx(0, 1)  # CNOT
circuit_2q_with_hadamard.h([0, 1])  # Hadamard on both qubits (interference basis)
circuit_2q_with_hadamard.measure([0, 1], [0, 1])

counts_2q_with_h = run_circuit_and_get_counts(circuit_2q_with_hadamard, backend, shots=1000)
print(f"Counts: {counts_2q_with_h}")
print("Notice: Different interference pattern due to basis change")


# ============================================================================
# EXPERIMENT 5: Destructive Interference Pattern
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: Designing for Destructive Interference")
print("=" * 70)

"""
We can design circuits to cancel out unwanted states.
This is the basis of quantum algorithms like Grover's search!
"""

# Create a circuit that destructively interferes |0⟩ and constructively interferes |1⟩
print("\n--- Designed interference to favor |1⟩ ---")
circuit_designed = QuantumCircuit(1, 1, name="Designed Interference")
circuit_designed.h(0)  # Superposition: (|0⟩ + |1⟩)/√2
circuit_designed.z(0)  # Z gate: marks |1⟩ with phase -1
circuit_designed.h(0)  # Hadamard again to interfere
circuit_designed.measure(0, 0)

counts_designed = run_circuit_and_get_counts(circuit_designed, backend, shots=1000)
print(f"Counts: {counts_designed}")
print("Expected: Mostly |1⟩ due to constructive interference")

sv_designed = get_statevector(circuit_designed.remove_final_measurements(inplace=False))
print(f"Statevector: {sv_designed}\n")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: Mach-Zehnder Interferometer Results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Mach-Zehnder Interferometer: Effect of Phase Shift", fontsize=14, fontweight='bold')

mzi_results = [
    (counts_mzi_no_phase, "No Phase Shift\n(Constructive)"),
    (counts_mzi_phase_pi2, "π/2 Phase Shift\n(Partial)"),
    (counts_mzi_phase_pi, "π Phase Shift\n(Destructive)")
]

for idx, (counts, title) in enumerate(mzi_results):
    ax = axes[idx]
    states = ['0', '1']
    values = [counts.get(state, 0) for state in states]

    colors = ['#ff7f0e' if v > 500 else '#1f77b4' for v in values]
    ax.bar(states, values, color=colors)
    ax.set_ylabel('Counts')
    ax.set_title(title)
    ax.set_ylim([0, 1000])

plt.tight_layout()
plt.savefig('interference_mzi.png', dpi=150, bbox_inches='tight')
print("✓ Saved: interference_mzi.png")


# Figure 2: Phase Sweep Interference Pattern
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(phases, prob_0_list, 'o-', label='P(|0⟩)', linewidth=2, markersize=8, color='#1f77b4')
ax.plot(phases, prob_1_list, 's-', label='P(|1⟩)', linewidth=2, markersize=8, color='#ff7f0e')
ax.set_xlabel('Phase (radians)', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Quantum Interference Pattern: Probability vs Phase', fontsize=14, fontweight='bold')
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('interference_phase_sweep.png', dpi=150, bbox_inches='tight')
print("✓ Saved: interference_phase_sweep.png")


# Figure 3: Two-Qubit Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Two-Qubit Interference: With and Without Middle Hadamard", fontsize=14, fontweight='bold')

plot_data = [
    (counts_2q_no_h, "Without Hadamard\n(Bell State)", axes[0]),
    (counts_2q_with_h, "With Hadamard\n(Interference)", axes[1])
]

for counts, title, ax in plot_data:
    states = sorted(counts.keys())
    values = [counts.get(state, 0) for state in states]
    ax.bar(range(len(states)), values, color='#1f77b4', alpha=0.7)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_ylabel('Counts')
    ax.set_title(title)
    ax.set_ylim([0, 1000])

plt.tight_layout()
plt.savefig('interference_two_qubit.png', dpi=150, bbox_inches='tight')
print("✓ Saved: interference_two_qubit.png")


# Figure 4: Amplitude visualization (theoretical)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Quantum Amplitude Interference (Theoretical)", fontsize=14, fontweight='bold')

# No phase shift
ax = axes[0, 0]
amplitudes = [0.5, 0.5]  # Equal amplitudes
interference = 2 * 0.5  # Constructive: 0.5 + 0.5
ax.bar(['Path 1', 'Path 2', 'Combined'], [0.5, 0.5, interference], color=['#1f77b4', '#1f77b4', '#2ca02c'])
ax.set_ylabel('Amplitude')
ax.set_title('No Phase: Constructive Interference')
ax.set_ylim([0, 1.2])
ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Max possible')

# π phase shift
ax = axes[0, 1]
amplitudes = [0.5, -0.5]  # Opposite amplitudes
ax.bar(['Path 1', 'Path 2', 'Combined'], [0.5, -0.5, 0], color=['#1f77b4', '#ff7f0e', '#d62728'])
ax.set_ylabel('Amplitude')
ax.set_title('π Phase: Destructive Interference')
ax.set_ylim([-1, 1])
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Probability comparison (classical vs quantum)
ax = axes[1, 0]
x = np.array([0, 1])
classical = np.array([0.5, 0.5])  # Two paths each 50%
quantum_construct = np.array([1.0, 0])  # Constructive interference
quantum_destruct = np.array([0, 1.0])  # Destructive interference
width = 0.25

ax.bar(x - width, classical, width, label='Classical (no interference)', color='gray', alpha=0.7)
ax.bar(x, quantum_construct, width, label='Quantum (constructive)', color='#2ca02c')
ax.bar(x + width, quantum_destruct, width, label='Quantum (destructive)', color='#d62728')
ax.set_ylabel('Probability')
ax.set_xlabel('Outcome')
ax.set_title('Classical vs Quantum Probabilities')
ax.set_xticks(x)
ax.set_xticklabels(['|0⟩', '|1⟩'])
ax.legend()
ax.set_ylim([0, 1.2])

# Interference pattern visualization
ax = axes[1, 1]
theta = np.linspace(0, 4*np.pi, 1000)
interference_signal = np.cos(theta)
ax.fill_between(theta, 0, interference_signal**2, alpha=0.5, color='#1f77b4', label='P(outcome)')
ax.plot(theta, interference_signal, '--', color='#ff7f0e', linewidth=2, label='Amplitude pattern')
ax.set_xlabel('Phase variation')
ax.set_ylabel('Probability / Amplitude')
ax.set_title('Typical Quantum Interference Pattern')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('interference_theory.png', dpi=150, bbox_inches='tight')
print("✓ Saved: interference_theory.png")

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE!")
print("=" * 70)
print("\nKey Insights:")
print("✓ Quantum amplitudes interfere BEFORE measurement")
print("✓ Phase shifts control constructive vs destructive interference")
print("✓ Same superposition can produce different results with different interference")
print("✓ This is fundamentally different from classical probability!")
print("\nNext Steps:")
print("1. Modify phases in the Mach-Zehnder circuit")
print("2. Try adding more gates between beam splitters")
print("3. Explore 3-qubit interference patterns")
print("4. Build a circuit that amplifies desired states (Grover's preparation)\n")
