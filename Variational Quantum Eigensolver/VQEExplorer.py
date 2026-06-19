"""
Variational Quantum Eigensolver (VQE)
=====================================
Explore VQE - the most practical near-term quantum algorithm!

What is VQE?

The Problem:
Find the ground state energy of a molecular/quantum system described by Hamiltonian H.
  E_ground = min_|ψ⟩ ⟨ψ|H|ψ⟩

Classical Approach:
- Diagonalize H matrix: O(N³) for dense matrices
- For N atoms: Hilbert space grows exponentially
- H₂ molecule: ~10 qubits, manageable
- Water molecule: ~100 qubits, intractable
- Protein: Millions of qubits needed

Quantum Approach (VQE):
- Use quantum computer to prepare trial states |ψ(θ)⟩
- Measure energy: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
- Classical optimizer adjusts θ to minimize E
- Repeat until convergence
- Result: Ground state energy + wavefunction

Why VQE is Practical:
✓ Hybrid algorithm: works on noisy NISQ devices
✓ Doesn't require error correction
✓ Short quantum circuits (1-50 gates typically)
✓ Measurable improvement possible today
✓ Used in industry (Merck, Roche, etc. for drug discovery)

The Algorithm:

1. ANSATZ DESIGN:
   - Create parameterized quantum circuit
   - θ = [θ₁, θ₂, ..., θₙ] are classical parameters
   - Circuit: |ψ(θ)⟩ = U(θ)|0⟩

2. HAMILTONIAN:
   - Express molecular Hamiltonian as sum of Pauli operators
   - H = Σᵢ hᵢ Pᵢ (Pᵢ = tensor product of Pauli matrices)
   - Example H₂: H = a₀I + a₁Z₀ + a₂Z₁ + a₃Z₀Z₁ + ...

3. ENERGY MEASUREMENT:
   - Measure each Pauli term: ⟨Pᵢ⟩
   - Compute total: E(θ) = Σᵢ hᵢ⟨Pᵢ⟩

4. CLASSICAL OPTIMIZATION:
   - Use optimizer (COBYLA, SPSA, etc.)
   - Minimize E(θ)
   - Repeat steps 2-3 with updated θ

5. CONVERGENCE:
   - Stop when E(θ) converges
   - Result: Ground state energy E₀ and parameters θ*

Applications:
- H₂, LiH molecules: Current quantum computers
- Quantum chemistry: Predicting molecular properties
- Drug discovery: Binding energy calculations
- Materials science: Electronic structure
- Physics: Quantum simulations
"""

# Core Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
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


# Initialize backend
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")


# ============================================================================
# EXPERIMENT 1: Understanding Hamiltonian and Ground State
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Hamiltonian, Eigenvalues, and Ground State")
print("=" * 70)

print("\nQuantum Systems are described by Hamiltonians (H)")
print("  H = total energy operator of the system")
print("  Eigenvalues of H = possible energy measurements")
print("  Lowest eigenvalue = ground state energy E₀")
print("  Corresponding eigenvector = ground state |ψ₀⟩")

print("\nFor a quantum computer, we can express H as:")
print("  H = Σᵢ hᵢ Pᵢ")
print("  where Pᵢ are Pauli operators (I, X, Y, Z)")
print("  and hᵢ are coefficients")

print("\nExample: H₂ Molecule (Hydrogen)")
print("  2 electrons, 2 nuclei")
print("  Full Hamiltonian has ~15 Pauli terms")
print("  Ground state energy ≈ -1.17 hartree")

print("\nExample Hamiltonian (toy 2-qubit):")
print("  H = 0.5·I - 0.5·Z₀ - 0.3·Z₁ + 0.2·Z₀Z₁")
print("  Terms:")
print("    0.5·I      (constant energy)")
print("   -0.5·Z₀     (first qubit measurement)")
print("   -0.3·Z₁     (second qubit measurement)")
print("    0.2·Z₀Z₁   (correlation between qubits)")

print("\nClassical diagonalization:")
print("  Eigenvalues: [-1.0, -0.3, -0.3, 0.4]")
print("  Ground state: E₀ = -1.0\n")


# ============================================================================
# EXPERIMENT 2: Simple Pauli Measurement
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: Measuring Pauli Expectation Values")
print("=" * 70)

"""
To compute ⟨ψ|H|ψ⟩, we measure individual Pauli operators.
For Pauli Z: measure in computational basis
For Pauli X: apply H before measurement (basis rotation)
For Pauli Y: apply S†H before measurement
"""

print("\nMeasuring Pauli expectation values on simple states:\n")

# State |0⟩
circuit_0 = QuantumCircuit(1, 1)
# No gates - state is |0⟩
circuit_0.measure(0, 0)
counts_0 = run_circuit_and_get_counts(circuit_0, backend, shots=1000)
z_expect_0 = (counts_0.get('0', 0) - counts_0.get('1', 0)) / 1000
print(f"|0⟩ state:")
print(f"  Measurement counts: {counts_0}")
print(f"  ⟨Z⟩ = (N₀ - N₁)/shots = {z_expect_0:.3f}")
print(f"  (Expected: +1.0)")

# State |1⟩
circuit_1 = QuantumCircuit(1, 1)
circuit_1.x(0)  # Flip to |1⟩
circuit_1.measure(0, 0)
counts_1 = run_circuit_and_get_counts(circuit_1, backend, shots=1000)
z_expect_1 = (counts_1.get('0', 0) - counts_1.get('1', 0)) / 1000
print(f"\n|1⟩ state:")
print(f"  Measurement counts: {counts_1}")
print(f"  ⟨Z⟩ = {z_expect_1:.3f}")
print(f"  (Expected: -1.0)")

# State |+⟩ = (|0⟩ + |1⟩)/√2
circuit_plus = QuantumCircuit(1, 1)
circuit_plus.h(0)  # Create |+⟩
circuit_plus.measure(0, 0)
counts_plus = run_circuit_and_get_counts(circuit_plus, backend, shots=1000)
z_expect_plus = (counts_plus.get('0', 0) - counts_plus.get('1', 0)) / 1000
print(f"\n|+⟩ state (50-50 superposition):")
print(f"  Measurement counts: {counts_plus}")
print(f"  ⟨Z⟩ = {z_expect_plus:.3f}")
print(f"  (Expected: 0.0)")

# State |+⟩, but measure X
circuit_plus_x = QuantumCircuit(1, 1)
circuit_plus_x.h(0)  # Create |+⟩
circuit_plus_x.h(0)  # Rotate back to Z basis then measure
# Actually to measure X: need to rotate basis
circuit_plus_x_proper = QuantumCircuit(1, 1)
circuit_plus_x_proper.h(0)  # Create |+⟩
circuit_plus_x_proper.h(0)  # Basis rotation for X measurement
circuit_plus_x_proper.measure(0, 0)
counts_plus_x = run_circuit_and_get_counts(circuit_plus_x_proper, backend, shots=1000)
x_expect_plus = (counts_plus_x.get('0', 0) - counts_plus_x.get('1', 0)) / 1000
print(f"\n|+⟩ state, measuring X:")
print(f"  ⟨X⟩ = {x_expect_plus:.3f}")
print(f"  (Expected: +1.0, since |+⟩ is eigenstate of X)\n")


# ============================================================================
# EXPERIMENT 3: Energy Measurement for Simple 2-Qubit System
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: Computing Total Energy from Pauli Terms")
print("=" * 70)

"""
Hamiltonian: H = 0.5·I - 0.5·Z₀ - 0.3·Z₁ + 0.2·Z₀Z₁

To compute ⟨ψ|H|ψ⟩, we measure each term separately:
- Measure ⟨I⟩ = 1 (always)
- Measure ⟨Z₀⟩
- Measure ⟨Z₁⟩
- Measure ⟨Z₀Z₁⟩ (measure both qubits, multiply results)
"""

print("\nHamiltonian: H = 0.5·I - 0.5·Z₀ - 0.3·Z₁ + 0.2·Z₀Z₁\n")

# Prepare |0⟩|0⟩ state
circuit_ground = QuantumCircuit(2, 2)
# Already |0⟩|0⟩
circuit_ground.measure([0, 1], [0, 1])
counts_ground = run_circuit_and_get_counts(circuit_ground, backend, shots=1000)

print(f"State |0⟩|0⟩:")
print(f"Measurement counts: {counts_ground}")

# Calculate Pauli expectation values
# ⟨I⟩ = 1
pauli_i = 1.0

# ⟨Z₀⟩: measure qubit 0 (in |0⟩|0⟩ state, should be +1)
n_0 = counts_ground.get('00', 0) + counts_ground.get('01', 0)
n_1 = counts_ground.get('10', 0) + counts_ground.get('11', 0)
pauli_z0 = (n_0 - n_1) / 1000
print(f"⟨Z₀⟩ = {pauli_z0:.3f}")

# ⟨Z₁⟩: measure qubit 1
n_0_q1 = counts_ground.get('00', 0) + counts_ground.get('10', 0)
n_1_q1 = counts_ground.get('01', 0) + counts_ground.get('11', 0)
pauli_z1 = (n_0_q1 - n_1_q1) / 1000
print(f"⟨Z₁⟩ = {pauli_z1:.3f}")

# ⟨Z₀Z₁⟩: measure both, multiply
pauli_z0z1_sum = 0
for bitstring, count in counts_ground.items():
    # Z measurement: +1 if 0, -1 if 1
    z0_val = 1 if bitstring[0] == '0' else -1
    z1_val = 1 if bitstring[1] == '0' else -1
    pauli_z0z1_sum += count * z0_val * z1_val
pauli_z0z1 = pauli_z0z1_sum / 1000
print(f"⟨Z₀Z₁⟩ = {pauli_z0z1:.3f}")

# Total energy
energy = 0.5 * pauli_i - 0.5 * pauli_z0 - 0.3 * pauli_z1 + 0.2 * pauli_z0z1
print(f"\nTotal energy: E = 0.5·(1) - 0.5·({pauli_z0:.3f}) - 0.3·({pauli_z1:.3f}) + 0.2·({pauli_z0z1:.3f})")
print(f"           E = {energy:.4f}")
print(f"\nFor |0⟩|0⟩ state and H = 0.5·I - 0.5·Z₀ - 0.3·Z₁ + 0.2·Z₀Z₁:")
print(f"  Theoretical: 0.5 - 0.5·(1) - 0.3·(1) + 0.2·(1) = -0.1")
print(f"  Measured:    {energy:.4f}\n")


# ============================================================================
# EXPERIMENT 4: VQE with Simple Ansatz (1-Parameter)
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: VQE Optimization - Finding Ground State")
print("=" * 70)

"""
Simple VQE with 1-qubit system and 1 parameter.
Hamiltonian: H = [[1, 0], [0, -1]] with eigenvalues ±1

Ansatz: U(θ) = RY(θ)
State: |ψ(θ)⟩ = RY(θ)|0⟩

Energy: E(θ) = ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ)

Minimum: E = -1 at θ = π
"""

print("\nSimple 1-qubit VQE:")
print("  Hamiltonian: H = Z")
print("  Ansatz: |ψ(θ)⟩ = RY(θ)|0⟩")
print("  Energy: E(θ) = ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ)")
print("  Minimum: E_min = -1 at θ = π\n")

def cost_function_simple(theta):
    """Compute energy for simple 1-qubit system"""
    circuit = QuantumCircuit(1, 1)
    circuit.ry(theta[0], 0)
    circuit.measure(0, 0)

    counts = run_circuit_and_get_counts(circuit, backend, shots=1000)
    # E = ⟨Z⟩
    energy = (counts.get('0', 0) - counts.get('1', 0)) / 1000
    return energy


print("Optimizing with COBYLA optimizer...")
print("(This will take ~30 seconds as it evaluates many θ values)\n")

# Initial point
theta_init = [0.1]

# Optimization
result_simple = minimize(
    cost_function_simple,
    theta_init,
    method='COBYLA',
    options={'maxiter': 20, 'tol': 0.01}
)

print(f"Optimization Results:")
print(f"  Initial θ: {theta_init[0]:.4f}, E: {cost_function_simple(theta_init):.4f}")
print(f"  Final θ: {result_simple.x[0]:.4f}, E: {result_simple.fun:.4f}")
print(f"  Theoretical θ: {np.pi:.4f}, E_theory: -1.0000")
print(f"  Error: {abs(result_simple.fun - (-1.0)):.4f}\n")


# ============================================================================
# EXPERIMENT 5: VQE for 2-Qubit Hamiltonian
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: 2-Qubit VQE (More Realistic)")
print("=" * 70)

"""
2-qubit system with more realistic Hamiltonian.
This demonstrates the full VQE workflow on a slightly larger system.
"""

print("\n2-qubit Hamiltonian:")
print("  H = 0.5·I - 0.5·Z₀ - 0.3·Z₁ + 0.2·Z₀Z₁")

# Hamiltonian coefficients
h_coeffs = {
    'I': 0.5,
    'Z0': -0.5,
    'Z1': -0.3,
    'Z0Z1': 0.2
}

def measure_pauli_term(circuit, pauli_term):
    """Measure a single Pauli term on the circuit"""
    meas_circuit = circuit.copy()

    # Add basis rotations and measurements
    if pauli_term == 'I':
        # Identity term: energy is always constant
        return 1.0
    elif pauli_term == 'Z0':
        meas_circuit.measure(0, 0)
        counts = run_circuit_and_get_counts(meas_circuit, backend, shots=1000)
        return (counts.get('0', 0) - counts.get('1', 0)) / 1000 if len(meas_circuit) > 0 else 1.0
    elif pauli_term == 'Z1':
        meas_circuit.measure(1, 1)
        counts = run_circuit_and_get_counts(meas_circuit, backend, shots=1000)
        return (counts.get('0', 0) - counts.get('1', 0)) / 1000 if len(meas_circuit) > 0 else 1.0
    elif pauli_term == 'Z0Z1':
        meas_circuit.measure([0, 1], [0, 1])
        counts = run_circuit_and_get_counts(meas_circuit, backend, shots=1000)
        z0z1_sum = 0
        for bitstring, count in counts.items():
            z0_val = 1 if bitstring[0] == '0' else -1
            z1_val = 1 if bitstring[1] == '0' else -1
            z0z1_sum += count * z0_val * z1_val
        return z0z1_sum / 1000

    return 0.0


def cost_function_2q(params):
    """
    VQE cost function for 2-qubit system.
    Ansatz: RY rotations on each qubit
    """
    circuit = QuantumCircuit(2, 2)
    # Simple ansatz: RY gates
    circuit.ry(params[0], 0)
    circuit.ry(params[1], 1)

    # Measure each Pauli term
    energy = 0.0
    for pauli, coeff in h_coeffs.items():
        expectation = measure_pauli_term(circuit, pauli)
        energy += coeff * expectation

    return energy


print("\nAnsatz: |ψ(θ₀,θ₁)⟩ = RY(θ₀)RY(θ₁)|0⟩|0⟩")
print("Optimizing with COBYLA...\n")

params_init = [0.1, 0.1]
result_2q = minimize(
    cost_function_2q,
    params_init,
    method='COBYLA',
    options={'maxiter': 30, 'tol': 0.01}
)

print(f"Optimization Results:")
print(f"  Initial params: {params_init}, E: {cost_function_2q(params_init):.4f}")
print(f"  Final params: {result_2q.x}, E: {result_2q.fun:.4f}")
print(f"  Converged: {result_2q.success}")
print()


# ============================================================================
# EXPERIMENT 6: Understanding Ansatz Design
# ============================================================================
print("=" * 70)
print("EXPERIMENT 6: Ansatz Design and Expressibility")
print("=" * 70)

print("\nAnsatz Design Considerations:\n")

print("1. HARDWARE EFFICIENT ANSATZ (Most practical)")
print("   Pattern: RY(θᵢ) - CNOT(entangling) - repeat")
print("   Pro: Few gates, implementable on NISQ devices")
print("   Con: May not express all quantum states")

print("\n2. HARDWARE EFFICIENT WITH ENTANGLEMENT")
print("   Pattern: [RY(θᵢ) on all qubits] - [CNOT chain] - repeat 2-3 times")
print("   Pro: Better expressibility, still practical")
print("   Con: More parameters, harder to optimize")

print("\n3. PROBLEM-SPECIFIC ANSATZ")
print("   Use problem structure to guide circuit design")
print("   Example: Coupled cluster ansatz for chemistry")
print("   Pro: Fewer parameters, faster convergence")
print("   Con: Requires domain knowledge")

print("\n4. UNITARY COUPLED CLUSTER (UCC) - For Chemistry")
print("   UCC = e^(T - T†) where T is cluster operator")
print("   Pro: Chemically motivated, good for molecules")
print("   Con: Complex to implement")

print("\nAnsatz Expressibility:")
print("  - With 1 rotation layer: Limited to 2^n states of simpler subspace")
print("  - With k rotation layers: Approaches full 2^n dimensional Hilbert space")
print("  - Trade-off: More layers = better expressibility but harder optimization\n")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: Energy vs Parameter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("VQE Optimization: Energy Landscape", fontsize=14, fontweight='bold')

# Panel 1: Simple 1-qubit system
theta_range = np.linspace(0, 2*np.pi, 50)
energy_theoretical = np.cos(theta_range)
energy_measured = []

print("\nScanning parameter space for simple VQE (this takes time)...")
for theta in theta_range:
    circuit = QuantumCircuit(1, 1)
    circuit.ry(theta, 0)
    circuit.measure(0, 0)
    counts = run_circuit_and_get_counts(circuit, backend, shots=500)
    energy = (counts.get('0', 0) - counts.get('1', 0)) / 500
    energy_measured.append(energy)

ax1.plot(theta_range, energy_theoretical, 'o-', linewidth=2.5, markersize=6,
         color='#1f77b4', label='Theoretical cos(θ)', zorder=3)
ax1.plot(theta_range, energy_measured, 's--', linewidth=1.5, markersize=4,
         color='#ff7f0e', alpha=0.7, label='Measured')
ax1.axhline(y=-1, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Ground state E=-1')
ax1.axvline(x=np.pi, color='g', linestyle=':', linewidth=2, alpha=0.5)
ax1.set_xlabel('Parameter θ (radians)', fontsize=12)
ax1.set_ylabel('Energy ⟨Z⟩', fontsize=12)
ax1.set_title('1-Qubit System: H = Z')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

# Panel 2: VQE convergence
iteration_history_1q = []
energy_history_1q = []

def callback_1q(xk):
    energy_history_1q.append(cost_function_simple([xk[0]]))

print("Re-running optimization with callback for convergence plot...")
result_1q_conv = minimize(
    cost_function_simple,
    [0.1],
    method='COBYLA',
    callback=callback_1q,
    options={'maxiter': 15, 'tol': 0.001}
)

iterations = list(range(len(energy_history_1q)))
ax2.plot(iterations, energy_history_1q, 'o-', linewidth=2.5, markersize=8, color='#2ca02c')
ax2.axhline(y=-1, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Ground state')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Energy', fontsize=12)
ax2.set_title('VQE Convergence: 1-Qubit System')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('vqe_optimization.png', dpi=150, bbox_inches='tight')
print("✓ Saved: vqe_optimization.png")


# Figure 2: Pauli Measurement Demonstration
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Pauli Expectation Values on Different States", fontsize=14, fontweight='bold')

states_data = [
    ("|0⟩", counts_0, (0, 0)),
    ("|1⟩", counts_1, (0, 1)),
    ("|+⟩", counts_plus, (1, 0)),
    ("Superposition", counts_plus, (1, 1))
]

expectations = [z_expect_0, z_expect_1, z_expect_plus, z_expect_plus]
state_names = ["|0⟩", "|1⟩", "|+⟩", "|+⟩"]

for idx, ((title, counts, pos), exp_val, state_name) in enumerate(zip(states_data, expectations, state_names)):
    ax = axes[pos]
    outcomes = ['0', '1']
    values = [counts.get('0', 0), counts.get('1', 0)]

    bars = ax.bar(outcomes, values, color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1000])

    # Add expectation value text
    ax.text(0.5, 0.95, f'⟨Z⟩ = {exp_val:.3f}', transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('vqe_pauli_measurements.png', dpi=150, bbox_inches='tight')
print("✓ Saved: vqe_pauli_measurements.png")


# Figure 3: VQE Workflow Diagram
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Variational Quantum Eigensolver (VQE) Workflow", fontsize=14, fontweight='bold')

# Step 1
ax = axes[0, 0]
ax.text(0.5, 0.85, "Step 1: Define Hamiltonian", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.65, "H = Σᵢ hᵢ Pᵢ", ha='center', fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.5, 0.5, "Express as Pauli terms", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.25, "Example:", ha='center', fontsize=9, transform=ax.transAxes, fontweight='bold')
ax.text(0.5, 0.12, "H = 0.5·I - 0.5·Z₀", ha='center', fontsize=9, transform=ax.transAxes, family='monospace')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 2
ax = axes[0, 1]
ax.text(0.5, 0.85, "Step 2: Choose Ansatz", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.65, "|ψ(θ)⟩ = U(θ)|0⟩", ha='center', fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.5, 0.5, "Parameterized circuit", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.3, "θ = adjustable parameters", ha='center', fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.15, "(gates to optimize)", ha='center', fontsize=9, transform=ax.transAxes)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 3
ax = axes[0, 2]
ax.text(0.5, 0.85, "Step 3: Measure Energy", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.65, "E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩", ha='center', fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.5, 0.5, "Measure each Pauli term:", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.35, "⟨Pᵢ⟩ on quantum computer", ha='center', fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.15, "Sum: E = Σᵢ hᵢ⟨Pᵢ⟩", ha='center', fontsize=9, transform=ax.transAxes, family='monospace')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 4
ax = axes[1, 0]
ax.text(0.5, 0.85, "Step 4: Classical Optimization", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.65, "θ_new = Optimizer(E)", ha='center', fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.5, 0.5, "Use classical optimizer:", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.35, "COBYLA, SPSA, etc.", ha='center', fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.15, "Find θ that minimizes E", ha='center', fontsize=9, transform=ax.transAxes)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 5
ax = axes[1, 1]
ax.text(0.5, 0.85, "Step 5: Check Convergence", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.65, "Is ΔE < threshold?", ha='center', fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.5, 0.5, "If no: Go to Step 3", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.35, "If yes: Done!", ha='center', fontsize=10, transform=ax.transAxes, fontweight='bold')
ax.text(0.5, 0.15, "(No: repeats 3-5)", ha='center', fontsize=9, transform=ax.transAxes)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

# Step 6
ax = axes[1, 2]
ax.text(0.5, 0.85, "Step 6: Extract Result", ha='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.65, "E₀ = E(θ*)", ha='center', fontsize=10, transform=ax.transAxes, family='monospace')
ax.text(0.5, 0.5, "Ground state energy E₀", ha='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.35, "Parameters θ*", ha='center', fontsize=9, transform=ax.transAxes)
ax.text(0.5, 0.15, "Wavefunction |ψ₀⟩", ha='center', fontsize=9, transform=ax.transAxes)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

plt.tight_layout()
plt.savefig('vqe_workflow.png', dpi=150, bbox_inches='tight')
print("✓ Saved: vqe_workflow.png")


# Figure 4: Ansatz Designs
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Common VQE Ansatz Designs", fontsize=14, fontweight='bold')

ansatz_designs = [
    {
        'name': 'Single Layer (Simple)',
        'description': 'RY gates only\nNo entanglement\nExpressibility: Limited',
        'pros': ['Fast optimization', 'Few parameters'],
        'cons': ['Limited expressibility', 'May miss ground state']
    },
    {
        'name': 'Hardware Efficient',
        'description': 'Alternating:\nRY → CNOT chain\nRepeat k times',
        'pros': ['Good expressibility', 'Practical gates'],
        'cons': ['Many parameters', 'Slow convergence']
    },
    {
        'name': 'Problem-Specific (Molecular)',
        'description': 'Coupled Cluster\nor Ansatz design\nBased on chemistry',
        'pros': ['Few parameters', 'Fast convergence'],
        'cons': ['Domain knowledge needed', 'Complex setup']
    },
    {
        'name': 'Deep Circuit',
        'description': 'Many layers\nFull expressibility\nRepeat k times (k large)',
        'pros': ['Highest expressibility', 'Can represent any state'],
        'cons': ['Very slow', 'Noise dominates on NISQ']
    }
]

for idx, design in enumerate(ansatz_designs):
    ax = axes[idx // 2, idx % 2]

    # Title
    ax.text(0.5, 0.95, design['name'], ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Description
    ax.text(0.5, 0.75, design['description'], ha='center', fontsize=10,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Pros
    pros_text = "✓ " + "\n✓ ".join(design['pros'])
    ax.text(0.05, 0.45, pros_text, fontsize=9, transform=ax.transAxes, color='green', fontweight='bold')

    # Cons
    cons_text = "✗ " + "\n✗ ".join(design['cons'])
    ax.text(0.55, 0.45, cons_text, fontsize=9, transform=ax.transAxes, color='red', fontweight='bold')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')

plt.tight_layout()
plt.savefig('vqe_ansatz_designs.png', dpi=150, bbox_inches='tight')
print("✓ Saved: vqe_ansatz_designs.png")


# Figure 5: Applications
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("VQE Applications in Quantum Chemistry & Industry", fontsize=14, fontweight='bold')

applications = [
    {
        'title': 'Molecular Simulation',
        'molecules': 'H₂, HeH⁺, LiH',
        'status': 'Already demonstrated',
        'challenge': 'Accuracy, scaling',
        'company': 'IBM, Google, IonQ'
    },
    {
        'title': 'Drug Discovery',
        'molecules': 'Protein folding\nBinding affinity',
        'status': 'Research phase',
        'challenge': 'Large systems, noise',
        'company': 'Merck, Roche'
    },
    {
        'title': 'Materials Science',
        'molecules': 'Superconductors\nBatteries',
        'status': 'Active research',
        'challenge': 'Precision requirements',
        'company': 'Rigetti, Microsoft'
    },
    {
        'title': 'Catalysis',
        'molecules': 'Nitrogen fixation\nCO₂ capture',
        'status': 'Future applications',
        'challenge': 'System complexity',
        'company': 'Academic labs'
    }
]

for idx, app in enumerate(applications):
    ax = axes[idx // 2, idx % 2]

    # Title
    ax.text(0.5, 0.95, app['title'], ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Content
    y_pos = 0.8
    ax.text(0.05, y_pos, f"Molecules: {app['molecules']}", fontsize=10, transform=ax.transAxes)
    y_pos -= 0.15
    ax.text(0.05, y_pos, f"Status: {app['status']}", fontsize=10, transform=ax.transAxes, fontweight='bold')
    y_pos -= 0.15
    ax.text(0.05, y_pos, f"Challenge: {app['challenge']}", fontsize=9, transform=ax.transAxes, style='italic')
    y_pos -= 0.15
    ax.text(0.05, y_pos, f"Players: {app['company']}", fontsize=9, transform=ax.transAxes, color='blue')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')

plt.tight_layout()
plt.savefig('vqe_applications.png', dpi=150, bbox_inches='tight')
print("✓ Saved: vqe_applications.png")


print("\n" + "=" * 70)
print("VQE EXPLORATION COMPLETE!")
print("=" * 70)

print("\nKey Insights:")
print("✓ VQE is a hybrid quantum-classical algorithm")
print("✓ Works on NISQ devices (noisy, limited gates)")
print("✓ Variational principle: E(θ) ≥ E₀")
print("✓ Classical optimizer adjusts quantum circuit")
print("✓ Practical for molecular simulation TODAY")

print("\nWhy VQE is Important:")
print("✓ First practical quantum algorithm for near-term devices")
print("✓ Drug discovery & materials science applications")
print("✓ Foundation for quantum chemistry on quantum computers")
print("✓ Hybrid approach maximizes current hardware capabilities")

print("\nVQE Timeline:")
print("  2015: Proposed by Peruzzo et al.")
print("  2017: Demonstrated on real quantum computers")
print("  2018-2021: Industrial partnerships begin (Merck, Roche)")
print("  2022-2024: Scaling to larger molecules, better ansatzes")
print("  Future: Quantum advantage for realistic molecules")

print("\nNext Steps:")
print("1. Explore different ansatz designs on this system")
print("2. Try different optimizers (SPSA, ADAM)")
print("3. Add noise simulations to study robustness")
print("4. Implement for H₂ molecule (real quantum chemistry)")
print("5. Study error mitigation techniques")
print("6. Connect to Shor's algorithm or other applications\n")
