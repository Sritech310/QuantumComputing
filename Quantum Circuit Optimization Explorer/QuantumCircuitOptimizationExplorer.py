"""
Quantum Circuit Optimization
=============================
Explore techniques to reduce gate count, circuit depth, and optimize for hardware!

Why Optimize Quantum Circuits?

The Challenge:
  1. GATE COUNT: More gates → more error (noise accumulates)
  2. CIRCUIT DEPTH: Longer sequences → more decoherence (T1/T2 decay)
  3. HARDWARE CONSTRAINTS: Different qubits have different connectivity
  4. EXECUTION TIME: Deeper circuits take longer
  5. RESOURCE USAGE: Two-qubit gates are expensive

The Goal:
  Minimize: gate count, depth, 2Q gates, T-gates
  Subject to: correctness, hardware constraints
  Trade-off: Sometimes use more 1Q gates instead of 2Q gates

Optimization Techniques:

1. GATE MERGING (Single-Qubit Optimization)
   Combine consecutive single-qubit gates into one
   Example: RZ(θ₁)·RZ(θ₂) → RZ(θ₁+θ₂)
   Reduction: 2 gates → 1 gate

2. GATE CANCELLATION (Remove Redundancies)
   Remove sequences that cancel: U·U† = I
   Example: H·H = I, X·X = I, S·S† = I
   Reduction: Potentially 50-100% of gates

3. COMMUTATION (Reorder Gates)
   Reorder commuting gates for better structure
   Example: CNOT on different qubits can be parallelized
   Reduction: Circuit depth by parallelization

4. NATIVE GATE OPTIMIZATION
   Map to hardware-native gates (not all gates are native)
   Example: IBM native gates: {RZ, SX, X, CNOT}
   Reduction: Fewer decompositions needed

5. TWO-QUBIT GATE REDUCTION
   Minimize expensive 2Q gates
   Techniques: gate teleportation, better decompositions
   Reduction: 30-50% of 2Q gates

6. QUBIT MAPPING
   Map logical qubits to physical qubits
   Consider hardware connectivity constraints
   Minimize SWAP gates needed
   Reduction: Extra gates for routing

7. CIRCUIT SYNTHESIS
   Use optimal gate sequences for common operations
   Pre-computed libraries of optimal circuits
   Example: Toffoli with 1, 2, or 3 auxiliary qubits

8. PATTERN MATCHING (Peephole Optimization)
   Find inefficient patterns → replace with optimal versions
   Example: RZ-CNOT-RZ → more efficient decomposition
   Reduction: 10-30% of gates

Cost Metrics:

1. GATE COUNT: Total number of gates
   - 1-qubit gates: cheap
   - 2-qubit gates: expensive (100x slower, noisier)
   - Measurement: cheap (classical operation)

2. CIRCUIT DEPTH
   - Minimum time needed (gates on different qubits can be parallel)
   - Increases decoherence time
   - Critical for NISQ devices

3. TWO-QUBIT GATE COUNT
   - Most expensive gates (highest error rates)
   - Limit circuit fidelity more than 1Q gates

4. T-GATE COUNT
   - Used for non-Clifford operations
   - Expensive for magic state distillation
   - Target for optimizations in some applications

5. CRITICAL PATH LENGTH
   - Maximum dependent chain of gates
   - Determines minimum execution time

Hardware Constraints:

1. NATIVE GATES
   - IBM: {RZ, SX, X, CNOT}
   - Google: {FSIM, PhasedXPowGate}
   - IonQ: {RZ, RX, Molmer-Sorensen}

2. CONNECTIVITY
   - Linear chain: 1-2-3-4-5 (limited, requires SWAPs)
   - 2D grid: Most superconducting qubits
   - All-to-all: Trapped ion systems

3. TIMING
   - 1Q gate: ~25-50 ns
   - 2Q gate: ~200-400 ns
   - Measurement: ~1-5 μs

Optimization Trade-offs:

- More 1Q gates vs fewer 2Q gates: Usually trade 2-3 singles for one 2Q
- Depth vs gate count: Sometimes add gates to reduce depth
- Compilation time vs runtime: Complex optimization takes time
"""

# Core imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager, passes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import library as qiskit_lib
from qiskit_ibm_runtime import SamplerV2 as Sampler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import qiskit

print("Python:", sys.version.split()[0])
print("Qiskit:", qiskit.__version__)


def count_gates(circuit):
    """Count gates by type"""
    counts = {
        '1q_gates': 0,
        '2q_gates': 0,
        'total_gates': 0,
        'gate_types': {}
    }

    for inst, qargs, cargs in circuit.data:
        gate_name = inst.name
        num_qubits = len(qargs)

        if num_qubits == 1 and inst.name not in ['measure', 'barrier']:
            counts['1q_gates'] += 1
        elif num_qubits == 2:
            counts['2q_gates'] += 1

        if inst.name not in ['measure', 'barrier']:
            counts['total_gates'] += 1
            counts['gate_types'][gate_name] = counts['gate_types'].get(gate_name, 0) + 1

    return counts


def get_circuit_depth(circuit):
    """Get circuit depth (considering parallelization)"""
    return circuit.depth()


def analyze_circuit(circuit, name=""):
    """Comprehensive circuit analysis"""
    stats = {
        'name': name,
        'qubits': circuit.num_qubits,
        'depth': get_circuit_depth(circuit),
        'size': circuit.size(),
        **count_gates(circuit)
    }
    return stats


def print_circuit_stats(stats):
    """Print formatted circuit statistics"""
    print(f"\n{stats['name']}:")
    print(f"  Qubits: {stats['qubits']}")
    print(f"  Total Gates: {stats['total_gates']}")
    print(f"  1-Qubit Gates: {stats['1q_gates']}")
    print(f"  2-Qubit Gates: {stats['2q_gates']}")
    print(f"  Circuit Depth: {stats['depth']}")
    print(f"  Gate Types: {stats['gate_types']}")


# Initialize backend
backend = AerSimulator()
print(f"Using backend: {backend.name}\n")


# ============================================================================
# EXPERIMENT 1: Understanding Circuit Metrics
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Circuit Metrics and Cost Analysis")
print("=" * 70)

print("\nCircuit Optimization Metrics:\n")

print("1. GATE COUNT")
print("   Total gates in circuit")
print("   Focus: Reduce overall count")
print("   Trade-off: Sometimes add 1Q gates to remove 2Q gates")

print("\n2. CIRCUIT DEPTH")
print("   Longest path of dependent gates")
print("   Considers parallelization (gates on different qubits in parallel)")
print("   Focus: Minimize depth (reduces decoherence time)")
print("   Trade-off: May increase gate count to reduce depth")

print("\n3. TWO-QUBIT GATE COUNT")
print("   Most expensive gates (100x slower than 1Q, noisier)")
print("   Major source of errors")
print("   Focus: Aggressively minimize 2Q gates")
print("   Cost: ~10-20 μs per gate vs ~25-50 ns for 1Q")

print("\n4. T-GATE COUNT")
print("   Non-Clifford gates (expensive for error correction)")
print("   Important for magic state distillation")
print("   Focus: Minimize for error-corrected algorithms")

print("\n5. CRITICAL PATH LENGTH")
print("   Maximum chain of dependent gates")
print("   Determines minimum possible execution time")
print("   Focus: Reduce for faster execution")

print("\nExample Cost Analysis:")
print("  Circuit A: 100 gates, 50 depth, 20 2Q gates")
print("  Circuit B: 80 gates, 30 depth, 18 2Q gates")
print("  Better choice depends on hardware:")
print("    - Short T2 times: Choose B (lower depth)")
print("    - Long T2 times: Choose A (fewer gates)")
print("    - Many 2Q errors: Choose B (fewer 2Q gates)")
print()


# ============================================================================
# EXPERIMENT 2: Gate Merging and Cancellation
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: Gate Merging and Cancellation")
print("=" * 70)

"""
Simple optimizations that catch obvious redundancies
"""

print("\nGate Merging Example:")
print("  Before: RZ(π/4) → RZ(π/4) (2 gates)")
print("  After:  RZ(π/2) (1 gate)")
print("  Savings: 1 gate eliminated\n")

# Example 1: Merging
circuit_before_merge = QuantumCircuit(1, name="Before Merge")
circuit_before_merge.rz(np.pi/4, 0)
circuit_before_merge.rz(np.pi/4, 0)

circuit_after_merge = QuantumCircuit(1, name="After Merge")
circuit_after_merge.rz(np.pi/2, 0)

stats_before_merge = analyze_circuit(circuit_before_merge, "Before Merge")
stats_after_merge = analyze_circuit(circuit_after_merge, "After Merge")

print_circuit_stats(stats_before_merge)
print_circuit_stats(stats_after_merge)

print(f"Savings: {stats_before_merge['total_gates'] - stats_after_merge['total_gates']} gates")

print("\n" + "-" * 70)
print("\nGate Cancellation Example:")
print("  Before: H → H → X → X (4 gates)")
print("  After:  I (identity, can be removed)")
print("  Savings: 4 gates eliminated\n")

# Example 2: Cancellation
circuit_before_cancel = QuantumCircuit(1, name="Before Cancellation")
circuit_before_cancel.h(0)
circuit_before_cancel.h(0)
circuit_before_cancel.x(0)
circuit_before_cancel.x(0)

# Manually optimize
circuit_after_cancel = QuantumCircuit(1, name="After Cancellation")
# All gates cancel to identity

stats_before_cancel = analyze_circuit(circuit_before_cancel, "Before Cancellation")
stats_after_cancel = analyze_circuit(circuit_after_cancel, "After Cancellation")

print_circuit_stats(stats_before_cancel)
print_circuit_stats(stats_after_cancel)

print(f"Savings: {stats_before_cancel['total_gates'] - stats_after_cancel['total_gates']} gates")
print()


# ============================================================================
# EXPERIMENT 3: Depth Reduction Through Parallelization
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: Circuit Depth Reduction")
print("=" * 70)

"""
Reorder gates to exploit parallelization
Gates on different qubits can execute simultaneously
"""

print("\nDepth Optimization Example:")
print("  Before (Serial):")
print("    Layer 1: H(0)")
print("    Layer 2: CNOT(0,1)")
print("    Layer 3: H(2)")
print("    Depth: 3")

print("\n  After (Parallel):")
print("    Layer 1: H(0), H(2)")
print("    Layer 2: CNOT(0,1)")
print("    Depth: 2")

print("\n  Savings: 1 layer of depth\n")

# Example: Serial execution
circuit_serial = QuantumCircuit(3, name="Serial (Before Optimization)")
circuit_serial.h(0)
circuit_serial.cx(0, 1)
circuit_serial.h(2)

stats_serial = analyze_circuit(circuit_serial, "Serial Execution")

# Example: Parallel execution (same circuit, reordered)
circuit_parallel = QuantumCircuit(3, name="Parallel (Optimized)")
circuit_parallel.h(0)
circuit_parallel.h(2)
circuit_parallel.cx(0, 1)

stats_parallel = analyze_circuit(circuit_parallel, "Parallel Execution")

print_circuit_stats(stats_serial)
print_circuit_stats(stats_parallel)

print(f"Depth reduction: {stats_serial['depth'] - stats_parallel['depth']} layers")
print()


# ============================================================================
# EXPERIMENT 4: Two-Qubit Gate Optimization
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: Two-Qubit Gate Optimization")
print("=" * 70)

"""
Two-qubit gates are the most expensive.
Different decompositions have different costs.
"""

print("\nTwo-Qubit Gate Decompositions:\n")

print("1. DIRECT CNOT")
print("   Cost: 1 two-qubit gate")
print("   Depth: 1 layer")
print("   Best when: Hardware has good CNOT fidelity")

print("\n2. CNOT with SWAPs")
print("   Used when: Qubits not adjacent")
print("   Cost: SWAP = 3 CNOTs = 3 two-qubit gates")
print("   Example: Connect q0-q2 through q1")
print("     SWAP(0,1) + CNOT(0,2) + SWAP(0,1)")

print("\n3. DECOMPOSITION INTO NATIVE GATES")
print("   Example (IonQ): Use Molmer-Sorensen instead of CNOT")
print("   Cost: Different fidelity and timing")

print("\n4. GATE TELEPORTATION")
print("   Use entanglement to reduce 2Q gates")
print("   Cost: Pre-entanglement overhead")
print("   Benefit: Better parallelization")

print("\nExample Circuits:\n")

# Example 1: Direct CNOT
circuit_direct = QuantumCircuit(2, name="Direct CNOT")
circuit_direct.cx(0, 1)

# Example 2: With decomposition
circuit_decomposed = QuantumCircuit(2, name="CNOT Decomposed")
# CNOT = U3(π/2,0,π) × CNOT × U3(0,0,0)
# For simplicity, just show CNOT
circuit_decomposed.cx(0, 1)

stats_direct = analyze_circuit(circuit_direct, "Direct CNOT")
stats_decomposed = analyze_circuit(circuit_decomposed, "Decomposed")

print_circuit_stats(stats_direct)

print("\nKey Insight:")
print("  Different hardware prefers different decompositions")
print("  Optimization must account for target hardware")
print()


# ============================================================================
# EXPERIMENT 5: Hardware-Specific Optimization
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: Hardware-Specific Optimization")
print("=" * 70)

"""
Different quantum computers have different native gates and connectivity
Transpilation maps logical circuit to hardware-specific circuit
"""

print("\nHardware Profiles:\n")

hardware_profiles = [
    {
        'name': 'IBM Quantum',
        'native_gates': '{RZ, SX, X, CNOT}',
        'connectivity': '2D grid (limited)',
        '1q_time': '~25-50 ns',
        '2q_time': '~200-400 ns',
        'notes': 'Superconducting qubits, good for deep circuits'
    },
    {
        'name': 'Google Sycamore',
        'native_gates': '{FSIM, PhasedXPowGate}',
        'connectivity': '2D grid',
        '1q_time': '~20 ns',
        '2q_time': '~20-40 ns',
        'notes': 'Very fast gates, low connectivity overhead'
    },
    {
        'name': 'IonQ',
        'native_gates': '{RZ, RX, Molmer-Sorensen}',
        'connectivity': 'All-to-all',
        '1q_time': '~10 μs',
        '2q_time': '~100 μs',
        'notes': 'Long coherence, no routing needed'
    },
    {
        'name': 'Rigetti',
        'native_gates': '{RX, RZ, CZ}',
        'connectivity': 'Tunable (qubit-qubit couplings)',
        '1q_time': '~25 ns',
        '2q_time': '~150 ns',
        'notes': 'Good 2Q fidelity, hybrid approach'
    }
]

for hw in hardware_profiles:
    print(f"{hw['name']:20} | Native: {hw['native_gates']:30} | Connectivity: {hw['connectivity']}")
    print(f"  1Q time: {hw['1q_time']:15} 2Q time: {hw['2q_time']:15}")
    print(f"  Notes: {hw['notes']}\n")


# Demonstrate transpilation
print("Transpilation Example (converting to IBM-native gates):\n")

circuit_generic = QuantumCircuit(2, name="Generic Circuit")
circuit_generic.h(0)
circuit_generic.cx(0, 1)
circuit_generic.ry(np.pi/4, 1)

print("Original circuit (using generic gates):")
stats_generic = analyze_circuit(circuit_generic, "Generic")
print_circuit_stats(stats_generic)

# Transpile to IBM backend
backend_ibm_like = AerSimulator()
pm = generate_preset_pass_manager(backend=backend_ibm_like, optimization_level=2)
circuit_transpiled = pm.run(circuit_generic)

print("\nAfter transpilation (optimized for hardware):")
stats_transpiled = analyze_circuit(circuit_transpiled, "Transpiled (IBM-like)")
print_circuit_stats(stats_transpiled)

print(f"\nGate count change: {stats_generic['total_gates']} → {stats_transpiled['total_gates']}")
print(f"Depth change: {stats_generic['depth']} → {stats_transpiled['depth']}")
print()


# ============================================================================
# EXPERIMENT 6: Optimization Levels
# ============================================================================
print("=" * 70)
print("EXPERIMENT 6: Different Optimization Levels")
print("=" * 70)

"""
Qiskit provides different optimization levels:
- Level 0: No optimization (mapping only)
- Level 1: Light optimization
- Level 2: Aggressive optimization
- Level 3: Very aggressive (slow compilation)
"""

print("\nOptimization Levels in Qiskit:\n")

levels = [
    {
        'level': 0,
        'name': 'No optimization',
        'techniques': 'Only layout and routing',
        'compilation_time': 'Fast',
        'result_quality': 'Poor'
    },
    {
        'level': 1,
        'name': 'Light',
        'techniques': 'Gate merging, cancellation',
        'compilation_time': 'Fast',
        'result_quality': 'Good'
    },
    {
        'level': 2,
        'name': 'Aggressive',
        'techniques': 'All level 1 + commutation, depth optimization',
        'compilation_time': 'Slow',
        'result_quality': 'Very good'
    },
    {
        'level': 3,
        'name': 'Very Aggressive',
        'techniques': 'All level 2 + pattern matching, synthesis',
        'compilation_time': 'Very slow',
        'result_quality': 'Excellent'
    }
]

for level in levels:
    print(f"Level {level['level']}: {level['name']}")
    print(f"  Techniques: {level['techniques']}")
    print(f"  Compilation Time: {level['compilation_time']}")
    print(f"  Result Quality: {level['result_quality']}\n")

print("Recommendation:")
print("  - Level 1: Good balance for most circuits")
print("  - Level 2: For production/final compilation")
print("  - Level 3: Only for very important circuits (slow!)")
print("  - Level 0: Rarely used (defeats the purpose)")
print()


# ============================================================================
# EXPERIMENT 7: Optimization Trade-offs
# ============================================================================
print("=" * 70)
print("EXPERIMENT 7: Optimization Trade-offs")
print("=" * 70)

print("\nWhen to Optimize (Trade-offs):\n")

print("1. ALWAYS OPTIMIZE")
print("   - Gate count → Always reduces errors")
print("   - 2Q gate count → Always improves fidelity")
print("   - T-gate count → Important for error correction")

print("\n2. OPTIMIZE FOR DEPTH (Sometimes)")
print("   - Short T2 times → Reduce decoherence")
print("   - Limited routing → May increase 2Q gates")
print("   - Trade: 1-2 extra 1Q gates per reduced 2Q gate")

print("\n3. OPTIMIZE FOR SPEED (Sometimes)")
print("   - Time-sensitive algorithms")
print("   - Reduce circuit depth to minimize execution time")
print("   - Trade: May add gates for parallelization")

print("\n4. OPTIMIZE FOR CONNECTIVITY (Hardware-dependent)")
print("   - All-to-all hardware: Minimize SWAPs")
print("   - Limited connectivity: May need routing")
print("   - Trade: 3-4 CNOTs per SWAP to connect qubits")

print("\nDecision Tree:")
print("  Is T2 short? → Yes: Optimize depth")
print("                 No: Optimize gate count")
print("  Many 2Q errors? → Yes: Minimize 2Q gates")
print("  Hardware has connectivity constraints? → Yes: Plan routing")
print()


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: Gate Merging and Cancellation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Circuit Optimization: Gate Merging and Cancellation", fontsize=14, fontweight='bold')

# Panel 1: Before/After Merging
ax = axes[0, 0]
gates_merge_before = ['RZ', 'RZ']
gates_merge_after = ['RZ']
x = [0, 1]
labels = ['Before\n(2 gates)', 'After\n(1 gate)']

bars = ax.bar(x, [2, 1], color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Number of Gates', fontsize=11)
ax.set_title('Gate Merging: RZ(π/4) + RZ(π/4) → RZ(π/2)', fontsize=12, fontweight='bold')
ax.set_ylim([0, 3])

# Add value labels
for bar, val in zip(bars, [2, 1]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(val)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Panel 2: Cancellation
ax = axes[0, 1]
gates_cancel = ['Before\n(H·H·X·X)', 'After\n(I)']
x = [0, 1]

bars = ax.bar(x, [4, 0], color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(gates_cancel)
ax.set_ylabel('Number of Gates', fontsize=11)
ax.set_title('Gate Cancellation: Redundant Gates Removed', fontsize=12, fontweight='bold')
ax.set_ylim([0, 5])

# Add value labels
for bar, val in zip(bars, [4, 0]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2, f'{int(val)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Panel 3: Depth Reduction
ax = axes[1, 0]
depths = ['Serial\nExecution', 'Parallel\nOptimized']
depth_vals = [3, 2]

bars = ax.bar(range(len(depths)), depth_vals, color=['#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(depths)))
ax.set_xticklabels(depths)
ax.set_ylabel('Circuit Depth (layers)', fontsize=11)
ax.set_title('Depth Reduction Through Parallelization', fontsize=12, fontweight='bold')
ax.set_ylim([0, 4])

# Add value labels
for bar, val in zip(bars, depth_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(val)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Panel 4: Cost Analysis
ax = axes[1, 1]
circuits = ['Circuit A\n(100g, 50d, 20·2q)', 'Circuit B\n(80g, 30d, 18·2q)']
metrics = ['Gate Count', 'Depth', '2Q Gates']
circuit_a = [100, 50, 20]
circuit_b = [80, 30, 18]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, circuit_a, width, label='Circuit A', color='#1f77b4', alpha=0.7)
bars2 = ax.bar(x + width/2, circuit_b, width, label='Circuit B', color='#2ca02c', alpha=0.7)

ax.set_ylabel('Value', fontsize=11)
ax.set_title('Cost Comparison: Different Metrics', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=10)
ax.set_ylim([0, 120])

plt.tight_layout()
plt.savefig('qco_optimization_basics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qco_optimization_basics.png")


# Figure 2: Optimization Levels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Optimization Trade-offs: Quality vs Compilation Time", fontsize=14, fontweight='bold')

# Panel 1: Quality vs Time
optimization_levels = ['L0\n(None)', 'L1\n(Light)', 'L2\n(Aggressive)', 'L3\n(Very Agg.)']
quality_scores = [2, 6, 8, 9]
compilation_times = [0.1, 1, 5, 20]

ax = ax1
colors_opt = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
ax.scatter(compilation_times, quality_scores, s=500, c=colors_opt, alpha=0.6,
           edgecolors='black', linewidth=2, zorder=3)

for label, ct, qs in zip(optimization_levels, compilation_times, quality_scores):
    ax.annotate(label, (ct, qs), fontsize=11, fontweight='bold', ha='center', va='center')

ax.set_xlabel('Compilation Time (seconds, log scale)', fontsize=12)
ax.set_ylabel('Result Quality Score', fontsize=12)
ax.set_title('Quality vs Compilation Time')
ax.set_xscale('log')
ax.set_xlim([0.05, 50])
ax.set_ylim([0, 10])
ax.grid(True, alpha=0.3)

# Add recommendation region
ax.fill_between([0.5, 2], 0, 10, alpha=0.1, color='green', label='Recommended')
ax.legend(fontsize=10)

# Panel 2: Gate Count Reduction
ax = ax2
circuits_opt = ['L0\nNo Opt', 'L1\nLight', 'L2\nAgg', 'L3\nVery Agg']
gate_reductions = [100, 85, 72, 68]
colors_gate = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

bars = ax.bar(range(len(circuits_opt)), gate_reductions, color=colors_gate, alpha=0.7,
              edgecolor='black', linewidth=2)

ax.set_xticks(range(len(circuits_opt)))
ax.set_xticklabels(circuits_opt)
ax.set_ylabel('Gate Count (% of original)', fontsize=12)
ax.set_title('Gate Count Reduction by Optimization Level')
ax.set_ylim([0, 110])
ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Unoptimized')

# Add value labels
for bar, val in zip(bars, gate_reductions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{int(val)}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('qco_optimization_levels.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qco_optimization_levels.png")


# Figure 3: Hardware Profiles
fig, ax = plt.subplots(figsize=(14, 8))

hardware_names = ['IBM\nQuantum', 'Google\nSycamore', 'IonQ', 'Rigetti']
native_gate_counts = [4, 2, 3, 3]
connectivity_scores = [6, 7, 10, 8]  # 1-10 scale
speed_scores = [7, 9, 5, 7]  # 1-10 scale

x = np.arange(len(hardware_names))
width = 0.25

bars1 = ax.bar(x - width, native_gate_counts, width, label='Native Gates', color='#1f77b4', alpha=0.7)
bars2 = ax.bar(x, connectivity_scores, width, label='Connectivity (1-10)', color='#ff7f0e', alpha=0.7)
bars3 = ax.bar(x + width, speed_scores, width, label='Speed Score (1-10)', color='#2ca02c', alpha=0.7)

ax.set_ylabel('Score / Count', fontsize=12)
ax.set_title('Hardware Profiles: Different Optimization Priorities', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(hardware_names)
ax.legend(fontsize=11)
ax.set_ylim([0, 12])

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2, f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('qco_hardware_profiles.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qco_hardware_profiles.png")


# Figure 4: Optimization Decision Tree
fig, ax = plt.subplots(figsize=(14, 10))

ax.text(0.5, 0.95, "Circuit Optimization Decision Tree", ha='center', fontsize=14, fontweight='bold',
        transform=ax.transAxes)

# Root question
ax.add_patch(Rectangle((0.35, 0.80), 0.3, 0.08, transform=ax.transAxes,
                       facecolor='#1f77b4', alpha=0.6, edgecolor='black', linewidth=2))
ax.text(0.5, 0.84, "Is circuit too deep?", ha='center', va='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)

# Left branch: Yes (deep circuit)
ax.plot([0.5, 0.25], [0.80, 0.65], 'k-', linewidth=2, transform=ax.transAxes)
ax.text(0.35, 0.73, "YES\nT2 short", ha='center', fontsize=10, transform=ax.transAxes, fontweight='bold')

ax.add_patch(Rectangle((0.05, 0.55), 0.4, 0.08, transform=ax.transAxes,
                       facecolor='#ff7f0e', alpha=0.6, edgecolor='black', linewidth=2))
ax.text(0.25, 0.59, "Optimize for DEPTH", ha='center', va='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)

# Right branch: No (shallow circuit)
ax.plot([0.5, 0.75], [0.80, 0.65], 'k-', linewidth=2, transform=ax.transAxes)
ax.text(0.65, 0.73, "NO\nT2 long", ha='center', fontsize=10, transform=ax.transAxes, fontweight='bold')

ax.add_patch(Rectangle((0.55, 0.55), 0.4, 0.08, transform=ax.transAxes,
                       facecolor='#2ca02c', alpha=0.6, edgecolor='black', linewidth=2))
ax.text(0.75, 0.59, "Optimize for GATE COUNT", ha='center', va='center', fontsize=11, fontweight='bold',
        transform=ax.transAxes)

# Depth optimization branches
ax.plot([0.25, 0.10], [0.55, 0.40], 'k-', linewidth=1.5, transform=ax.transAxes)
ax.plot([0.25, 0.40], [0.55, 0.40], 'k-', linewidth=1.5, transform=ax.transAxes)

ax.add_patch(Rectangle((-0.05, 0.30), 0.3, 0.08, transform=ax.transAxes,
                       facecolor='#ff7f0e', alpha=0.4, edgecolor='black', linewidth=1))
ax.text(0.10, 0.34, "Gate merging", ha='center', va='center', fontsize=9,
        transform=ax.transAxes)

ax.add_patch(Rectangle((0.25, 0.30), 0.3, 0.08, transform=ax.transAxes,
                       facecolor='#ff7f0e', alpha=0.4, edgecolor='black', linewidth=1))
ax.text(0.40, 0.34, "Parallelization", ha='center', va='center', fontsize=9,
        transform=ax.transAxes)

# Gate count optimization branches
ax.plot([0.75, 0.60], [0.55, 0.40], 'k-', linewidth=1.5, transform=ax.transAxes)
ax.plot([0.75, 0.90], [0.55, 0.40], 'k-', linewidth=1.5, transform=ax.transAxes)

ax.add_patch(Rectangle((0.45, 0.30), 0.3, 0.08, transform=ax.transAxes,
                       facecolor='#2ca02c', alpha=0.4, edgecolor='black', linewidth=1))
ax.text(0.60, 0.34, "Gate cancellation", ha='center', va='center', fontsize=9,
        transform=ax.transAxes)

ax.add_patch(Rectangle((0.65, 0.30), 0.3, 0.08, transform=ax.transAxes,
                       facecolor='#2ca02c', alpha=0.4, edgecolor='black', linewidth=1))
ax.text(0.80, 0.34, "2Q minimization", ha='center', va='center', fontsize=9,
        transform=ax.transAxes)

# Final recommendation
ax.add_patch(Rectangle((0.2, 0.05), 0.6, 0.15, transform=ax.transAxes,
                       facecolor='#9467bd', alpha=0.3, edgecolor='black', linewidth=2))
ax.text(0.5, 0.14, "Final: Apply optimization level 2 for balance", ha='center', fontsize=11,
        transform=ax.transAxes, fontweight='bold')
ax.text(0.5, 0.08, "Level 3 only if very critical (slow compilation)", ha='center', fontsize=10,
        transform=ax.transAxes, style='italic')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.axis('off')

plt.tight_layout()
plt.savefig('qco_decision_tree.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qco_decision_tree.png")


# Figure 5: Optimization Impact Summary
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Overall Impact of Circuit Optimization", fontsize=14, fontweight='bold')

# Panel 1: Circuit evolution
ax = axes[0, 0]
stages = ['Original', 'After L1', 'After L2', 'After L3']
gate_counts = [150, 120, 95, 85]
colors_stages = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

bars = ax.bar(range(len(stages)), gate_counts, color=colors_stages, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages)
ax.set_ylabel('Gate Count', fontsize=11)
ax.set_title('Gate Count Reduction Through Optimization Levels')
ax.set_ylim([0, 180])

for bar, val in zip(bars, gate_counts):
    height = bar.get_height()
    reduction = ((150 - val) / 150) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height + 3, f'{int(val)}\n({reduction:.0f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 2: Error accumulation
ax = axes[0, 1]
stages_err = ['Original', 'After L1', 'After L2']
error_rates = [0.85, 0.88, 0.91]  # Fidelity

bars = ax.bar(range(len(stages_err)), error_rates, color=['#d62728', '#ff7f0e', '#2ca02c'],
              alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xticks(range(len(stages_err)))
ax.set_xticklabels(stages_err)
ax.set_ylabel('Estimated Fidelity', fontsize=11)
ax.set_title('Improved Fidelity Through Optimization\n(Fewer gates → less error)')
ax.set_ylim([0.80, 0.95])
ax.axhline(y=1.0, color='g', linestyle='--', linewidth=1.5, alpha=0.3, label='Ideal')

for bar, val in zip(bars, error_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{val:.1%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.legend()

# Panel 3: Depth vs Gates trade-off
ax = axes[1, 0]
circuits_names = ['A: Heavy', 'B: Balanced', 'C: Shallow']
depths = [50, 35, 25]
gates = [200, 180, 160]

scatter = ax.scatter(gates, depths, s=800, c=['#d62728', '#ff7f0e', '#2ca02c'],
                    alpha=0.6, edgecolors='black', linewidth=2, zorder=3)

for name, g, d in zip(circuits_names, gates, depths):
    ax.annotate(name, (g, d), fontsize=11, fontweight='bold', ha='center', va='center')

ax.set_xlabel('Total Gate Count', fontsize=11)
ax.set_ylabel('Circuit Depth', fontsize=11)
ax.set_title('Trade-off: Depth vs Gate Count')
ax.grid(True, alpha=0.3)

# Panel 4: Compilation time
ax = axes[1, 1]
opt_levels_comp = ['L1', 'L2', 'L3']
compile_times = [0.5, 3, 15]
gate_reduction = [15, 37, 43]

ax2 = ax.twinx()

bars1 = ax.bar(np.arange(len(opt_levels_comp)) - 0.2, compile_times, 0.4,
               label='Compilation Time', color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(np.arange(len(opt_levels_comp)) + 0.2, gate_reduction, 0.4,
                label='Gate Reduction %', color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xticks(range(len(opt_levels_comp)))
ax.set_xticklabels(opt_levels_comp)
ax.set_ylabel('Compilation Time (seconds)', fontsize=11, color='#1f77b4')
ax2.set_ylabel('Gate Reduction (%)', fontsize=11, color='#2ca02c')
ax.set_title('Compilation Time vs Optimization Benefit')
ax.tick_params(axis='y', labelcolor='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#2ca02c')
ax.set_ylim([0, 20])
ax2.set_ylim([0, 50])

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('qco_optimization_impact.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qco_optimization_impact.png")


print("\n" + "=" * 70)
print("QUANTUM CIRCUIT OPTIMIZATION EXPLORATION COMPLETE!")
print("=" * 70)

print("\nKey Insights:")
print("✓ Gate count directly affects error rates (fewer gates → better fidelity)")
print("✓ Circuit depth determines decoherence impact (shorter is better for short T2)")
print("✓ 2Q gates are most expensive (100x slower, noisier than 1Q)")
print("✓ Hardware-specific optimization is essential (native gates vary)")
print("✓ Trade-offs exist: sometimes add gates to reduce depth")

print("\nOptimization Techniques:")
print("  1. Gate Merging: Combine adjacent 1Q gates")
print("  2. Gate Cancellation: Remove U·U† sequences")
print("  3. Parallelization: Reorder for parallelism")
print("  4. Commutation: Exploit commuting gates")
print("  5. Native Gate Optimization: Map to hardware gates")
print("  6. 2Q Gate Reduction: Minimize expensive gates")
print("  7. Pattern Matching: Replace inefficient patterns")
print("  8. Qubit Mapping: Optimize qubit placement")

print("\nPractical Optimization Workflow:")
print("  1. Choose optimization level (typically L2)")
print("  2. Select target hardware")
print("  3. Run transpiler with constraints")
print("  4. Verify correctness (same unitary)")
print("  5. Measure improvement (gates, depth, fidelity)")

print("\nWhen Optimization Matters Most:")
print("  • Deep circuits (>100 gates)")
print("  • Short T2 times")
print("  • High 2Q gate error rates")
print("  • Limited qubit connectivity")
print("  • Production/repeated execution")

print("\nOptimization Myth vs Reality:")
print("  MYTH: More optimization = always better")
print("  REALITY: Trade-offs exist, compile time matters")
print()
print("  MYTH: Optimization is only for experts")
print("  REALITY: Qiskit/Cirq handle it automatically (L1-L2 default)")
print()
print("  MYTH: Can optimize away all errors")
print("  REALITY: Mitigation is complementary (need both optimization + mitigation)")

print("\nNext Steps:")
print("1. Apply optimization levels to your VQE program")
print("2. Compare fidelity with/without optimization")
print("3. Measure compilation time vs gate reduction")
print("4. Explore device-specific optimization")
print("5. Combine optimization with error mitigation")
print("6. Study transpiler passes in detail")
print("7. Profile circuit bottlenecks\n")
