"""
Quantum Classification & Quantum Machine Learning
==================================================
Explore quantum approaches to machine learning classification!

What is Quantum Classification?

Classical Classification:
  Input: Feature vector x = [x₁, x₂, ..., xₙ]
  Process: ML model f(x, θ) with parameters θ
  Output: Probability of each class
  Training: Optimize θ to minimize error

Quantum Classification:
  Input: Feature vector x → encoded into quantum state
  Process: Parameterized quantum circuit U(θ) with trainable gates
  Output: Measure quantum state → class prediction
  Training: Classical optimizer adjusts θ
  Key: Quantum feature space may reveal patterns classical ML misses

Why Quantum Classification?

Potential Advantages:
  1. Quantum Feature Space: Map to exponentially larger space
  2. Quantum Entanglement: Encode correlations between features
  3. Interference: Amplify correct class, cancel noise
  4. Superposition: Process multiple paths simultaneously
  5. Kernel Methods: Compute similarity in quantum space

Current Reality (NISQ Era):
  - Not faster than best classical ML yet
  - Useful for hybrid approaches
  - Proof-of-concept for quantum advantage
  - Near-term applications emerging

Key Components:

1. FEATURE MAP (Encoding)
   Convert classical data → quantum state
   Example: x = [x₁, x₂] → RY(x₁)|0⟩ → CNOT → RY(x₂)
   Creates: |ψ(x)⟩ encoding the input features

2. ANSATZ (Variational Circuit)
   Parameterized quantum circuit with adjustable gates
   Example: [RY(θ₁) - CNOT - RY(θ₂) - CNOT - ...]
   Similar to hidden layers in neural networks
   Purpose: Learn decision boundary

3. MEASUREMENT
   Measure qubits to get output
   Probabilities → class prediction
   Example: P(0) > 0.5 → Class A, else Class B

4. CLASSICAL OPTIMIZATION
   Classical optimizer (COBYLA, Adam, etc.)
   Adjusts θ to maximize classification accuracy
   Loss function: Cross-entropy, hinge loss, etc.

5. TRAINING & TESTING
   Train: Adjust θ on training data
   Test: Evaluate on unseen data
   Metric: Accuracy, precision, recall, etc.

Types of Quantum Classifiers:

1. QUANTUM NEURAL NETWORKS (QNN)
   - Parameterized circuits like neural networks
   - Multiple layers of rotation + entanglement
   - Flexible architecture
   - Con: May have barren plateaus (training hard)

2. VARIATIONAL QUANTUM CLASSIFIERS (VQC)
   - Similar to VQE but for classification
   - Classier-specific ansatz
   - Good for small datasets
   - Con: Limited scalability

3. QUANTUM KERNEL METHODS
   - Encode data into quantum state |ψ(x)⟩
   - Compute kernel K(x,y) = |⟨ψ(x)|ψ(y)⟩|²
   - Use classical SVM with quantum kernel
   - Pro: Mathematically elegant
   - Con: Sampling overhead

4. QUANTUM APPROXIMATE OPTIMIZATION (QAOA)
   - Optimize parameterized circuit for specific problem
   - Hybrid classical-quantum
   - Good for combinatorial problems
   - Con: Problem-specific

Challenges:

1. BARREN PLATEAUS
   - Random circuits have vanishing gradients
   - Hard to train without good initialization
   - Solution: Problem-informed ansatz design

2. NOISE IN NISQ DEVICES
   - Quantum gates have errors
   - Limits circuit depth
   - Solution: Error mitigation, shallow circuits

3. DATA ENCODING
   - How to map classical data to quantum?
   - Feature map design crucial
   - Different maps have different expressibility

4. SCALING
   - Exponential Hilbert space
   - But sampling is classical-hard
   - Quantum advantage unclear for most problems

5. TRAINING TIME
   - Classical optimizer calls circuit many times
   - Each evaluation runs quantum circuit + measurement
   - Overhead can exceed classical methods

Current Applications:

- Quantum ML research (Pennylane, Qiskit ML)
- Hybrid classical-quantum approaches
- Small datasets (few samples)
- Toy problems and demos
- Real applications emerging (startups exploring)

Expected Timeline:

2024: NISQ demonstrations, hybrid algorithms
2025-2027: Better algorithms, more qubits
2028+: Potential quantum advantage for specific problems
2030+: Scaled quantum ML systems

Realistic Expectations:

NOT coming soon:
  ✗ Beating classical ML on ImageNet
  ✗ Revolutionary speedups for all ML
  ✗ Replacing all deep learning

Potentially valuable:
  ✓ Hybrid classical-quantum workflows
  ✓ Specific domains (chemistry, optimization)
  ✓ Novel approaches to kernel methods
  ✓ Understanding quantum properties of data
"""

# Core imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import sys
import qiskit

print("Python:", sys.version.split()[0])
print("Qiskit:", qiskit.__version__)


def run_circuit_and_get_counts(circuit, backend, shots=1000):
    """Run circuit and return measurement counts"""
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
# EXPERIMENT 1: Understanding Quantum Classification
# ============================================================================
print("=" * 70)
print("EXPERIMENT 1: Quantum Classification Concepts")
print("=" * 70)

print("\nQuantum Classification Pipeline:\n")

print("1. DATA PREPARATION")
print("   - Classical dataset: {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}")
print("   - Features: x ∈ ℝᵈ (d-dimensional)")
print("   - Labels: y ∈ {0, 1} (binary classification)")
print("   - Normalization: Scale features to [0, 2π] or [-1, 1]")

print("\n2. FEATURE ENCODING")
print("   - Convert x → quantum state |ψ(x)⟩")
print("   - Method 1: Angle encoding: RY(2·arcsin(xᵢ))|0⟩")
print("   - Method 2: Amplitude encoding: |ψ⟩ ∝ Σᵢ xᵢ|i⟩")
print("   - Method 3: IQP encoding: Entangled rotations")

print("\n3. PARAMETERIZED CIRCUIT")
print("   - Trainable quantum circuit U(θ)")
print("   - Multiple layers of:")
print("     • Single-qubit rotations: RY(θᵢ), RZ(θᵢ)")
print("     • Entanglement: CNOT between qubits")
print("   - Learn decision boundary in quantum space")

print("\n4. MEASUREMENT")
print("   - Measure output qubit")
print("   - P(0): probability of measuring 0")
print("   - P(1): probability of measuring 1")
print("   - Prediction: P(0) > threshold → Class 0, else Class 1")

print("\n5. CLASSICAL OPTIMIZATION")
print("   - Loss: L(θ) = cross-entropy or hinge loss")
print("   - Optimizer: COBYLA, Adam, SPSA")
print("   - Goal: Minimize L(θ) on training data")

print("\n6. EVALUATION")
print("   - Train accuracy: How well on training set")
print("   - Test accuracy: How well on unseen data")
print("   - Generalization: Avoid overfitting")
print()


# ============================================================================
# EXPERIMENT 2: Simple 2-Feature Classification
# ============================================================================
print("=" * 70)
print("EXPERIMENT 2: 2-Qubit Quantum Classifier")
print("=" * 70)

"""
Simple binary classification on 2D dataset
2 qubits: 1 for encoding features, 1 for output
"""

print("\nGenerating synthetic dataset...")

# Generate 2-class dataset
n_samples = 20
np.random.seed(42)
X, y = make_blobs(n_samples=n_samples, n_features=2, centers=2, random_state=42)

# Normalize features to [-1, 1]
X = StandardScaler().fit_transform(X)
X = np.tanh(X)  # Map to approximately [-1, 1]

print(f"Dataset: {n_samples} samples, 2 features, 2 classes")
print(f"Class 0: {np.sum(y==0)} samples")
print(f"Class 1: {np.sum(y==1)} samples")
print(f"Feature range: [{X.min():.2f}, {X.max():.2f}]\n")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples\n")


def create_feature_encoding(x, qc):
    """Encode 2D feature into 2-qubit state"""
    # Amplitude encoding: encode both features
    theta1 = np.arcsin(x[0])
    theta2 = np.arcsin(x[1])

    qc.ry(2 * theta1, 0)
    qc.ry(2 * theta2, 1)
    qc.cx(0, 1)  # Entangle


def create_ansatz(params, qc):
    """Parameterized circuit for learning"""
    # Simple ansatz: rotation + entanglement layers
    n_params_per_layer = 2  # 2 qubits, 1 parameter each
    n_layers = len(params) // n_params_per_layer

    for layer in range(n_layers):
        for qubit in range(2):
            idx = layer * n_params_per_layer + qubit
            qc.ry(params[idx], qubit)
        qc.cx(0, 1)  # Entangle


def quantum_classifier(x, params, shots=500):
    """Run quantum classifier and return P(1)"""
    qc = QuantumCircuit(2, 1, name="QC")

    # Feature encoding
    create_feature_encoding(x, qc)

    # Ansatz
    create_ansatz(params, qc)

    # Measure output qubit
    qc.measure(1, 0)

    # Run circuit
    counts = run_circuit_and_get_counts(qc, backend, shots=shots)

    # Get probability of class 1
    p_1 = counts.get('1', 0) / shots
    return p_1


# Train the quantum classifier
print("Training quantum classifier (this will take ~2-3 minutes)...\n")

n_params = 4  # 2 layers, 2 qubits each
params_init = np.random.random(n_params) * np.pi

training_losses = []
training_accs = []

def loss_function(params):
    """Classification loss on training data"""
    loss = 0
    correct = 0

    for x, label in zip(X_train, y_train):
        # Get probability of class 1
        p_1 = quantum_classifier(x, params, shots=200)

        # Binary cross-entropy loss
        p_1 = np.clip(p_1, 1e-6, 1 - 1e-6)
        if label == 1:
            loss += -np.log(p_1)
        else:
            loss += -np.log(1 - p_1)

        # Accuracy
        prediction = 1 if p_1 > 0.5 else 0
        if prediction == label:
            correct += 1

    avg_loss = loss / len(X_train)
    accuracy = correct / len(X_train)

    training_losses.append(avg_loss)
    training_accs.append(accuracy)

    if len(training_losses) % 5 == 0:
        print(f"Iteration {len(training_losses)}: Loss = {avg_loss:.4f}, Accuracy = {accuracy*100:.1f}%")

    return avg_loss


# Optimize
print("Optimization (COBYLA with limited iterations for demo):\n")
result = minimize(
    loss_function,
    params_init,
    method='COBYLA',
    options={'maxiter': 20, 'tol': 0.001}
)

params_trained = result.x

print(f"\nTraining complete!")
print(f"Initial loss: {training_losses[0]:.4f}")
print(f"Final loss: {training_losses[-1]:.4f}")
print(f"Initial accuracy: {training_accs[0]*100:.1f}%")
print(f"Final accuracy: {training_accs[-1]*100:.1f}%\n")

# Evaluate on test set
print("Evaluating on test set...\n")

test_accs = []
for x, label in zip(X_test, y_test):
    p_1 = quantum_classifier(x, params_trained, shots=500)
    prediction = 1 if p_1 > 0.5 else 0
    is_correct = prediction == label
    test_accs.append(is_correct)
    print(f"  Sample: x={x}, True={label}, P(1)={p_1:.2f}, Pred={prediction}, {'✓' if is_correct else '✗'}")

test_accuracy = np.mean(test_accs)
print(f"\nTest accuracy: {test_accuracy*100:.1f}%\n")


# ============================================================================
# EXPERIMENT 3: Comparison with Classical ML
# ============================================================================
print("=" * 70)
print("EXPERIMENT 3: Quantum vs Classical Classifier")
print("=" * 70)

print("\nTraining classical SVM for comparison...\n")

# Classical SVM
clf_svm = SVC(kernel='rbf', C=1.0)
clf_svm.fit(X_train, y_train)

train_acc_svm = clf_svm.score(X_train, y_train)
test_acc_svm = clf_svm.score(X_test, y_test)

print(f"Classical SVM Results:")
print(f"  Training accuracy: {train_acc_svm*100:.1f}%")
print(f"  Test accuracy: {test_acc_svm*100:.1f}%\n")

# Quantum classifier accuracy (already computed)
train_acc_quantum = training_accs[-1]

print(f"Quantum Classifier Results:")
print(f"  Training accuracy: {train_acc_quantum*100:.1f}%")
print(f"  Test accuracy: {test_accuracy*100:.1f}%\n")

print(f"Comparison:")
print(f"  Quantum train edge: {(train_acc_quantum - train_acc_svm)*100:+.1f}%")
print(f"  Quantum test edge: {(test_accuracy - test_acc_svm)*100:+.1f}%")
print()


# ============================================================================
# EXPERIMENT 4: Feature Map Analysis
# ============================================================================
print("=" * 70)
print("EXPERIMENT 4: Quantum Feature Maps")
print("=" * 70)

print("\nDifferent Feature Encoding Methods:\n")

encodings = [
    {
        'name': 'Angle Encoding',
        'description': 'RY(2·arcsin(xᵢ))|0⟩ on each qubit',
        'pros': ['Simple', 'Shallow circuit', 'Differentiable'],
        'cons': ['Limited expressibility', 'x must be in [-1,1]'],
        'use': 'Small datasets, few features'
    },
    {
        'name': 'Amplitude Encoding',
        'description': '|ψ⟩ = (x₁|0⟩ + x₂|1⟩ + ...)/norm',
        'pros': ['Exponential compression', 'Full state info'],
        'cons': ['Requires log(n) qubits for n features', 'State prep overhead'],
        'use': 'High-dimensional data'
    },
    {
        'name': 'IQP Encoding',
        'description': 'Entangled structure: RZ(x²) + CZ + RZ(x²)',
        'pros': ['Efficient', 'Quantum advantage potential'],
        'cons': ['More complex', 'Harder to train'],
        'use': 'Research, specific problems'
    },
    {
        'name': 'Basis Encoding',
        'description': 'Binary string: |x⟩ where x ∈ {0,1}ⁿ',
        'pros': ['Natural for binary data', 'No encoding overhead'],
        'cons': ['Only binary features', 'Limited expressibility'],
        'use': 'Binary classification problems'
    }
]

for enc in encodings:
    print(f"{enc['name']}:")
    print(f"  Description: {enc['description']}")
    print(f"  Pros: {', '.join(enc['pros'])}")
    print(f"  Cons: {', '.join(enc['cons'])}")
    print(f"  Best for: {enc['use']}\n")


# ============================================================================
# EXPERIMENT 5: Quantum Kernel Methods
# ============================================================================
print("=" * 70)
print("EXPERIMENT 5: Quantum Kernel Methods")
print("=" * 70)

print("\nQuantum Kernel Method Concept:\n")

print("Traditional Kernel Method (Classical):")
print("  1. Define kernel K(x,y) = φ(x)·φ(y)")
print("  2. Use with SVM: solve convex optimization")
print("  3. Classification: f(x) = Σᵢ αᵢyᵢK(xᵢ,x) + b")

print("\nQuantum Kernel Method:")
print("  1. Feature map: x → |ψ(x)⟩ (quantum state)")
print("  2. Quantum kernel: K(x,y) = |⟨ψ(x)|ψ(y)⟩|²")
print("  3. Measurement: Run circuit(x,y) → measure overlap")
print("  4. Use result with classical SVM")

print("\nAdvantages:")
print("  ✓ Quantum-enhanced similarity measure")
print("  ✓ May separate data better than classical")
print("  ✓ Hybrid: use quantum for kernel, classical for SVM")
print("  ✓ Potentially exponential feature space")

print("\nChallenges:")
print("  ✗ Sampling overhead (need multiple measurements)")
print("  ✗ Noise in quantum circuits affects kernel")
print("  ✗ Unclear when quantum > classical")
print("  ✗ Scaling questionable for large datasets")
print()


# ============================================================================
# EXPERIMENT 6: Challenges and Considerations
# ============================================================================
print("=" * 70)
print("EXPERIMENT 6: Quantum ML Challenges")
print("=" * 70)

print("\nKey Challenges in Quantum Classification:\n")

challenges = [
    {
        'name': 'Barren Plateaus',
        'description': 'Random circuits have vanishing gradients',
        'impact': 'Hard to train, get stuck in local minima',
        'solution': 'Use problem-informed ansatz, good initialization'
    },
    {
        'name': 'Data Encoding',
        'description': 'How to convert classical data to quantum?',
        'impact': 'Feature map design crucial for performance',
        'solution': 'Experiment with different encodings, domain knowledge'
    },
    {
        'name': 'Noise & Errors',
        'description': 'NISQ devices have gate errors, decoherence',
        'impact': 'Limits circuit depth, reduces accuracy',
        'solution': 'Use error mitigation, keep circuits shallow'
    },
    {
        'name': 'Sampling Overhead',
        'description': 'Each evaluation requires many measurements',
        'impact': 'Training is slow (quantum bottleneck)',
        'solution': 'Hybrid approaches, batch optimization'
    },
    {
        'name': 'Scalability',
        'description': 'How to scale to realistic datasets?',
        'impact': 'Current systems limited to small problems',
        'solution': 'Fault-tolerant quantum computers (future)'
    },
    {
        'name': 'Classical Comparison',
        'description': 'Classical ML is very good (decades of R&D)',
        'impact': 'Hard to beat on most practical problems',
        'solution': 'Find niche problems, hybrid approaches'
    }
]

for i, challenge in enumerate(challenges, 1):
    print(f"{i}. {challenge['name']}")
    print(f"   Problem: {challenge['description']}")
    print(f"   Impact: {challenge['impact']}")
    print(f"   Solution: {challenge['solution']}\n")


# ============================================================================
# VISUALIZATION
# ============================================================================
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Figure 1: Dataset Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Quantum Classification: 2D Dataset and Decision Boundaries", fontsize=14, fontweight='bold')

# Panel 1: Original dataset
ax = axes[0]
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', s=100,
                     alpha=0.7, edgecolors='black', linewidth=1.5, label='Training')
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', s=100, alpha=0.3,
          marker='^', edgecolors='black', linewidth=1.5, label='Test')
ax.set_xlabel('Feature 1', fontsize=11)
ax.set_ylabel('Feature 2', fontsize=11)
ax.set_title('Original Dataset')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Quantum classifier decision boundary
ax = axes[1]
h = 0.02
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Create prediction grid (sample for speed)
Z_quantum = np.zeros(xx.shape)
for i in range(0, xx.shape[0], 5):  # Sample every 5th point for speed
    for j in range(0, xx.shape[1], 5):
        x_point = np.array([xx[i, j], yy[i, j]])
        p_1 = quantum_classifier(x_point, params_trained, shots=100)
        Z_quantum[i, j] = p_1

ax.contourf(xx, yy, Z_quantum, levels=20, cmap='RdYlBu', alpha=0.6)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', s=100,
          alpha=0.9, edgecolors='black', linewidth=1.5)
ax.set_xlabel('Feature 1', fontsize=11)
ax.set_ylabel('Feature 2', fontsize=11)
ax.set_title('Quantum Classifier Decision Boundary')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Panel 3: Classical SVM decision boundary
ax = axes[2]
Z_svm = clf_svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z_svm, levels=20, cmap='RdYlBu', alpha=0.6)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', s=100,
          alpha=0.9, edgecolors='black', linewidth=1.5)
ax.set_xlabel('Feature 1', fontsize=11)
ax.set_ylabel('Feature 2', fontsize=11)
ax.set_title('Classical SVM Decision Boundary')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('qc_decision_boundaries.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qc_decision_boundaries.png")


# Figure 2: Training Convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Quantum Classifier Training", fontsize=14, fontweight='bold')

# Panel 1: Loss convergence
iterations = range(len(training_losses))
ax1.plot(iterations, training_losses, 'o-', linewidth=2.5, markersize=6, color='#1f77b4')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Loss vs Iteration')
ax1.grid(True, alpha=0.3)

# Panel 2: Accuracy convergence
ax2.plot(iterations, np.array(training_accs) * 100, 's-', linewidth=2.5, markersize=6, color='#2ca02c')
ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Random guess')
ax2.axhline(y=test_accuracy * 100, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Test accuracy')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Training Accuracy (%)', fontsize=12)
ax2.set_title('Accuracy vs Iteration')
ax2.set_ylim([0, 105])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('qc_training_convergence.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qc_training_convergence.png")


# Figure 3: Quantum vs Classical Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Quantum vs Classical Classification", fontsize=14, fontweight='bold')

# Panel 1: Training accuracy comparison
ax = axes[0, 0]
classifiers = ['Quantum', 'Classical (SVM)']
train_accs = [train_acc_quantum * 100, train_acc_svm * 100]
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(classifiers, train_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Training Accuracy')
ax.set_ylim([0, 110])
for bar, val in zip(bars, train_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 2: Test accuracy comparison
ax = axes[0, 1]
test_accs_comp = [test_accuracy * 100, test_acc_svm * 100]
bars = ax.bar(classifiers, test_accs_comp, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Test Accuracy')
ax.set_ylim([0, 110])
for bar, val in zip(bars, test_accs_comp):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 3: Pros/Cons
ax = axes[1, 0]
ax.axis('off')
pros_cons_text = """
QUANTUM CLASSIFIER
Pros:
  • Novel approach
  • Quantum effects
  • Hybrid potential

Cons:
  • Slow training
  • Noise sensitive
  • Hard to scale

CLASSICAL SVM
Pros:
  • Proven, fast
  • Robust
  • Scalable

Cons:
  • Limited novelty
  • Fixed architecture
  • No quantum effects
"""
ax.text(0.5, 0.5, pros_cons_text, ha='center', va='center', fontsize=10, family='monospace',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 4: Key metrics
ax = axes[1, 1]
ax.axis('off')
metrics_text = f"""
PERFORMANCE SUMMARY

Dataset: {len(X_train)} training, {len(X_test)} test samples
Features: 2D, Binary classification

Quantum:
  Training acc: {train_acc_quantum*100:.1f}%
  Test acc: {test_accuracy*100:.1f}%
  Quantum edge: {(test_accuracy-test_acc_svm)*100:+.1f}%

Classical (SVM):
  Training acc: {train_acc_svm*100:.1f}%
  Test acc: {test_acc_svm*100:.1f}%

Conclusion:
  Current quantum ML: comparable to classical
  Advantage: Hybrid, proof-of-concept
  Potential: Future with better hardware
"""
ax.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10, family='monospace',
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('qc_quantum_vs_classical.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qc_quantum_vs_classical.png")


# Figure 4: Feature Map Types
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Quantum Feature Encoding Methods", fontsize=14, fontweight='bold')

encoding_info = [
    {
        'title': 'Angle Encoding',
        'circuit': 'RY(2·arcsin(x₁))→RY(2·arcsin(x₂))',
        'expressibility': 'Low',
        'complexity': 'Simple (O(n))',
        'best_for': 'Small datasets'
    },
    {
        'title': 'Amplitude Encoding',
        'circuit': '|ψ⟩ = Σxᵢ|i⟩ / ||x||',
        'expressibility': 'High (exponential)',
        'complexity': 'Hard',
        'best_for': 'Dense data'
    },
    {
        'title': 'IQP Encoding',
        'circuit': 'Entangled RZ + CZ structure',
        'expressibility': 'High (tunable)',
        'complexity': 'Medium (O(n²))',
        'best_for': 'Quantum advantage'
    },
    {
        'title': 'Basis Encoding',
        'circuit': 'Binary: |x₁⟩|x₂⟩...',
        'expressibility': 'Low',
        'complexity': 'None (direct)',
        'best_for': 'Binary data'
    }
]

for idx, enc in enumerate(encoding_info):
    ax = axes[idx // 2, idx % 2]
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, enc['title'], ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes)

    # Content
    y_pos = 0.8
    ax.text(0.05, y_pos, f"Circuit: {enc['circuit']}", fontsize=10, transform=ax.transAxes, family='monospace')
    y_pos -= 0.15
    ax.text(0.05, y_pos, f"Expressibility: {enc['expressibility']}", fontsize=10, transform=ax.transAxes)
    y_pos -= 0.15
    ax.text(0.05, y_pos, f"Complexity: {enc['complexity']}", fontsize=10, transform=ax.transAxes)
    y_pos -= 0.15
    ax.text(0.05, y_pos, f"Best for: {enc['best_for']}", fontsize=10, transform=ax.transAxes, fontweight='bold')

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                              fill=False, edgecolor='black', linewidth=2))

plt.tight_layout()
plt.savefig('qc_feature_encodings.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qc_feature_encodings.png")


# Figure 5: Quantum ML Landscape
fig, ax = plt.subplots(figsize=(14, 8))

# Timeline and status
years = np.array([2020, 2022, 2024, 2026, 2028, 2030, 2035])
qaoa_readiness = [3, 4, 5, 6, 7, 8, 9]  # 1-10 scale
vqc_readiness = [2, 3, 4, 5, 6, 7, 8]
qkm_readiness = [1, 2, 3, 4, 6, 7, 8]
qnn_readiness = [1, 2, 3, 5, 7, 8, 9]
classical_ml = [9.5] * len(years)  # Constant high performance

ax.plot(years, qaoa_readiness, 'o-', linewidth=2.5, markersize=10, label='QAOA', color='#1f77b4')
ax.plot(years, vqc_readiness, 's-', linewidth=2.5, markersize=10, label='VQC', color='#ff7f0e')
ax.plot(years, qkm_readiness, '^-', linewidth=2.5, markersize=10, label='Quantum Kernels', color='#2ca02c')
ax.plot(years, qnn_readiness, 'd-', linewidth=2.5, markersize=10, label='QNN', color='#d62728')
ax.plot(years, classical_ml, '--', linewidth=3, markersize=0, label='Classical ML', color='gray', alpha=0.5)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Readiness Level (1-10)', fontsize=12)
ax.set_title('Quantum ML Readiness Timeline', fontsize=14, fontweight='bold')
ax.set_ylim([0, 10.5])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')

# Add regions
ax.fill_between([2020, 2024], 0, 10, alpha=0.1, color='red', label='Research phase')
ax.fill_between([2024, 2030], 0, 10, alpha=0.1, color='yellow')
ax.fill_between([2030, 2035], 0, 10, alpha=0.1, color='green', label='Expected advantage')

ax.text(2022, 9.5, 'Research', fontsize=10, ha='center', fontweight='bold')
ax.text(2027, 9.5, 'Hybrid Era', fontsize=10, ha='center', fontweight='bold')
ax.text(2032, 9.5, 'Advantage?', fontsize=10, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('qc_ml_readiness.png', dpi=150, bbox_inches='tight')
print("✓ Saved: qc_ml_readiness.png")


print("\n" + "=" * 70)
print("QUANTUM CLASSIFICATION EXPLORATION COMPLETE!")
print("=" * 70)

print("\nKey Insights:")
print("✓ Quantum classifiers work (demonstrated on 2D dataset)")
print("✓ Training is slow (many circuit evaluations needed)")
print("✓ Classical ML still competitive (decades of R&D)")
print("✓ Hybrid approaches promising (quantum kernel + classical SVM)")
print("✓ Quantum advantage unclear for most practical problems")

print("\nCurrent State (2024):")
print("  ✓ Proof-of-concept demonstrations working")
print("  ✓ Hybrid classical-quantum approaches emerging")
print("  ✓ Small datasets (toy problems) viable")
print("  ✗ No clear quantum advantage yet")
print("  ✗ Scaling remains challenging")

print("\nFuture Potential:")
print("  2024-2026: Better algorithms, error mitigation")
print("  2026-2028: Larger experiments, specialized domains")
print("  2028-2030: Potential advantage for specific problems")
print("  2030+: Fault-tolerant QC, revolutionary algorithms")

print("\nRealistic Timeline for Quantum ML Advantage:")
print("  • Optimization: Maybe 2025-2026")
print("  • Simulation: Maybe 2026-2027")
print("  • Machine Learning: Maybe 2027-2030")
print("  • Drug Discovery: Maybe 2028-2032")

print("\nWhere Quantum ML Might Win:")
print("  • Problems with exponential structure")
print("  • High-dimensional feature spaces")
print("  • Small, labeled datasets (few samples)")
print("  • Kernel method speedups")
print("  • Hybrid classical-quantum workflows")

print("\nWhere Classical ML Still Wins:")
print("  • Large datasets (millions of samples)")
print("  • Complex, unstructured data (images, text)")
print("  • Production systems (proven, optimized)")
print("  • Most current applications")

print("\nYou've Now Explored:")
print("  1. ✓ Quantum fundamentals (superposition, entanglement)")
print("  2. ✓ Quantum algorithms (Grover, Deutsch-Jozsa, Phase Estimation, VQE)")
print("  3. ✓ Practical considerations (noise, errors, mitigation)")
print("  4. ✓ Circuit optimization (gate reduction, compilation)")
print("  5. ✓ **Quantum Machine Learning (classification)**")

print("\nRemaining Frontiers:")
print("  • Shor's Algorithm (factoring → cryptography)")
print("  • Quantum Simulation (chemistry, materials)")
print("  • Quantum Optimization (QAOA, etc.)")
print("  • Quantum Error Correction (QECC)")
print("  • Fault-Tolerant Quantum Computing")
print()
