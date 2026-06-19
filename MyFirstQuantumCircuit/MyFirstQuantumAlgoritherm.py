

# Core Qiskit imports
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# IBM Runtime specific imports
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options, Estimator, SamplerV2 as Sampler

import sys
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
import matplotlib.pyplot as plt


print("Python:", sys.version.split()[0])
print("Qiskit:", qiskit.__version__)
print("Qiskit Aer:", qiskit_aer.__version__)
print("Qiskit IBM Runtime:", qiskit_ibm_runtime.__version__)




from qiskit.result.utils import Result
def run_circuit_and_get_counts(circuit, backend, shots=1000):
  """
  Runs a quantum circuit on a specified backend and returns the measurement counts.

  Args:
    Circuit (QuantiumCircuit): The quantum circuit to run.
    backend (AerBackend): The backend to run the circuit on.
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


QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='*******************************************', overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel='ibm_quantum_platform')

# Load saved credentials
service = QiskitRuntimeService()

# Use the least busy backend, or uncomment the loading of a specific backend like "ibm_fez"
# backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
# backend = service.get_backend("ibm_fez")
# backend = service.get_backend("ibm_fez")
# backend = service.get_backend("ibm_cairo")
backend = AerSimulator()

print(backend.name)



circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

counts = run_circuit_and_get_counts(circuit, backend)
print(counts)
plot_histogram(counts)
plt.show()