# utils/quantum_chemistry_utils.py

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import List

# --- MODIFICATION: Import the noise-aware Estimator from Qiskit Aer ---
from qiskit_aer.primitives import Estimator as AerEstimator

def get_h2_hamiltonian(distance: float) -> SparsePauliOp:
    """
    Generates the qubit Hamiltonian for a Hydrogen (H2) molecule
    at a given interatomic distance.

    Args:
        distance (float): The distance between the two hydrogen atoms in Angstroms.

    Returns:
        SparsePauliOp: The qubit Hamiltonian for the H2 molecule.
    """
    driver = PySCFDriver(atom=f"H 0 0 0; H 0 0 {distance}")
    problem = driver.run()
    
    mapper = JordanWignerMapper()
    qubit_hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
    
    return qubit_hamiltonian

def calculate_energy_expectation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    weights: List[float]
) -> float:
    """
    Calculates the expectation value of a Hamiltonian with respect to a circuit.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit (ansatz).
        hamiltonian (SparsePauliOp): The Hamiltonian to measure.
        weights (List[float]): The parameters to assign to the circuit.
        
    Returns:
        float: The calculated energy.
    """
    # --- FIX: Use the AerEstimator which can handle noise instructions ---
    estimator = AerEstimator()
    
    parameter_values = [weights]
    
    job = estimator.run(
        circuits=[circuit], 
        observables=[hamiltonian], 
        parameter_values=parameter_values
    )
    result = job.result()
    
    energy = result.values[0]
    
    return energy
