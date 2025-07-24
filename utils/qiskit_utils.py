import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter, ParameterExpression
from typing import List, Dict, Optional, Tuple, Union
from sklearn.metrics import f1_score, precision_score, recall_score 

from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    phase_damping_error
)

def get_circuit_probabilities(
    circuit: QuantumCircuit,
    noise_model: Optional[NoiseModel] = None,
    shots: int = 1024,
    optimization_level: int = 1 
) -> dict[str, float]:
    """
    Calculates the outcome probabilities of a quantum circuit.
    - Automatically uses 'density_matrix' simulator if a noise_model is provided
      or if the circuit contains non-unitary noise instructions.
    - Uses 'statevector' simulator for ideal, noiseless circuits.
    """
    if circuit is None or circuit.num_qubits == 0:
        return {"": 1.0}

    # This is the intelligent selection logic. It checks for both types of noise.
    has_noise_in_circuit = any(isinstance(instr.operation, QuantumError) for instr in circuit.data)

    #print('noise_in_circuit:', has_noise_in_circuit)
    
    if noise_model or has_noise_in_circuit:
        simulation_method = 'density_matrix'
    else:
        simulation_method = 'statevector'


    simulator = AerSimulator(method=simulation_method, precision='double')

    # This block handles both statevector and density matrix simulations efficiently.
    if simulation_method in ['statevector', 'density_matrix']:
        temp_circuit = circuit.copy()
        temp_circuit.remove_final_measurements(inplace=True)
        temp_circuit.cregs = []
        
        if simulation_method == 'statevector':
            temp_circuit.save_statevector()
        else:
            temp_circuit.save_density_matrix()

        job = simulator.run(temp_circuit, shots=1, noise_model=noise_model, optimization_level=optimization_level)
        result = job.result()
        
        if not result.success:
            print(f"Warning: {simulation_method} simulation failed. Reason: {result.status}")
            return {}
            
        if simulation_method == 'statevector':
            statevector = result.get_statevector(temp_circuit)
            return statevector.probabilities_dict()
        else:
            density_matrix = result.data(0)['density_matrix']
            return density_matrix.probabilities_dict()

    # This fallback to a shot-based simulator is kept as a safety measure,
    # but should not be needed with the logic above.
    job = simulator.run(circuit, shots=shots, noise_model=noise_model, optimization_level=optimization_level)
    result = job.result()
    counts = result.get_counts(circuit)
    num_classical_bits_in_circuit = circuit.num_clbits
    
    if num_classical_bits_in_circuit == 0:
        return {"": 1.0} if "" in counts and len(counts) == 1 else {}

    all_possible_outcomes = [bin(i)[2:].zfill(num_classical_bits_in_circuit) for i in range(2**num_classical_bits_in_circuit)]
    probabilities = {outcome: 0.0 for outcome in all_possible_outcomes}
    total_shots_observed = sum(counts.values())

    if total_shots_observed > 0:
        for k, v in counts.items():
            formatted_k = k.zfill(num_classical_bits_in_circuit)
            if formatted_k in probabilities:
                probabilities[formatted_k] = v / total_shots_observed
    print(probabilities)
    return probabilities

def get_single_qubit_expectation_from_probs(
    outcome_probabilities: Dict[str, float],
    num_measured_qubits: int,
    target_qubit_index: int, 
    observable_type: str 
) -> float:
    if not outcome_probabilities: return 0.0 
    if not (0 <= target_qubit_index < num_measured_qubits):
        print(f"Error (get_single_qubit_expectation): target_qubit_index {target_qubit_index} out of range for {num_measured_qubits} qubits.")
        return np.nan 
    calculated_value = 0.0
    for bitstring, probability in outcome_probabilities.items():
        if len(bitstring) != num_measured_qubits: continue
        string_char_index = (num_measured_qubits - 1) - target_qubit_index
        if not (0 <= string_char_index < len(bitstring)): continue
        qubit_state_char = bitstring[string_char_index]
        if observable_type == "P1":
            if qubit_state_char == '1': calculated_value += probability
        elif observable_type == "Z":
            if qubit_state_char == '0': calculated_value += probability
            elif qubit_state_char == '1': calculated_value -= probability
        else: raise ValueError(f"Unknown observable_type: {observable_type}.")
    return calculated_value

def calculate_custom_single_qubit_observables(
    circuit_to_run: QuantumCircuit,
    parameter_bindings: Optional[Dict[Union[Parameter, ParameterExpression], float]] = None, 
    observables_to_calculate: Optional[List[Tuple[int, str]]] = None, 
    measure_all_qubits: bool = True, 
    shots: int = 1024,
    noise_model: Optional[NoiseModel] = None,
    _pre_bound_circuit: Optional[QuantumCircuit] = None 
) -> Dict[str, float]:
    if observables_to_calculate is None: observables_to_calculate = []
    if not observables_to_calculate: return {}

    if circuit_to_run is None or circuit_to_run.num_qubits == 0:
        results = {}
        for target_idx, obs_type in observables_to_calculate:
            results[f"{obs_type}_q{target_idx}"] = np.nan 
        return results

    if _pre_bound_circuit is not None:
        working_circuit = _pre_bound_circuit 
    else:
        working_circuit = circuit_to_run.copy()
        if parameter_bindings:
            if working_circuit.parameters:
                try:
                    working_circuit = working_circuit.assign_parameters(parameter_bindings, inplace=False)
                except Exception as e:
                    print(f"Error (calculate_custom): Assigning parameters: {e}")
                    return {f"Error_binding": np.nan}

    outcome_probs = get_circuit_probabilities(
        working_circuit, noise_model=noise_model, shots=shots
    )
    
    num_bits_in_outcome = working_circuit.num_qubits

    results = {}
    if not outcome_probs:
        for target_idx, obs_type in observables_to_calculate:
            results[f"{obs_type}_q{target_idx}"] = np.nan 
        return results
    
    for target_idx, obs_type in observables_to_calculate:
        value = get_single_qubit_expectation_from_probs(
            outcome_probs, num_bits_in_outcome, target_idx, obs_type
        )
        results[f"{obs_type}_q{target_idx}"] = value
    return results

def assign_binary_label_from_observables(
    observable_results: Dict[str, float],
    target_qubit_index: int = 0,
    method: str = "Z_expectation", 
    threshold: Optional[float] = None,
    class_0_is_positive_Z: bool = True
) -> Optional[int]:

    if method == "Z_expectation":
        obs_key = f"Z_q{target_qubit_index}"
        current_threshold = 0.0 if threshold is None else threshold
        value = observable_results.get(obs_key)
        
        # --- ADD THIS DEBUG LINE ---
        #print(f"DEBUG (assign_binary_label): Observable value for Z_q{target_qubit_index} is {value}")
        # -------------------------

        if value is None or np.isnan(value): return None
        return 0 if value > current_threshold else 1 if class_0_is_positive_Z else (1 if value > current_threshold else 0)

    elif method == "P1_probability":
        obs_key = f"P1_q{target_qubit_index}"
        current_threshold = 0.5 if threshold is None else threshold
        value = observable_results.get(obs_key)
        if value is None or np.isnan(value): return None
        return 1 if value > current_threshold else 0
    else: raise ValueError(f"Unsupported method: {method}.")

def _get_shifted_parameter_bindings(
    base_parameter_bindings: Dict[Union[Parameter, ParameterExpression], float],
    param_to_shift: Union[Parameter, ParameterExpression], 
    shift_value: float
) -> Dict[Union[Parameter, ParameterExpression], float]:
    shifted_bindings = base_parameter_bindings.copy()
    if param_to_shift in shifted_bindings:
        shifted_bindings[param_to_shift] = shifted_bindings.get(param_to_shift, 0.0) + shift_value
    else:
        if isinstance(param_to_shift, Parameter):
             shifted_bindings[param_to_shift] = shift_value
    return shifted_bindings

def gradient_of_observable_parameter_shift(
    qc_template: QuantumCircuit,
    base_parameter_bindings: Dict[Union[Parameter, ParameterExpression], float], 
    variational_param_to_shift: Union[Parameter, ParameterExpression], 
    observable_def: Tuple[int, str], 
    shift_val: float = np.pi / 2, 
    shots: int = 1024,
    noise_model: Optional[NoiseModel] = None
) -> float:
    params_plus_shift = _get_shifted_parameter_bindings(base_parameter_bindings, variational_param_to_shift, shift_val)
    obs_results_plus = calculate_custom_single_qubit_observables(
        qc_template, params_plus_shift, [observable_def], True, shots, noise_model
    )
    params_minus_shift = _get_shifted_parameter_bindings(base_parameter_bindings, variational_param_to_shift, -shift_val)
    obs_results_minus = calculate_custom_single_qubit_observables(
        qc_template, params_minus_shift, [observable_def], True, shots, noise_model
    )
    obs_key = f"{observable_def[1]}_q{observable_def[0]}"
    val_plus = obs_results_plus.get(obs_key, np.nan)
    val_minus = obs_results_minus.get(obs_key, np.nan)
    if np.isnan(val_plus) or np.isnan(val_minus):
        return np.nan
    gradient = 0.5 * (val_plus - val_minus) 
    return gradient

def get_prob_of_true_labels_batch(
    qc_template: QuantumCircuit,
    feature_params: List[Parameter],
    var_params: List[Parameter],
    data_points: np.ndarray,
    variational_weights: List[float],
    true_labels_int: np.ndarray,
    noise_model: Optional[NoiseModel] = None,
    shots: int = 1024
) -> np.ndarray:
    probs_of_true_label_list = []
    actual_circuit_params = list(qc_template.parameters)

    for i, data_point_features in enumerate(data_points):
        parameter_bindings = build_parameter_bindings(
            actual_circuit_params,
            feature_params,
            data_point_features,
            var_params,
            variational_weights
        )
        
        bound_circuit = qc_template.assign_parameters(parameter_bindings, inplace=False)
        outcome_probabilities = get_circuit_probabilities(bound_circuit, noise_model=noise_model, shots=shots)

        true_label_for_sample = true_labels_int[i]
        prob_of_true = 0.0

        if not outcome_probabilities: 
            probs_of_true_label_list.append(0.0)
            continue
        
        num_outcome_bits = bound_circuit.num_qubits

        if num_outcome_bits == 1:
            target_bitstring = str(true_label_for_sample)
            prob_of_true = outcome_probabilities.get(target_bitstring.zfill(num_outcome_bits), 0.0)
        elif num_outcome_bits > 1:
            desired_q0_state = str(true_label_for_sample)
            for bitstr, prob in outcome_probabilities.items():
                if len(bitstr) != num_outcome_bits: continue
                if bitstr.endswith(desired_q0_state): 
                    prob_of_true += prob
        else: # num_q == 0
            prob_of_true = outcome_probabilities.get("", 0.0) if true_label_for_sample == 0 else 0.0
        
        probs_of_true_label_list.append(float(prob_of_true))

    return np.array(probs_of_true_label_list)

def build_parameter_bindings(
    target_circuit_qiskit_parameters: List[Parameter],
    feature_param_objects_ordered: Optional[List[Parameter]] = None,
    feature_values_ordered: Optional[np.ndarray] = None,
    var_param_objects_ordered: Optional[List[Parameter]] = None,
    var_values_ordered: Optional[List[float]] = None
    ) -> Dict[Parameter, float]:
    bindings: Dict[Parameter, float] = {}
    feature_value_lookup: Dict[Parameter, float] = {}
    if feature_param_objects_ordered and feature_values_ordered is not None:
        for p_obj in feature_param_objects_ordered:
            try:
                param_idx = int(p_obj.name[1:])
                if param_idx < len(feature_values_ordered):
                    feature_value_lookup[p_obj] = feature_values_ordered[param_idx]
            except ValueError: pass
    var_value_lookup: Dict[Parameter, float] = {}
    if var_param_objects_ordered and var_values_ordered is not None:
        for i, p_obj in enumerate(var_param_objects_ordered):
            if i < len(var_values_ordered): var_value_lookup[p_obj] = var_values_ordered[i]
    for p_in_circuit in target_circuit_qiskit_parameters:
        if not isinstance(p_in_circuit, Parameter): continue
        if p_in_circuit in feature_value_lookup: bindings[p_in_circuit] = feature_value_lookup[p_in_circuit]
        elif p_in_circuit in var_value_lookup: bindings[p_in_circuit] = var_value_lookup[p_in_circuit]
    return bindings

def apply_physical_noise(
    original_circuit: QuantumCircuit,
    instruction: Union[Dict, List[Dict]]
) -> Optional[QuantumCircuit]:
    if not instruction:
        return original_circuit

    noisy_circuit = original_circuit.copy()
    instructions_list = [instruction] if isinstance(instruction, dict) else instruction

    for single_instruction in instructions_list:
        try:
            noise_type = single_instruction['type']
            p = single_instruction['probability']
            error: QuantumError

            if 'target_qubit' in single_instruction:
                target_qubit = single_instruction['target_qubit']
                if target_qubit >= noisy_circuit.num_qubits:
                    print(f"Warning (apply_physical_noise): Target qubit {target_qubit} is out of bounds.")
                    continue
                
                if noise_type == 'bit_flip':
                    error = pauli_error([('X', p), ('I', 1 - p)])
                elif noise_type == 'phase_flip':
                    error = pauli_error([('Z', p), ('I', 1 - p)])
                elif noise_type == 'depolarizing':
                    error = depolarizing_error(p, 1)
                elif noise_type == 'amplitude_damping':
                    error = amplitude_damping_error(p)
                elif noise_type == 'phase_damping':
                    error = phase_damping_error(p)
                else:
                    print(f"Warning (apply_physical_noise): Unknown single-qubit noise type '{noise_type}'")
                    continue
                
                noisy_circuit.append(error, [target_qubit])

            elif 'target_qubits' in single_instruction:
                target_qubits = single_instruction['target_qubits']
                if any(q >= noisy_circuit.num_qubits for q in target_qubits):
                    print(f"Warning (apply_physical_noise): Target qubits {target_qubits} are out of bounds.")
                    continue
                
                if noise_type == 'two_qubit_depolarizing':
                    error = depolarizing_error(p, 2)
                else:
                    print(f"Warning (apply_physical_noise): Unknown two-qubit noise type '{noise_type}'")
                    continue

                noisy_circuit.append(error, target_qubits)
            
            else:
                print(f"Warning (apply_physical_noise): Instruction did not contain a valid target. Instruction: {single_instruction}")
                continue
        except (KeyError, Exception) as e:
            print(f"Error applying one instruction in a combo: {e}. Instruction was: {single_instruction}")
            continue 
            
    return noisy_circuit

def calculate_accuracy(predicted_labels: Union[List[int], np.ndarray], true_labels: np.ndarray) -> float:
    if isinstance(predicted_labels, list):
        predicted_labels_arr = np.array(predicted_labels)
    else:
        if not isinstance(predicted_labels, np.ndarray):
            try: 
                predicted_labels_arr = np.array(predicted_labels)
            except: 
                print("Error (calculate_accuracy): predicted_labels could not be converted to a NumPy array.")
                return 0.0
        else: 
            predicted_labels_arr = predicted_labels

    if predicted_labels_arr.size == 0: 
        return 1.0 if true_labels.size == 0 else 0.0
    
    if not isinstance(true_labels, np.ndarray): 
        true_labels_arr = np.array(true_labels)
    else: 
        true_labels_arr = true_labels

    valid_indices = predicted_labels_arr != -1

    if not np.any(valid_indices):
        return 0.0

    valid_preds = predicted_labels_arr[valid_indices]
    valid_true = true_labels_arr[valid_indices]

    if valid_true.size == 0:
        return 0.0

    correct_predictions = np.sum(valid_preds == valid_true)
    accuracy = correct_predictions / valid_true.size
    
    return accuracy

def calculate_circuit_accuracy(
    qc: QuantumCircuit, trained_weights: List[float], feature_params: List[Parameter], var_params: List[Parameter],
    x_test: np.ndarray, y_test_01: np.ndarray, shots: int, class_idx: int, class_obs_type: str,
    label_assignment_method: str, label_assignment_threshold: float, class_0_is_positive_z: bool,
    noise_model: Optional[NoiseModel] = None
) -> Tuple[float, np.ndarray]:
    if qc is None or qc.num_qubits == 0:
        return 0.0, np.full_like(y_test_01, -1, dtype=int)

    if x_test.size == 0 or y_test_01.size == 0: 
        return 0.0, np.array([])
        
    predictions = []
    actual_circuit_params = list(qc.parameters)
    obs_def = (class_idx, class_obs_type)

    for dp_features in x_test:
        bindings = build_parameter_bindings(actual_circuit_params, feature_params, dp_features, var_params, trained_weights)
        obs_results = calculate_custom_single_qubit_observables(qc, bindings, [obs_def], True, shots, noise_model=noise_model)
        pred_label = assign_binary_label_from_observables(obs_results, class_idx, label_assignment_method, label_assignment_threshold, class_0_is_positive_z)
        predictions.append(pred_label if pred_label is not None else -1)
        
    predictions_arr = np.array(predictions, dtype=int)
    accuracy = calculate_accuracy(predictions_arr, y_test_01)
    
    return float(accuracy), predictions_arr

def calculate_circuit_cross_entropy(
    qc: QuantumCircuit,
    trained_weights: List[float],
    feature_params: List[Parameter],
    var_params: List[Parameter],
    x_test: np.ndarray,
    y_test_01: np.ndarray,
    shots: int,
    noise_model: Optional[NoiseModel] = None
) -> float:
    if qc is None or qc.num_qubits == 0:
        print(qc)
        return float('inf')

    if x_test.size == 0 or y_test_01.size == 0:
        return float('inf')

    probs_of_true_label = get_prob_of_true_labels_batch(
        qc, feature_params, var_params, x_test, trained_weights, y_test_01,
        noise_model, shots
    )

    epsilon = 1e-9
    clipped_probs = np.clip(probs_of_true_label, epsilon, 1. - epsilon)
    cross_entropy_losses = -np.log(clipped_probs)
    avg_loss = np.mean(cross_entropy_losses)

    # if np.any(probs_of_true_label == 0):
    #     print(f"  [DEBUG] Warning: 0-probability prediction detected. Loss might be high.")

    
    return float(avg_loss) if not np.isnan(avg_loss) else float('inf')
