# trainer_utils.py

import numpy as np
from typing import List, Dict, Optional, Tuple

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit_algorithms.optimizers import SPSA, ADAM
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

import config
from utils.qiskit_utils import (
    get_prob_of_true_labels_batch,
    build_parameter_bindings,
    calculate_custom_single_qubit_observables,
    gradient_of_observable_parameter_shift,
    calculate_circuit_accuracy,
    calculate_circuit_cross_entropy
)

def train_for_cross_entropy(
    qc_template: "QuantumCircuit",
    feature_params: List["Parameter"],
    var_params: List["Parameter"],
    x_train: np.ndarray,
    y_train_01: np.ndarray,
    shots: int,
    spsa_iters: int,
    noise_model: Optional["NoiseModel"] = None,
    initial_weights_for_optimizer: Optional[List[float]] = None
) -> Tuple[List[float], List[float]]:
    """
    Trains a quantum circuit's variational parameters to minimize Cross-Entropy Loss
    using a manual SPSA implementation that is consistent with the main trainer's config.
    """
    # --- MODIFICATION: SPSA hyperparameters are now pulled from the config file ---
    a = config.SPSA_LEARNING_RATE
    c = config.SPSA_PERTURBATION
    A = 100  # A standard value for stabilization, can also be moved to config if needed
    alpha = 0.602
    gamma = 0.101
    # ----------------------------------------------------------------------------

    num_weights = len(var_params)
    
    if initial_weights_for_optimizer is not None and len(initial_weights_for_optimizer) == num_weights:
        weights = np.array(initial_weights_for_optimizer)
        print("  ... Starting optimization from pre-trained weights (Warm Start).")
    else:
        weights = np.random.rand(num_weights) * 2 * np.pi
        if initial_weights_for_optimizer is not None:
             print("  ... WARNING: Pre-trained weights have incorrect size. Starting with random weights.")

    cost_history = []

    for k in range(spsa_iters):
        a_k = a / (k + 1 + A)**alpha
        c_k = c / (k + 1)**gamma
        delta_k = 2 * np.random.randint(0, 2, size=num_weights) - 1

        weights_plus = weights + c_k * delta_k
        weights_minus = weights - c_k * delta_k

        cost_plus = calculate_circuit_cross_entropy(
            qc_template, weights_plus, feature_params, var_params,
            x_train, y_train_01, shots, noise_model
        )
        cost_minus = calculate_circuit_cross_entropy(
            qc_template, weights_minus, feature_params, var_params,
            x_train, y_train_01, shots, noise_model
        )

        g_k = (cost_plus - cost_minus) / (2 * c_k * delta_k)
        weights = weights - a_k * g_k
        
        current_cost = calculate_circuit_cross_entropy(
            qc_template, weights, feature_params, var_params,
            x_train, y_train_01, shots, noise_model
        )
        cost_history.append(current_cost)
        
        if (k + 1) % 10 == 0:
            print(f"    ... SPSA Iteration {k+1}/{spsa_iters}, Current Loss: {current_cost:.4f}")

    return weights.tolist(), cost_history


def train_and_evaluate_circuit(
    qc_to_evaluate: QuantumCircuit, initial_weights_for_optimizer: List[float],
    feature_params_in_qc: List[Parameter], variational_params_in_qc: List[Parameter],
    x_train: np.ndarray, y_train_01: np.ndarray, y_train_pm1: np.ndarray,
    x_test: np.ndarray, y_test_01: np.ndarray, dataset_name_tag: str,
    optimizer_choice: str, shots: int, spsa_iters: int, adam_iters: int, adam_lr: float
    ) -> Tuple[Optional[List[float]], Optional[float]]:
    
    # Guard clause to handle invalid or untrainable circuits
    if qc_to_evaluate is None or qc_to_evaluate.num_qubits == 0 or not variational_params_in_qc:
        return [], 0.0

    actual_circuit_qiskit_parameters = list(qc_to_evaluate.parameters)
    
    def cost_fn_spsa(current_weights_arr: np.ndarray) -> float:
        cost = calculate_circuit_cross_entropy(
            qc=qc_to_evaluate,
            trained_weights=list(current_weights_arr),
            feature_params=feature_params_in_qc,
            var_params=variational_params_in_qc,
            x_test=x_train,
            y_test_01=y_train_01,
            shots=shots,
            noise_model=None
        )
        return cost if not (np.isnan(cost) or np.isinf(cost)) else 1e6

    def loss_grad_adam_ce(current_weights_arr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculates Cross-Entropy loss and its gradient for the Adam optimizer."""
        avg_loss = calculate_circuit_cross_entropy(
            qc_to_evaluate, list(current_weights_arr), feature_params_in_qc, variational_params_in_qc,
            x_train, y_train_01, shots
        )
        
        num_opt_p = len(current_weights_arr)
        batch_grads = np.zeros((len(x_train), num_opt_p))
        obs_def = (config.CLASSIFICATION_QUBIT_INDEX, config.CLASSIFICATION_OBSERVABLE_TYPE)

        for i, dp_features in enumerate(x_train):
            base_bindings = build_parameter_bindings(
                actual_circuit_qiskit_parameters, feature_params_in_qc, dp_features,
                variational_params_in_qc, list(current_weights_arr))
            
            obs_res = calculate_custom_single_qubit_observables(
                qc_to_evaluate, base_bindings, [obs_def], True, shots
            )
            pred_obs_val = obs_res.get(f"{obs_def[1]}_q{obs_def[0]}", 0.0)
            
            prob_class_0 = (1 + pred_obs_val) / 2.0
            true_label = y_train_01[i]
            
            dL_dp = prob_class_0 - true_label
            
            for k_grad in range(num_opt_p):
                param_to_shift = variational_params_in_qc[k_grad]
                
                dO_d_theta = gradient_of_observable_parameter_shift(
                    qc_to_evaluate, base_bindings.copy(), param_to_shift, obs_def, 
                    shots=shots
                )
                
                dp_dO = 0.5
                grad_comp = dL_dp * dp_dO * (dO_d_theta if not np.isnan(dO_d_theta) else 0.0)
                batch_grads[i, k_grad] = grad_comp

        avg_grads = np.mean(batch_grads, axis=0)
        return avg_loss, avg_grads

    trained_weights_final = np.array(initial_weights_for_optimizer)
    if len(initial_weights_for_optimizer) > 0 and (spsa_iters > 0 or adam_iters > 0):
        if optimizer_choice == "SPSA":
            opt = SPSA(
                maxiter=spsa_iters,
                learning_rate=config.SPSA_LEARNING_RATE,
                perturbation=config.SPSA_PERTURBATION,
                resamplings=config.SPSA_RESAMPLINGS
            )
            res = opt.minimize(fun=cost_fn_spsa, x0=np.array(initial_weights_for_optimizer))
            trained_weights_final = res.x
        elif optimizer_choice == "Adam":
            opt = ADAM(maxiter=adam_iters, lr=adam_lr)
            res = opt.minimize(fun=lambda p: loss_grad_adam_ce(p)[0], x0=np.array(initial_weights_for_optimizer), jac=lambda p: loss_grad_adam_ce(p)[1])
            trained_weights_final = res.x

    final_accuracy = 0.0
    if x_test.size > 0 and y_test_01.size > 0:
         final_accuracy, _ = calculate_circuit_accuracy(
            qc=qc_to_evaluate,
            trained_weights=list(trained_weights_final),
            feature_params=feature_params_in_qc,
            var_params=variational_params_in_qc,
            x_test=x_test,
            y_test_01=y_test_01,
            shots=shots,
            class_idx=config.CLASSIFICATION_QUBIT_INDEX,
            class_obs_type=config.CLASSIFICATION_OBSERVABLE_TYPE,
            label_assignment_method=config.LABEL_ASSIGNMENT_METHOD,
            label_assignment_threshold=config.LABEL_ASSIGNMENT_THRESHOLD,
            class_0_is_positive_z=config.CLASS_0_IS_POSITIVE_Z_EXPECTATION,
            noise_model=None
        )
        
    return list(trained_weights_final), final_accuracy




    def loss_grad_adam_ce(current_weights_arr: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculates Cross-Entropy loss and its gradient for the Adam optimizer."""
        avg_loss = calculate_circuit_cross_entropy(
            qc_to_evaluate, list(current_weights_arr), feature_params_in_qc, variational_params_in_qc,
            x_train, y_train_01, shots
        )
        
        num_opt_p = len(current_weights_arr)
        batch_grads = np.zeros((len(x_train), num_opt_p))
        obs_def = (config.CLASSIFICATION_QUBIT_INDEX, config.CLASSIFICATION_OBSERVABLE_TYPE)

        for i, dp_features in enumerate(x_train):
            base_bindings = build_parameter_bindings(
                actual_circuit_qiskit_parameters, feature_params_in_qc, dp_features,
                variational_params_in_qc, list(current_weights_arr))
            
            obs_res = calculate_custom_single_qubit_observables(
                qc_to_evaluate, base_bindings, [obs_def], True, shots
            )
            pred_obs_val = obs_res.get(f"{obs_def[1]}_q{obs_def[0]}", 0.0)
            
            prob_class_0 = (1 + pred_obs_val) / 2.0
            true_label = y_train_01[i]
            
            dL_dp = prob_class_0 - true_label
            
            for k_grad in range(num_opt_p):
                param_to_shift = variational_params_in_qc[k_grad]
                
                dO_d_theta = gradient_of_observable_parameter_shift(
                    qc_to_evaluate, base_bindings.copy(), param_to_shift, obs_def, 
                    shots=shots
                )
                
                dp_dO = 0.5
                grad_comp = dL_dp * dp_dO * (dO_d_theta if not np.isnan(dO_d_theta) else 0.0)
                batch_grads[i, k_grad] = grad_comp

        avg_grads = np.mean(batch_grads, axis=0)
        return avg_loss, avg_grads

    trained_weights_final = np.array(initial_weights_for_optimizer)
    if len(initial_weights_for_optimizer) > 0 and (spsa_iters > 0 or adam_iters > 0):
        if optimizer_choice == "SPSA":
            opt = SPSA(
                maxiter=spsa_iters,
                learning_rate=config.SPSA_LEARNING_RATE,
                perturbation=config.SPSA_PERTURBATION,
                resamplings=config.SPSA_RESAMPLINGS
            )
            res = opt.minimize(fun=cost_fn_spsa, x0=np.array(initial_weights_for_optimizer))
            trained_weights_final = res.x
        elif optimizer_choice == "Adam":
            opt = ADAM(maxiter=adam_iters, lr=adam_lr)
            res = opt.minimize(fun=lambda p: loss_grad_adam_ce(p)[0], x0=np.array(initial_weights_for_optimizer), jac=lambda p: loss_grad_adam_ce(p)[1])
            trained_weights_final = res.x

    final_accuracy = 0.0
    if x_test.size > 0 and y_test_01.size > 0:
         final_accuracy, _ = calculate_circuit_accuracy(
            qc=qc_to_evaluate,
            trained_weights=list(trained_weights_final),
            feature_params=feature_params_in_qc,
            var_params=variational_params_in_qc,
            x_test=x_test,
            y_test_01=y_test_01,
            shots=shots,
            class_idx=config.CLASSIFICATION_QUBIT_INDEX,
            class_obs_type=config.CLASSIFICATION_OBSERVABLE_TYPE,
            label_assignment_method=config.LABEL_ASSIGNMENT_METHOD,
            label_assignment_threshold=config.LABEL_ASSIGNMENT_THRESHOLD,
            class_0_is_positive_z=config.CLASS_0_IS_POSITIVE_Z_EXPECTATION,
            noise_model=None
        )
        
    return list(trained_weights_final), final_accuracy


def train_for_mse(
    qc_to_optimize: QuantumCircuit,
    feature_params: List[Parameter],
    var_params: List[Parameter],
    x_train: np.ndarray,
    y_train_pm1: np.ndarray,
    shots: int,
    spsa_iters: int = 15,
    noise_model: Optional[NoiseModel] = None
) -> Tuple[List[float], List[float]]: # MODIFICATION: The return type hint is updated to show it returns a tuple.
    """
    Trains the quantum circuit using SPSA to minimize MSE.

    Returns:
        A tuple containing:
        - A list of the final optimized weights.
        - A list of the MSE cost at each SPSA iteration.
    """
    
    # MODIFICATION: This list will store the cost at each iteration.
    cost_history = []

    def mse_cost_fn(weights: np.ndarray) -> float:
        mse = calculate_circuit_mse(
            qc=qc_to_optimize,
            trained_weights=list(weights),
            feature_params=feature_params,
            var_params=var_params,
            x_test=x_train,
            y_test_pm1=y_train_pm1,
            shots=shots,
            class_idx=config.CLASSIFICATION_QUBIT_INDEX,
            class_obs_type=config.CLASSIFICATION_OBSERVABLE_TYPE,
            noise_model=noise_model,
            simulation_method='automatic'
        )
        return mse if not (np.isnan(mse) or np.isinf(mse)) else 4.0
    
    # MODIFICATION: A callback function that SPSA will call at each iteration to record the cost.
    def store_cost_history(nfev, parameters, fun, stepsize, accepted):
        """Saves the cost (fun) at each iteration."""
        cost_history.append(fun)

    if not var_params:
        # MODIFICATION: Return an empty list for weights and history if there's nothing to optimize.
        return [], []

    spsa_optimizer = SPSA(
        maxiter=spsa_iters,
        learning_rate=config.SPSA_LEARNING_RATE,
        perturbation=config.SPSA_PERTURBATION,
        resamplings=config.SPSA_RESAMPLINGS,
        # MODIFICATION: Pass the callback function to the optimizer.
        callback=store_cost_history 
    )
    
    initial_weights = np.random.uniform(0, 2 * np.pi, len(var_params))
    result = spsa_optimizer.minimize(fun=mse_cost_fn, x0=initial_weights)
    
    # MODIFICATION: Return both the final weights and the recorded history.
    return list(result.x), cost_history


def train_for_energy(
    qc_to_optimize: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    spsa_iters: int = 150,
) -> Tuple[List[float], List[float]]:
    """Optimizes weights to minimize the energy expectation of a Hamiltonian."""
    # Import locally to avoid ModuleNotFoundError when running classification
    from utils.quantum_chemistry_utils import calculate_energy_expectation
    
    loss_history = []
    
    def energy_cost_fn(weights: np.ndarray) -> float:
        energy = calculate_energy_expectation(
            circuit=qc_to_optimize,
            hamiltonian=hamiltonian,
            weights=list(weights)
        )
        return energy if not np.isnan(energy) else 0

    def store_intermediate_result(n_evals, params, value, stepsize, accepted):
        loss_history.append(value)

    if not qc_to_optimize.parameters:
        return [], []

    spsa_optimizer = SPSA(
        maxiter=spsa_iters,
        learning_rate=config.SPSA_LEARNING_RATE,
        perturbation=config.SPSA_PERTURBATION,
        callback=store_intermediate_result
    )
    
    initial_weights = np.random.uniform(-np.pi, np.pi, len(qc_to_optimize.parameters))
    result = spsa_optimizer.minimize(fun=energy_cost_fn, x0=initial_weights)
    
    return list(result.x), loss_history


def train_for_log_likelihood(
    qc_to_optimize: QuantumCircuit,
    feature_params: List[Parameter],
    var_params: List[Parameter],
    x_train: np.ndarray,
    y_train_01: np.ndarray,
    shots: int,
    spsa_iters: int = 15,
    noise_model: Optional[NoiseModel] = None
) -> Tuple[List[float], List[float]]:
    
    cost_history = []

    def log_likelihood_cost_fn(weights: np.ndarray) -> float:
        # MODIFICATION IS HERE: Pass the required parameters
        probs_true = get_prob_of_true_labels_batch(
            qc_template=qc_to_optimize,
            feature_params=feature_params,      # ADDED
            var_params=var_params,              # ADDED
            data_points=x_train,
            variational_weights=list(weights),
            true_labels_int=y_train_01,
            shots=shots,
            noise_model=noise_model,
            simulation_method='automatic'
        )
        cost = -np.mean(np.log(probs_true + 1e-9))
        return cost if not (np.isnan(cost) or np.isinf(cost)) else 1e6

    def store_cost_history(nfev, parameters, fun, stepsize, accepted):
        cost_history.append(fun)

    if not var_params:
        return [], []

    # The 'resamplings' parameter from your original code is not a standard
    # part of Qiskit's SPSA and may cause an error. It's removed here for compatibility.
    spsa_optimizer = SPSA(
        maxiter=spsa_iters,
        learning_rate=config.SPSA_LEARNING_RATE,
        perturbation=config.SPSA_PERTURBATION,
        callback=store_cost_history 
    )
    
    initial_weights = np.random.uniform(0, 2 * np.pi, len(var_params))
    result = spsa_optimizer.minimize(fun=log_likelihood_cost_fn, x0=initial_weights)
    
    return list(result.x), cost_history