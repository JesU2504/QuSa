import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple, Union
import sys
import multiprocessing
from functools import partial
import traceback
import argparse
import matplotlib.pyplot as plt
import random
import torch
import math
from qiskit_aer.noise import depolarizing_error, NoiseModel

# --- Setup Project Root Path ---
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

import config
from Architect.architect import ArchitectAgent, create_quantum_circuit_from_tokens
from trainer_utils import train_for_cross_entropy
from utils.data_loader import load_txt_data
from utils.qiskit_utils import (
    calculate_circuit_cross_entropy,
    calculate_circuit_accuracy
)
from qiskit.transpiler import CouplingMap
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_aer.noise import NoiseModel
from sklearn.utils import shuffle

# --- ADD THIS BLOCK FOR REPRODUCIBILITY ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print(f"--- Using fixed random seed: {RANDOM_SEED} for reproducibility ---")
# -----------------------------------------

def plot_cost_history(cost_history: List[float], champion_name: str, save_dir: str):
    if not cost_history: return
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, marker='.', linestyle='-', markersize=5)
    plt.title(f"SPSA Optimization - Cost vs. Iteration\nChampion: {champion_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (Cross-Entropy Loss)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    safe_filename = "".join(c for c in champion_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
    plot_path = os.path.join(save_dir, f"cost_history_{safe_filename}.png")
    plt.savefig(plot_path)
    plt.close()

def get_circuit_params(qc: QuantumCircuit):
    """Helper to correctly extract sorted feature and variational parameters from any circuit."""
    feature_params = sorted([p for p in qc.parameters if p.name.startswith('p')], key=lambda p: int(p.name[1:]))
    var_params = sorted([p for p in qc.parameters if p.name.startswith('w')], key=lambda p: int(p.name[1:]))
    return feature_params, var_params

def process_champion_architecture(
    path: str,
    architect_base_helper: ArchitectAgent,
    x_train_set: np.ndarray,
    y_train_01: np.ndarray,
    x_eval_set: np.ndarray,
    y_eval_raw_set: np.ndarray,
    spsa_iters_final: int,
    max_optimization_retries: int,
    evaluation_indices: List[np.ndarray],
    testing_noise_model: NoiseModel,
    testing_coupling_map: CouplingMap,
    testing_basis_gates: List[str],
    save_dir: str
) -> tuple:
    name = ""
    try:
        if "hea_benchmark" in path:
            name = "Hardware-Efficient_Ansatz"
            print(f"\n--- Evaluating HEA Benchmark ---")
            qc_logical = TwoLocal(num_qubits=config.NUM_QUBITS_CIRCUIT, rotation_blocks='ry', entanglement_blocks='cx', entanglement='linear', reps=2, insert_barriers=True).decompose()
            initial_weights = [random.uniform(0, 2 * math.pi) for _ in qc_logical.parameters]
        else:
            name = os.path.basename(os.path.dirname(path))
            print(f"\n--- Evaluating Evolved Champion: {name} ---")
            with open(path, 'r') as f: champion_data = json.load(f)
            tokens = [architect_base_helper.global_token_pool[i] for i in champion_data['token_ids']]
            qc_logical = create_quantum_circuit_from_tokens(config.NUM_QUBITS_CIRCUIT, tokens, 0)
            initial_weights = champion_data.get('weights', None)

        if not qc_logical: return (name, {})
        
        qc_transpiled = transpile(qc_logical, coupling_map=testing_coupling_map, basis_gates=testing_basis_gates, optimization_level=1, seed_transpiler=RANDOM_SEED)

        trained_weights = initial_weights
        if spsa_iters_final > 0:
            print(f"  > [{name}] Re-training weights for {spsa_iters_final} iterations...")
            # Optional re-training logic would go here
            pass
        if not trained_weights: return (name, {})

        # --- UNIFIED CALIBRATION ---
        print(f"  > [{name}] Performing Unified Calibration for REALISTIC environment...")
        feature_params_realistic, var_params_realistic = get_circuit_params(qc_transpiled)
        acc_def_real, _ = calculate_circuit_accuracy(qc_transpiled, trained_weights, feature_params_realistic, var_params_realistic, x_train_set, y_train_01, config.SHOTS_CONFIG, config.CLASSIFICATION_QUBIT_INDEX, 'Z', 'Z_expectation', 0.0, True, None)
        acc_inv_real, _ = calculate_circuit_accuracy(qc_transpiled, trained_weights, feature_params_realistic, var_params_realistic, x_train_set, y_train_01, config.SHOTS_CONFIG, config.CLASSIFICATION_QUBIT_INDEX, 'Z', 'Z_expectation', 0.0, False, None)

        final_mapping_is_normal = acc_inv_real <= acc_def_real
        print(f"    > Best Realistic Calibration Acc: {max(acc_def_real, acc_inv_real):.2%}. Using this mapping for all evaluations.")

        # --- Evaluations ---
        print(f"  > [{name}] Evaluating final performance...")
        all_results = {"ideal": {"acc": [], "loss": []}, "realistic": {"acc": [], "loss": []}}

        feature_params_ideal, var_params_ideal = get_circuit_params(qc_logical)

        for i, indices in enumerate(evaluation_indices):
            x_subset, y_raw_subset = x_eval_set[indices], y_eval_raw_set[indices]
            y_01_subset = np.argmax(y_raw_subset, axis=1) if y_raw_subset.ndim == 2 else y_raw_subset.astype(int).ravel()
            
            # IDEAL evaluation using the unified calibration
            ideal_acc, ideal_preds = calculate_circuit_accuracy(qc_logical, trained_weights, feature_params_ideal, var_params_ideal, x_subset, y_01_subset, config.SHOTS_CONFIG, config.CLASSIFICATION_QUBIT_INDEX, 'Z', 'Z_expectation', 0.0, final_mapping_is_normal, None)
            if ideal_preds.size > 0 and not np.all(ideal_preds == ideal_preds[0]):
                ideal_loss = calculate_circuit_cross_entropy(qc_logical, trained_weights, feature_params_ideal, var_params_ideal, x_subset, y_01_subset, config.SHOTS_CONFIG, None)
                all_results["ideal"]["acc"].append(ideal_acc)
                all_results["ideal"]["loss"].append(ideal_loss)

            # --- DEBUGGING THE REALISTIC EVALUATION ---
            print(f"\nDEBUG: REALISTIC RUN {i+1}, evaluating on {len(x_subset)} samples...")
            print(f"DEBUG: Using final_mapping_is_normal = {final_mapping_is_normal}")
            
            # Inspect the raw probabilities for the VERY FIRST data point in the subset
            if len(x_subset) > 0:
                first_data_point = x_subset[0]
                from utils.qiskit_utils import get_circuit_probabilities, build_parameter_bindings
                
                bindings = build_parameter_bindings(list(qc_transpiled.parameters), feature_params_realistic, first_data_point, var_params_realistic, trained_weights)
                bound_circuit_for_debug = qc_transpiled.assign_parameters(bindings)
                
                raw_probs = get_circuit_probabilities(bound_circuit_for_debug, noise_model=testing_noise_model, shots=config.SHOTS_CONFIG)
                # print(f"DEBUG: Raw probabilities for the first data point under noise: {raw_probs}")
            # --- END OF DEBUG BLOCK ---

            # REALISTIC evaluation using the unified calibration
            realistic_acc, realistic_preds = calculate_circuit_accuracy(qc_transpiled, trained_weights, feature_params_realistic, var_params_realistic, x_subset, y_01_subset, config.SHOTS_CONFIG, config.CLASSIFICATION_QUBIT_INDEX, 'Z', 'Z_expectation', 0.0, final_mapping_is_normal, testing_noise_model)
            
            # print(f"DEBUG: Predictions array for this run: {realistic_preds}")
            # print(f"DEBUG: Accuracy for this run: {realistic_acc:.2%}")

            if realistic_preds.size > 0 and not np.all(realistic_preds == realistic_preds[0]):
                realistic_loss = calculate_circuit_cross_entropy(qc_transpiled, trained_weights, feature_params_realistic, var_params_realistic, x_subset, y_01_subset, config.SHOTS_CONFIG, testing_noise_model)
                all_results["realistic"]["acc"].append(realistic_acc)
                all_results["realistic"]["loss"].append(realistic_loss)
        
        # --- Compile results ---
        final_result_dict = {}
        for env_key, env_name in [("ideal", "Ideal"), ("realistic", "Realistic")]:
            final_result_dict[f"{env_name}_acc_mean"] = np.mean(all_results[env_key]["acc"]) if all_results[env_key]["acc"] else 0.0
            final_result_dict[f"{env_name}_acc_std"] = np.std(all_results[env_key]["acc"]) if all_results[env_key]["acc"] else 0.0
            final_result_dict[f"{env_name}_loss_mean"] = np.mean(all_results[env_key]["loss"]) if all_results[env_key]["loss"] else float('inf')
        return (name, final_result_dict)

    except Exception as e:
        print(f"ERROR processing {path}: {e}")
        traceback.print_exc()
        return (name, {})

# The __main__ block is correct and does not need changes.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare final champion architectures from co-evolution experiments.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset to use for comparison.")
    args = parser.parse_args()
    CHOSEN_DATASET = args.dataset
    NUM_EVALUATION_RUNS = 5
    spsa_iters_final = 0
    MAX_OPTIMIZATION_RETRIES = 1
    
    base_experiments_dir = os.path.join(PROJECT_ROOT, "experiments", CHOSEN_DATASET)
    comparison_save_dir = os.path.join(PROJECT_ROOT, "comparison_results", CHOSEN_DATASET)
    os.makedirs(comparison_save_dir, exist_ok=True)
    print(f"\nComparison results will be saved to: {comparison_save_dir}")
    
    experiments_to_run_relative = [
        "ideal_no_saboteur/champion_circuit_final.json", 
        "noisy_no_saboteur/champion_circuit_final.json", 
        "ideal_ppo_gnn_saboteur/champion_circuit_final.json",
        "noisy_ppo_gnn_saboteur/champion_circuit_final.json",
        # "hea_benchmark.json"
    ]
    experiments_to_run = [os.path.join(base_experiments_dir, p) for p in experiments_to_run_relative if os.path.exists(os.path.join(base_experiments_dir, p))]
    
    testing_noise_model, testing_coupling_map, testing_basis_gates = None, None, None
    SNAPSHOT_FILENAME = os.path.join(PROJECT_ROOT, 'backend_files', 'backend_snapshot_test.json')
    try:
        with open(SNAPSHOT_FILENAME, 'r') as f: snapshot_data = json.load(f)
        #testing_noise_model = NoiseModel.from_dict(snapshot_data['noise_model'])
        testing_coupling_map = CouplingMap(couplinglist=snapshot_data['coupling_map'])
        testing_basis_gates = snapshot_data['basis_gates']

        testing_noise_model = NoiseModel()
        testing_noise_model.add_all_qubit_quantum_error(depolarizing_error(0.0, 1), ['sx', 'x'])
        testing_noise_model.add_all_qubit_quantum_error(depolarizing_error(0.0, 2), ['cx'])
    except Exception as e:
        print(f"ERROR: Could not load backend snapshot: {e}"); sys.exit(1)
    
    print("Loading dataset...")
    data_path = os.path.join(PROJECT_ROOT, config.DATA_DIR, CHOSEN_DATASET)
    try:
        x_train_set, y_train_raw_set = (load_txt_data(os.path.join(data_path, f)) for f in ["x_train.txt", "y_train.txt"])
        x_eval_set, y_eval_raw_set = (load_txt_data(os.path.join(data_path, f)) for f in ["x_test.txt", "y_test.txt"])
    except FileNotFoundError as e:
        print(f"ERROR: Data files not found in '{data_path}'. Details: {e}"); sys.exit(1)
    
    y_train_01 = np.argmax(y_train_raw_set, axis=1) if y_train_raw_set.ndim == 2 else y_train_raw_set.astype(int).ravel()
    num_features = x_train_set.shape[1]
    architect_base_helper = ArchitectAgent(num_qubits=config.NUM_QUBITS_CIRCUIT, allowed_gates=config.ARCH_GATES_VOCAB, allowed_arguments=[f'p{i}' for i in range(num_features)] + [f'w{i}' for i in range(config.NUM_VARIATIONAL_PLACEHOLDERS)], token_sequence_length=config.TOKEN_SEQUENCE_LENGTH)
    evaluation_indices = [np.random.choice(len(x_eval_set), len(x_eval_set), replace=False) for _ in range(NUM_EVALUATION_RUNS)]
    
    partial_worker = partial(process_champion_architecture, architect_base_helper=architect_base_helper, x_train_set=x_train_set, y_train_01=y_train_01, x_eval_set=x_eval_set, y_eval_raw_set=y_eval_raw_set, spsa_iters_final=spsa_iters_final, max_optimization_retries=MAX_OPTIMIZATION_RETRIES, evaluation_indices=evaluation_indices, testing_noise_model=testing_noise_model, testing_coupling_map=testing_coupling_map, testing_basis_gates=testing_basis_gates, save_dir=comparison_save_dir)
    num_processes = min(os.cpu_count(), len(experiments_to_run))
    
    print(f"\n--- Spawning {num_processes} processes to evaluate champions ---")
    with multiprocessing.Pool(processes=num_processes) as pool:
        eval_results = pool.map(partial_worker, experiments_to_run)
    results = {name: res_dict for name, res_dict in eval_results if name and res_dict}
    print(f"\n\n--- Overall Comparison Summary (Averaged over {NUM_EVALUATION_RUNS} runs) ---")
    headers = ["Champion", "Ideal Acc", "Realistic Acc", "Robustness", "Ideal Perf (-Loss)", "Realistic Perf (-Loss)"]
    col_format = "{:<30} | {:<18} | {:<18} | {:<20} | {:<22} | {:<22}"
    print(col_format.format(*headers))
    print(f"{'-'*30} | {'-'*18} | {'-'*18} | {'-'*20} | {'-'*22} | {'-'*22}")
    results_file_path = os.path.join(comparison_save_dir, "comparison_summary.json")
    with open(results_file_path, 'w') as f: json.dump(results, f, indent=4)
    
    for name, data in sorted(results.items()):
        if not data: continue
        ideal_acc = data.get('Ideal_acc_mean', 0)
        realistic_acc = data.get('Realistic_acc_mean', 0)
        ideal_acc_str = f"{ideal_acc:.2%} (± {data.get('Ideal_acc_std', 0):.2%})"
        realistic_acc_str = f"{realistic_acc:.2%} (± {data.get('Realistic_acc_std', 0):.2%})"
        ideal_perf_str = f"{-data.get('Ideal_loss_mean', float('inf')):.4f}"
        realistic_perf_str = f"{-data.get('Realistic_loss_mean', float('inf')):.4f}"
        robustness_str = f"-{ideal_acc - realistic_acc:.2%}"
        print(col_format.format(name, ideal_acc_str, realistic_acc_str, robustness_str, ideal_perf_str, realistic_perf_str))
    print(f"{'-'*30} | {'-'*18} | {'-'*18} | {'-'*20} | {'-'*22} | {'-'*22}")