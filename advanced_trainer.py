import numpy as np
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_THREADING_LAYER']='GNU'
import json
import argparse
from typing import List, Dict, Optional, Tuple, Any
import random
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='qiskit_ibm_provider.api.session')

import multiprocessing
from functools import partial
import shutil

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit import transpile
from qiskit_aer.noise import NoiseModel

import config
from coevolution_agents import PPO_GNN_SaboteurAgent, DQNSaboteurAgent, GeneticArchitectAgent
from coevolution_utils import apply_sabotage_instruction
from utils.data_loader import load_txt_data
from Architect.architect import ArchitectAgent, create_quantum_circuit_from_tokens
from trainer_utils import train_and_evaluate_circuit
from utils.qiskit_utils import (
    calculate_circuit_cross_entropy,
    apply_physical_noise,
    calculate_circuit_accuracy
)
from sklearn.utils import shuffle

def get_save_directory_name(env, sab_mode, tag):
    """Generates the save directory name based on provided experiment settings."""
    base_name = f"experiments/{config.CHOSEN_DATASET}/{env}_{sab_mode}"
    if tag:
        return f"{base_name}_{tag}"
    return base_name

def plot_performance_trends(performance_history: Dict[str, List[float]], current_generation: int, total_generations: int, save_dir: str):
    """Plots the performance trends of the co-evolutionary run."""
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    history_path = os.path.join(save_dir, "performance_history.json")
    try:
        with open(history_path, 'w') as f: json.dump({k: np.array(v).tolist() for k, v in performance_history.items()}, f, indent=4)
    except Exception as e: print(f"Warning: Could not save performance history: {e}")

    plt.style.use('seaborn-v0_8-darkgrid')
    if not performance_history.get("avg_architect_reward"):
        print("Warning: Performance history is empty. Cannot generate plot.")
        return

    generations = range(1, len(performance_history["avg_architect_reward"]) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    plt.title(f"Co-Evolutionary Arms Race - Up to Gen {current_generation+1}/{total_generations}")

    color1 = 'blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Architect Fitness (-Loss on Validation)', color=color1)
    ax1.plot(generations, performance_history["best_architect_reward"], color=color1, linestyle='--', label="Architect Best Fitness")
    ax1.plot(generations, performance_history["avg_architect_reward"], color=color1, linestyle='-', alpha=0.6, label="Architect Avg Fitness")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    ax2 = ax1.twinx()
    color2 = 'orangered'
    ax2.set_ylabel('Saboteur Avg Fragility (Loss Inc. on Validation)', color=color2)
    ax2.plot(generations, performance_history["avg_saboteur_reward"], color=color2, linestyle='-', label="Saboteur Avg Reward")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(0, color=color2, linestyle=':', linewidth=1)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ga_arms_race_trend_gen{current_generation+1}.png"))
    plt.close(fig)

def get_current_curriculum_stage(generation: int, curriculum: Dict) -> Dict:
    """Determines the current curriculum stage based on the generation number."""
    sorted_stage_names = sorted(curriculum.keys(), key=lambda k: curriculum[k].get('end_gen', float('inf')))
    for stage_name in sorted_stage_names:
        stage = curriculum[stage_name]
        if 'end_gen' not in stage or generation < stage['end_gen']:
            return stage
    return curriculum[sorted_stage_names[-1]]


def evaluate_individual_worker(
    individual_tokens: List,
    sabotage_instruction: Optional[Dict],
    initial_weights: Optional[List[float]],
    x_train_main: np.ndarray, y_train_01_main: np.ndarray, y_train_pm1_main: np.ndarray,
    x_validation: np.ndarray, y_validation_01: np.ndarray, y_validation_pm1: np.ndarray,
    hardware_noise_model: Optional[NoiseModel], coupling_map: Optional[CouplingMap], basis_gates: Optional[List[str]],
    generation_info_str: str,
    num_qubits: int, saboteur_mode: str,
    env_mode: str, shots: int, clean_optimizer: str, spsa_iters: int, adam_iters: int, adam_lr: float,
    class_qubit_idx: int, class_obs_type: str,
    saboteur_active: bool
) -> Tuple:
    """
    Worker that trains a circuit in a fast, ideal (noiseless) environment
    and then evaluates its performance under the specified run conditions (ideal or noisy).
    """
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ['MKL_THREADING_LAYER']='GNU'

    qc_logical = create_quantum_circuit_from_tokens(num_qubits, individual_tokens, 0)
    if not qc_logical or qc_logical.num_qubits == 0:
        return (float('-inf'), 0.0, 0.0, None, [])

    transpiled_qc = transpile(qc_logical, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=1,seed_transpiler=42)
    feature_params = sorted([p for p in transpiled_qc.parameters if p.name.startswith('p')], key=lambda p: int(p.name[1:]))
    var_params = sorted([p for p in transpiled_qc.parameters if p.name.startswith('w')], key=lambda p: int(p.name[1:]))

    weights_for_opt = initial_weights if initial_weights is not None else [random.uniform(0, 2 * math.pi) for _ in var_params]

    trained_weights, _ = train_and_evaluate_circuit(
        qc_to_evaluate=transpiled_qc,
        initial_weights_for_optimizer=weights_for_opt,
        feature_params_in_qc=feature_params,
        variational_params_in_qc=var_params,
        x_train=x_train_main,
        y_train_01=y_train_01_main,
        y_train_pm1=y_train_pm1_main,
        x_test=np.array([]),
        y_test_01=np.array([]),
        dataset_name_tag=generation_info_str,
        optimizer_choice=clean_optimizer,
        shots=shots,
        spsa_iters=spsa_iters,
        adam_iters=adam_iters,
        adam_lr=adam_lr
    )

    circuit_for_evaluation = transpiled_qc
    evaluation_weights = trained_weights
    
    if spsa_iters == 0 and adam_iters == 0:
        if len(var_params) != len(trained_weights):
            print(f"ERROR: Inconsistent state detected. Circuit has {len(var_params)} parameters "
                  f"but was given {len(trained_weights)} weights. Disqualifying individual.")
            return (float('-inf'), 0.0, 0.0, None, [])

        if trained_weights:
            param_binds = {p: w for p, w in zip(var_params, trained_weights)}
            circuit_for_evaluation = transpiled_qc.assign_parameters(param_binds)
        
        evaluation_weights = []

    acc_default, _ = calculate_circuit_accuracy(circuit_for_evaluation, evaluation_weights, feature_params, var_params, x_train_main, y_train_01_main, shots, class_qubit_idx, class_obs_type, config.LABEL_ASSIGNMENT_METHOD, config.LABEL_ASSIGNMENT_THRESHOLD, True, hardware_noise_model)
    acc_inverted, _ = calculate_circuit_accuracy(circuit_for_evaluation, evaluation_weights, feature_params, var_params, x_train_main, y_train_01_main, shots, class_qubit_idx, class_obs_type, config.LABEL_ASSIGNMENT_METHOD, config.LABEL_ASSIGNMENT_THRESHOLD, False, hardware_noise_model)
    final_class_0_is_positive_z = acc_inverted <= acc_default

    _, baseline_predictions = calculate_circuit_accuracy(circuit_for_evaluation, evaluation_weights, feature_params, var_params, x_validation, y_validation_01, shots, class_qubit_idx, class_obs_type, config.LABEL_ASSIGNMENT_METHOD, config.LABEL_ASSIGNMENT_THRESHOLD, final_class_0_is_positive_z, hardware_noise_model)
    is_trivial_solution = baseline_predictions.size > 0 and np.all(baseline_predictions == baseline_predictions[0])

    if is_trivial_solution:
        return (float('-inf'), 0.0, 0.0, None, trained_weights)

    baseline_loss_val = calculate_circuit_cross_entropy(circuit_for_evaluation, evaluation_weights, feature_params, var_params, x_validation, y_validation_01, shots, hardware_noise_model)
    sabotaged_loss_val = baseline_loss_val

    qc_for_final_eval = circuit_for_evaluation
    weights_for_final_circuit = evaluation_weights

    if sabotage_instruction:
        qc_sabotaged = None
        sabotage_created_new_circuit = False # Flag to track circuit type

        if saboteur_mode in ['physical_noise', 'ppo_gnn_saboteur']:
            # This mode MODIFIES the existing circuit, it does not create a new one
            qc_sabotaged = apply_physical_noise(circuit_for_evaluation, sabotage_instruction)
            sabotage_created_new_circuit = False
        elif saboteur_mode == 'dqn_saboteur':
            # This mode CREATES A NEW circuit from tokens
            sabotaged_tokens = apply_sabotage_instruction(individual_tokens, sabotage_instruction)
            logical_sabotaged = create_quantum_circuit_from_tokens(num_qubits, sabotaged_tokens, 0)
            if logical_sabotaged:
                qc_sabotaged = transpile(logical_sabotaged, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=1, seed_transpiler=42)
                sabotage_created_new_circuit = True

        if qc_sabotaged:
            # Determine which weights to use for the sabotaged circuit
            sabotaged_weights = trained_weights if sabotage_created_new_circuit else evaluation_weights
            sabotaged_loss_val = calculate_circuit_cross_entropy(qc_sabotaged, sabotaged_weights, feature_params, var_params, x_validation, y_validation_01, shots, hardware_noise_model)
            
            qc_for_final_eval = qc_sabotaged
            weights_for_final_circuit = sabotaged_weights
        else:
            sabotaged_loss_val = float('inf')

    architect_fitness = -sabotaged_loss_val
    reward_saboteur = (sabotaged_loss_val - baseline_loss_val) if saboteur_active else 0.0

    final_accuracy, _ = calculate_circuit_accuracy(
        qc_for_final_eval,
        weights_for_final_circuit,
        feature_params,
        var_params,
        x_validation,
        y_validation_01,
        shots,
        class_qubit_idx,
        class_obs_type,
        config.LABEL_ASSIGNMENT_METHOD,
        config.LABEL_ASSIGNMENT_THRESHOLD,
        final_class_0_is_positive_z,
        hardware_noise_model
    )

    baseline_accuracy = max(acc_default, acc_inverted)
    raw_state_for_dqn = (individual_tokens, trained_weights, baseline_accuracy)
    return (architect_fitness, reward_saboteur, final_accuracy, raw_state_for_dqn, trained_weights)

def main():
    parser = argparse.ArgumentParser(description="Run co-evolutionary quantum architecture search.")
    parser.add_argument('--environment', type=str, default='ideal', choices=['ideal', 'noisy'])
    parser.add_argument('--saboteur', type=str, default='no_saboteur', choices=['no_saboteur', 'ppo_gnn_saboteur', 'dqn_saboteur'])
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='moons')
    args = parser.parse_args()

    config.EVALUATION_ENVIRONMENT = args.environment
    config.SABOTEUR_MODE = args.saboteur
    config.CHOSEN_DATASET = args.dataset
    config.EXPERIMENT_TAG = args.tag
    config.SABOTEUR_ACTIVE = config.SABOTEUR_MODE != 'no_saboteur'
    config.SAVE_DIRECTORY_NAME = get_save_directory_name(config.EVALUATION_ENVIRONMENT, config.SABOTEUR_MODE, config.EXPERIMENT_TAG)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    save_directory = config.SAVE_DIRECTORY_NAME
    if not os.path.exists(save_directory): os.makedirs(save_directory)

    try:
        config_snapshot_path = os.path.join(save_directory, "config_snapshot.py")
        shutil.copyfile("config.py", config_snapshot_path)
        print(f" > Saved config snapshot to {config_snapshot_path}")
    except Exception as e:
        print(f"Warning: Could not save config snapshot. Error: {e}")

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"--- Using fixed random seed: {RANDOM_SEED} ---")
    print(f"\n--- Starting Co-evolution ---")
    print(f"    - Environment: {config.EVALUATION_ENVIRONMENT}")
    print(f"    - Saboteur Mode: {config.SABOTEUR_MODE}")
    print(f"    - Results will be saved to: '{save_directory}'")

    if config.EVALUATION_ENVIRONMENT == 'noisy':
        SNAPSHOT_FILENAME = 'backend_files/backend_snapshot_A_linear.json'
        try:
            with open(SNAPSHOT_FILENAME, 'r') as f: snapshot_data = json.load(f)
            hardware_noise_model = NoiseModel.from_dict(snapshot_data['noise_model'])
            basis_gates = snapshot_data['basis_gates']
            hw_error_rates = snapshot_data['avg_error_rates']
            coupling_map = CouplingMap(couplinglist=snapshot_data['coupling_map'])
        except Exception as e: print(f"ERROR loading snapshot: {e}"); exit()
    else:
        hardware_noise_model, coupling_map, basis_gates, hw_error_rates = None, None, None, {'single_qubit': 0.001, 'two_qubit': 0.01}

    x_train_all, y_train_raw_all, x_test, y_test_raw = (load_txt_data(os.path.join(config.DATA_DIR, config.CHOSEN_DATASET, f)) for f in ["x_train.txt", "y_train.txt", "x_test.txt", "y_test.txt"])
    x_train_all, y_train_raw_all = shuffle(x_train_all, y_train_raw_all, random_state=RANDOM_SEED)

    x_validation, y_validation_raw = x_test, y_test_raw
    x_train_pool, y_train_pool_raw = x_train_all, y_train_raw_all

    def process_labels(y_raw):
        y_01 = np.argmax(y_raw, axis=1) if y_raw.ndim == 2 else y_raw.astype(int).ravel()
        y_pm1 = np.where(y_01 == 0, 1.0, -1.0)
        return y_01, y_pm1

    y_train_pool_01, y_train_pool_pm1 = process_labels(y_train_pool_raw)
    y_validation_01, y_validation_pm1 = process_labels(y_validation_raw)

    num_features = x_train_all.shape[1]
    architect_base_helper = ArchitectAgent(num_qubits=config.NUM_QUBITS_CIRCUIT, allowed_gates=config.ARCH_GATES_VOCAB, allowed_arguments=[f'p{i}' for i in range(num_features)] + [f'w{i}' for i in range(config.NUM_VARIATIONAL_PLACEHOLDERS)], token_sequence_length=config.TOKEN_SEQUENCE_LENGTH)
    architect_agent = GeneticArchitectAgent(population_size=config.GA_POPULATION_SIZE, token_sequence_length=config.TOKEN_SEQUENCE_LENGTH, num_total_tokens_in_pool=len(architect_base_helper.global_token_pool), elitism_count=config.GA_ELITISM_COUNT, mutation_rate=config.GA_MUTATION_RATE, crossover_rate=config.GA_CROSSOVER_RATE)
    saboteur_agent = None
    if config.SABOTEUR_ACTIVE:
        base_init_args = {'gate_vocab': config.ARCH_GATES_VOCAB, 'num_qubits_circuit': config.NUM_QUBITS_CIRCUIT, 'num_variational_placeholders': config.NUM_VARIATIONAL_PLACEHOLDERS, 'noise_strength_multipliers': config.SABOTEUR_NOISE_STRENGTH_MULTIPLIERS, 'base_noise_strength': config.SABOTEUR_NOISE_STRENGTH, 'hw_error_rates': hw_error_rates}
        device = "cpu"
        if config.SABOTEUR_MODE == 'ppo_gnn_saboteur':
            saboteur_agent = PPO_GNN_SaboteurAgent(**base_init_args, device_str=device)
        elif config.SABOTEUR_MODE == 'dqn_saboteur':
            dqn_args = base_init_args.copy(); dqn_args.update({'token_sequence_length': config.TOKEN_SEQUENCE_LENGTH, 'max_feature_param_idx': num_features - 1, 'max_eligible_targets_to_consider': 10})
            saboteur_agent = DQNSaboteurAgent(**dqn_args, device_str=device)

    checkpoint_path = os.path.join(save_directory, "training_checkpoint.pth")
    start_generation = 0
    performance_history = {"avg_saboteur_reward": [], "best_architect_reward": [], "avg_architect_reward": []}
    best_validation_fitness = -float('inf')
    generations_without_improvement = 0
    all_time_champion_individual = None
    all_time_champion_weights = []
    all_time_best_accuracy = 0.0
    
    # --- ADD THESE LINES ---
    final_gen_champion_tokens = None
    final_gen_champion_weights = []

    if config.RESUME_TRAINING and os.path.exists(checkpoint_path):
        print(f"--- Resuming training from checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        start_generation = checkpoint.get('generation', 0) + 1
        performance_history = checkpoint.get('performance_history', performance_history)
        best_validation_fitness = checkpoint.get('best_validation_fitness', -float('inf'))
        generations_without_improvement = checkpoint.get('generations_without_improvement', 0)
        all_time_champion_individual = checkpoint.get('all_time_champion_individual')
        all_time_champion_weights = checkpoint.get('all_time_champion_weights', [])
        all_time_best_accuracy = checkpoint.get('all_time_best_accuracy', 0.0)
        if saboteur_agent and 'saboteur_agent_state' in checkpoint:
            saboteur_agent.load_state(checkpoint['saboteur_agent_state'])
        if 'architect_agent_state' in checkpoint:
            architect_agent.load_state(checkpoint['architect_agent_state'])

    # This is the main block inside your main() function.

    final_generation = start_generation - 1
    try:
        for generation in range(start_generation, config.GA_NUM_GENERATIONS):
            final_generation = generation
            print(f"\n{'='*20} Generation {generation + 1}/{config.GA_NUM_GENERATIONS} {'='*20}")

            if config.SABOTEUR_ACTIVE and isinstance(saboteur_agent, PPO_GNN_SaboteurAgent):
                current_stage_settings = get_current_curriculum_stage(generation, config.SABOTEUR_CURRICULUM)
                saboteur_agent.update_curriculum(current_stage_settings)

            train_indices = np.random.choice(len(x_train_pool), size=min(len(x_train_pool), getattr(config, 'NUM_TRAIN_SAMPLES', len(x_train_pool))), replace=False)
            x_train_main, y_train_01_main, y_train_pm1_main = x_train_pool[train_indices], y_train_pool_01[train_indices], y_train_pool_pm1[train_indices]

            # 1. Unpack the (tokens, weights) tuples from the weight-aware agent
            population_with_weights = architect_agent.propose_circuits_from_population(architect_base_helper)
            population_tokens = [ind[0] for ind in population_with_weights]
            initial_weights_for_pop = [ind[1] for ind in population_with_weights]

            sabotage_instructions_for_pop = [None] * len(population_tokens)
            if config.SABOTEUR_ACTIVE and saboteur_agent:
                print(" > Saboteur is choosing actions...")
                for i, ind_tokens in enumerate(population_tokens):
                    qc_temp = create_quantum_circuit_from_tokens(config.NUM_QUBITS_CIRCUIT, ind_tokens, 0)
                    if qc_temp:
                        raw_state = (ind_tokens, [], -0.5)
                        if isinstance(saboteur_agent, PPO_GNN_SaboteurAgent): sabotage_instructions_for_pop[i] = saboteur_agent.get_action_and_store_in_memory(raw_state)
                        elif isinstance(saboteur_agent, DQNSaboteurAgent): sabotage_instructions_for_pop[i] = saboteur_agent.get_action(raw_state)

            num_processes = min(os.cpu_count(), len(population_tokens))

            partial_worker = partial(evaluate_individual_worker,
                                     x_train_main=x_train_main, y_train_01_main=y_train_01_main, y_train_pm1_main=y_train_pm1_main,
                                     x_validation=x_validation, y_validation_01=y_validation_01, y_validation_pm1=y_validation_pm1,
                                     hardware_noise_model=hardware_noise_model, coupling_map=coupling_map, basis_gates=basis_gates,
                                     generation_info_str=f"G{generation+1}",
                                     num_qubits=config.NUM_QUBITS_CIRCUIT, saboteur_mode=config.SABOTEUR_MODE,
                                     env_mode=config.EVALUATION_ENVIRONMENT, shots=config.SHOTS_CONFIG, clean_optimizer=config.CLEAN_CIRCUIT_OPTIMIZER,
                                     spsa_iters=config.SPSA_MAX_ITERATIONS_CLEAN, adam_iters=config.ADAM_MAX_ITERATIONS_CLEAN, adam_lr=config.ADAM_LEARNING_RATE_CLEAN,
                                     class_qubit_idx=config.CLASSIFICATION_QUBIT_INDEX, class_obs_type=config.CLASSIFICATION_OBSERVABLE_TYPE,
                                     saboteur_active=config.SABOTEUR_ACTIVE)
            
            # 2. Zip the correctly separated arguments for the worker
            eval_args = zip(population_tokens, sabotage_instructions_for_pop, initial_weights_for_pop)
            
            # 3. Use the robust asynchronous pool to prevent deadlocks
            results = []
            with multiprocessing.Pool(processes=num_processes) as pool:
                async_result = pool.starmap_async(partial_worker, eval_args)
                try:
                    results = async_result.get(timeout=600) # 10-minute timeout
                except multiprocessing.TimeoutError:
                    print("ERROR: Generation timed out. A worker likely failed. Skipping generation.")
                    pass # Leaves results list empty

            generation_fitness_scores, generation_fragility_scores, generation_accuracy_scores, generation_weights = [], [], [], []
            for i, result_tuple in enumerate(results):
                architect_fitness, reward_saboteur, accuracy, raw_state_for_dqn, weights = result_tuple
                generation_fitness_scores.append(architect_fitness)
                generation_fragility_scores.append(reward_saboteur)
                generation_accuracy_scores.append(accuracy)
                generation_weights.append(weights)
                if config.SABOTEUR_ACTIVE and saboteur_agent and sabotage_instructions_for_pop[i] is not None:
                    if isinstance(saboteur_agent, PPO_GNN_SaboteurAgent):
                        saboteur_agent.memory.rewards.append(reward_saboteur)
                        saboteur_agent.memory.is_terminals.append(True)
                    elif isinstance(saboteur_agent, DQNSaboteurAgent) and raw_state_for_dqn:
                        saboteur_agent.learn(raw_state_for_dqn, reward_saboteur, None, True)
            
            if config.SABOTEUR_ACTIVE and saboteur_agent:
                if isinstance(saboteur_agent, PPO_GNN_SaboteurAgent) and len(saboteur_agent.memory.actions) > 0:
                    saboteur_agent.update()
                elif isinstance(saboteur_agent, DQNSaboteurAgent) and len(saboteur_agent.memory) >= saboteur_agent.batch_size:
                    saboteur_agent.optimize_model()

            if not generation_fitness_scores: continue
            
            # 4. Pass weights to the updated agent and receive new champion details
            best_tokens, best_weights = architect_agent.update_population(generation_fitness_scores, generation_weights)

            # --- ADD THIS LOGIC ---
            if generation == final_generation:
                final_gen_champion_tokens = best_tokens
                final_gen_champion_weights = best_weights
            
            best_fitness, avg_fitness = np.max(generation_fitness_scores), np.mean(generation_fitness_scores)
            avg_fragility = np.mean([r for i, r in enumerate(generation_fragility_scores) if sabotage_instructions_for_pop[i] is not None] or [0])
            best_accuracy = np.max(generation_accuracy_scores) if generation_accuracy_scores else 0.0
            
            performance_history["best_architect_reward"].append(best_fitness)
            performance_history["avg_architect_reward"].append(avg_fitness)
            performance_history["avg_saboteur_reward"].append(avg_fragility)
            
            print(f"\nGen {generation + 1} Summary: Arch Fitness Best={best_fitness:.4f}, Avg={avg_fitness:.4f} | Arch Acc Best={best_accuracy:.2%} | Sabo Reward Avg={avg_fragility:.4f}")
            
            if best_accuracy > all_time_best_accuracy:
                all_time_best_accuracy = best_accuracy
            
            if best_fitness > best_validation_fitness:
                best_validation_fitness = best_fitness
                generations_without_improvement = 0
                # 5. Store both the tokens and weights of the new champion
                all_time_champion_individual = best_tokens
                all_time_champion_weights = best_weights
            else:
                generations_without_improvement += 1
            
            if (generation + 1) % config.GA_CHECKPOINT_FREQUENCY == 0 or (generation + 1) == config.GA_NUM_GENERATIONS:
                checkpoint_data = {'generation': generation, 'performance_history': performance_history, 'best_validation_fitness': best_validation_fitness, 'generations_without_improvement': generations_without_improvement, 'all_time_champion_individual': all_time_champion_individual, 'all_time_champion_weights': all_time_champion_weights, 'architect_agent_state': architect_agent.save_state(), 'all_time_best_accuracy': all_time_best_accuracy}
                if saboteur_agent: checkpoint_data['saboteur_agent_state'] = saboteur_agent.save_state()
                torch.save(checkpoint_data, checkpoint_path)
            
            if generations_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n--- EARLY STOPPING: No improvement for {config.EARLY_STOPPING_PATIENCE} generations. ---")
                break
    finally:
        print(f"\n--- Training Run Finished at Generation {final_generation + 1} ---")
        print(f"🏆 All-Time Best Validation Accuracy: {all_time_best_accuracy:.2%}")

        if all_time_champion_individual:
            print(f"\n{'='*20} Final Championship Match {'='*20}")

            contender_A_tokens_ids = all_time_champion_individual
            contender_A_tokens = [architect_base_helper.global_token_pool[i] for i in contender_A_tokens_ids]
            contender_A_weights = all_time_champion_weights

            # 6. Correctly select the Final Gen Champion to prevent mismatches
            contender_B_tokens_ids = final_gen_champion_tokens
            contender_B_weights = final_gen_champion_weights
            contender_B_tokens = [architect_base_helper.global_token_pool[i] for i in contender_B_tokens_ids]

            finalists_tokens = [contender_A_tokens, contender_B_tokens]
            finalists_weights = [contender_A_weights, contender_B_weights]
            finalist_names = ["All-Time Champion", "Final Gen Champion"]
            
            if saboteur_agent and isinstance(saboteur_agent, PPO_GNN_SaboteurAgent):
                saboteur_agent.update_curriculum(config.SABOTEUR_CURRICULUM['stage_3'])
                print(" > Saboteur is set to maximum difficulty for the championship.")

            final_sabotage = [None, None]
            if saboteur_agent:
                for i, tokens in enumerate(finalists_tokens):
                    raw_state = (tokens, [], -0.5)
                    if isinstance(saboteur_agent, DQNSaboteurAgent): final_sabotage[i] = saboteur_agent.get_action(raw_state)
                    elif isinstance(saboteur_agent, PPO_GNN_SaboteurAgent): final_sabotage[i] = saboteur_agent.get_action_and_store_in_memory(raw_state)
            
            match_x_train, match_y_01, match_y_pm1 = x_train_pool, y_train_pool_01, y_train_pool_pm1
            match_x_val, match_y_val_01, match_y_val_pm1 = x_validation, y_validation_01, y_validation_pm1
            
            match_partial_worker = partial(evaluate_individual_worker,
                                           x_train_main=match_x_train, y_train_01_main=match_y_01, y_train_pm1_main=match_y_pm1,
                                           x_validation=match_x_val, y_validation_01=match_y_val_01, y_validation_pm1=match_y_val_pm1,
                                           hardware_noise_model=hardware_noise_model, coupling_map=coupling_map, basis_gates=basis_gates,
                                           generation_info_str="Championship_Match",
                                           num_qubits=config.NUM_QUBITS_CIRCUIT, saboteur_mode=config.SABOTEUR_MODE,
                                           env_mode=config.EVALUATION_ENVIRONMENT, shots=config.SHOTS_CONFIG, clean_optimizer=config.CLEAN_CIRCUIT_OPTIMIZER,
                                           spsa_iters=0, adam_iters=0, adam_lr=0.0,
                                           class_qubit_idx=config.CLASSIFICATION_QUBIT_INDEX, class_obs_type=config.CLASSIFICATION_OBSERVABLE_TYPE,
                                           saboteur_active=config.SABOTEUR_ACTIVE)
            
            match_eval_args = zip(finalists_tokens, final_sabotage, finalists_weights)
            
            # Use robust pool for championship match as well
            match_results = []
            with multiprocessing.Pool(processes=2) as pool:
                async_result = pool.starmap_async(match_partial_worker, match_eval_args)
                try:
                    match_results = async_result.get(timeout=300)
                except multiprocessing.TimeoutError:
                    print("ERROR: Championship match timed out.")
                    match_results = [(-float('inf'), 0, 0, None, []), (-float('inf'), 0, 0, None, [])]
            
            if len(match_results) < 2:
                while len(match_results) < 2: match_results.append((-float('inf'), 0, 0, None, []))

            all_time_champ_final_fitness = float(match_results[0][0])
            final_gen_champ_final_fitness = float(match_results[1][0])
            
            print(f" > All-Time Champion Final Fitness: {all_time_champ_final_fitness:.4f}")
            print(f" > Final Gen Champion Final Fitness: {final_gen_champ_final_fitness:.4f}")

            if final_gen_champ_final_fitness > all_time_champ_final_fitness:
                print("\n🏆 The Final Generation's Champion wins the Championship Match!")
                true_champion_individual = contender_B_tokens_ids
                true_champion_weights = contender_B_weights
            else:
                print("\n🏆 The All-Time Champion successfully defends its title!")
                true_champion_individual = contender_A_tokens_ids
                true_champion_weights = contender_A_weights
            
            plot_performance_trends(performance_history, final_generation, config.GA_NUM_GENERATIONS, save_directory)
            final_champion_path = os.path.join(save_directory, "champion_circuit_final.json")
            champion_save_data = {"token_ids": [int(i) for i in true_champion_individual], "weights": true_champion_weights}
            with open(final_champion_path, 'w') as f: json.dump(champion_save_data, f, indent=4)
            if saboteur_agent: torch.save(saboteur_agent.save_state(), os.path.join(save_directory, "saboteur_model_final.pth"))
            print("Final artifacts saved.")
        else:
            print("No champion individual was found to save.")
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()