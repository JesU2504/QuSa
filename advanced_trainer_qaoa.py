# advanced_trainer_qaoa.py

import numpy as np
import os
import json
from typing import List, Dict, Tuple
import random
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='qiskit_ibm_provider.api.session')

import multiprocessing
from functools import partial

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA

import config
from coevolution_agents import PPO_GNN_SaboteurAgent, GeneticArchitectAgent
from Architect.architect import ArchitectAgent, create_quantum_circuit_from_tokens, QuantumToken
# Note: This utility is a general expectation value calculator, suitable for both VQE and QAOA
from utils.quantum_chemistry_utils import calculate_energy_expectation
from utils.qiskit_utils import apply_physical_noise

# --- New Function to train for MAXIMIZING expectation value ---
def train_for_max_expectation(
    qc_to_optimize: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    spsa_iters: int = 150,
) -> Tuple[List[float], List[float]]:
    """Optimizes weights to MAXIMIZE the expectation value of a Hamiltonian."""
    history = []
    
    # SPSA is a minimizer, so we minimize the *negative* of the expectation value.
    def cost_fn(weights: np.ndarray) -> float:
        exp_val = calculate_energy_expectation(
            circuit=qc_to_optimize,
            hamiltonian=hamiltonian,
            weights=list(weights)
        )
        return -exp_val if not np.isnan(exp_val) else 100.0

    def store_intermediate_result(n_evals, params, value, stepsize, accepted):
        # We store the actual (positive) expectation value in the history
        history.append(-value)

    if not qc_to_optimize.parameters:
        return [], []

    spsa_optimizer = SPSA(maxiter=spsa_iters, callback=store_intermediate_result)
    initial_weights = np.random.uniform(-np.pi, np.pi, len(qc_to_optimize.parameters))
    result = spsa_optimizer.minimize(fun=cost_fn, x0=initial_weights)
    
    return list(result.x), history

# --- New Function to define a Max-Cut Problem Hamiltonian ---
def get_max_cut_hamiltonian(num_qubits: int, edges: List[Tuple[int, int]]) -> SparsePauliOp:
    """Creates the Max-Cut cost Hamiltonian for a given graph."""
    pauli_list = []
    for (i, j) in edges:
        # Build the Z_i Z_j term
        pauli_str = ['I'] * num_qubits
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'
        # The cost function is C(x) = sum(0.5 * (1 - z_i z_j))
        # This translates to H = sum(0.5 * (I - Z_i Z_j))
        pauli_list.append(("".join(pauli_str), -0.5))
    
    # Add the constant term from the sum of 0.5 * I
    const_pauli_str = 'I' * num_qubits
    pauli_list.append((const_pauli_str, 0.5 * len(edges)))
    
    return SparsePauliOp.from_list(pauli_list)


def plot_performance_trends(performance_history: Dict[str, List[float]], current_generation: int, total_generations: int, save_dir: str):
    """Plots the performance history of the QAOA training."""
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    history_path = os.path.join(save_dir, "qaoa_performance_history.json")
    try:
        with open(history_path, 'w') as f: json.dump({k: np.array(v).tolist() for k, v in performance_history.items()}, f, indent=4)
    except Exception as e: print(f"Warning: Could not save performance history: {e}")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    if not performance_history.get("avg_architect_fitness"): return

    generations = range(1, len(performance_history["avg_architect_fitness"]) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    plt.title(f"Resource-Aware QAOA Co-Evolution - Up to Gen {current_generation+1}/{total_generations}")
    
    color1 = 'blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Architect Best Fitness (<H> - Penalty)', color=color1)
    ax1.plot(generations, performance_history["best_architect_fitness"], color=color1, linestyle='--', label="Architect Best Fitness")
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'orangered'
    ax2.set_ylabel('Saboteur Avg Reward (<H> Decrease)', color=color2)
    ax2.plot(generations, performance_history["avg_saboteur_reward"], color=color2, linestyle='-', label="Saboteur Avg Reward")
    ax2.tick_params(axis='y', labelcolor=color2)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'green'
    ax3.set_ylabel('Avg CNOT Count', color=color3)
    ax3.plot(generations, performance_history["avg_cnot_count"], color=color3, linestyle=':', label="Avg CNOT Count")
    ax3.tick_params(axis='y', labelcolor=color3)

    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(save_dir, f"qaoa_resource_aware_arms_race_trend_gen{current_generation+1}.png"))
    plt.close(fig)

def train_clean_ansatz_qaoa_worker(
    individual_tokens: List[QuantumToken],
    hamiltonian: SparsePauliOp,
    num_qubits: int
) -> Tuple:
    """Worker to train a clean QAOA ansatz and return its performance and CNOT count."""
    qc_logical = create_quantum_circuit_from_tokens(num_qubits, individual_tokens, 0)
    cnot_count = qc_logical.count_ops().get('cx', 0) if qc_logical else 0

    if not qc_logical or qc_logical.num_qubits == 0 or not qc_logical.parameters:
        return (individual_tokens, [], -100.0, cnot_count) # Return a very low score for invalid circuits

    trained_weights, _ = train_for_max_expectation(qc_logical, hamiltonian, spsa_iters=150)
    baseline_exp_val = calculate_energy_expectation(qc_logical, hamiltonian, trained_weights)
    
    return (individual_tokens, trained_weights, baseline_exp_val, cnot_count)

def evaluate_attacked_ansatz_qaoa_worker(
    individual_tokens: List[QuantumToken],
    trained_weights: List[float],
    sabotage_instruction: Dict,
    hamiltonian: SparsePauliOp,
    num_qubits: int
) -> float:
    """Worker to apply an attack and calculate the new expectation value."""
    qc_logical = create_quantum_circuit_from_tokens(num_qubits, individual_tokens, 0)
    if not qc_logical: return -100.0

    qc_sabotaged = apply_physical_noise(qc_logical, sabotage_instruction)
    
    if qc_sabotaged:
        attacked_exp_val = calculate_energy_expectation(qc_sabotaged, hamiltonian, trained_weights)
    else:
        attacked_exp_val = -100.0
        
    return attacked_exp_val

if __name__ == '__main__':
    # --- Configuration for QAOA Problem and Resource Awareness ---
    QAOA_NUM_QUBITS = 4
    # A simple square graph for Max-Cut
    GRAPH_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
    CNOT_BUDGET = 8
    CNOT_PENALTY_STRENGTH = 0.2
    
    # --- Local flag to control the Saboteur ---
    SABOTEUR_ACTIVE = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    MAX_CUT_HAMILTONIAN = get_max_cut_hamiltonian(QAOA_NUM_QUBITS, GRAPH_EDGES)
    print(f"--- Starting QAOA Co-evolution for Max-Cut (CNOT Budget: {CNOT_BUDGET}) ---")
    
    architect_base_helper = ArchitectAgent(
        num_qubits=QAOA_NUM_QUBITS,
        allowed_gates=config.ARCH_GATES_VOCAB,
        allowed_arguments=[f'w{i}' for i in range(config.NUM_VARIATIONAL_PLACEHOLDERS)],
        token_sequence_length=config.TOKEN_SEQUENCE_LENGTH
    )
    architect_agent = GeneticArchitectAgent(
        population_size=config.GA_POPULATION_SIZE,
        token_sequence_length=config.TOKEN_SEQUENCE_LENGTH,
        num_total_tokens_in_pool=len(architect_base_helper.global_token_pool),
        elitism_count=config.GA_ELITISM_COUNT,
        mutation_rate=config.GA_MUTATION_RATE,
        crossover_rate=config.GA_CROSSOVER_RATE
    )
    
    saboteur_agent = None
    if SABOTEUR_ACTIVE:
        print("Info: Configuring Saboteur for physical noise attacks.")
        config.SABOTEUR_MODE = 'ppo_gnn_saboteur' 
        base_init_args = {
            'gate_vocab': config.ARCH_GATES_VOCAB,
            'num_qubits_circuit': QAOA_NUM_QUBITS,
            'num_variational_placeholders': config.NUM_VARIATIONAL_PLACEHOLDERS,
            'noise_strength_multipliers': config.SABOTEUR_NOISE_STRENGTH_MULTIPLIERS,
            'base_noise_strength': config.SABOTEUR_NOISE_STRENGTH,
            'hw_error_rates': {'single_qubit': 0.001, 'two_qubit': 0.01} 
        }
        saboteur_agent = PPO_GNN_SaboteurAgent(**base_init_args, device_str="cpu")

    # --- MODIFICATION: Automatic directory naming ---
    experiment_mode = "collaborative" if SABOTEUR_ACTIVE else "noisy_no_saboteur"
    save_directory = f"experiments/qaoa_{experiment_mode}/budget_{CNOT_BUDGET}_penalty_{CNOT_PENALTY_STRENGTH}"
    
    if not os.path.exists(save_directory): os.makedirs(save_directory)
    print(f"Results will be saved to: '{save_directory}'")
    checkpoint_path = os.path.join(save_directory, "qaoa_checkpoint.pth")
    
    start_generation = 0
    performance_history = {
        "avg_saboteur_reward": [], "best_architect_fitness": [], "avg_architect_fitness": [],
        "avg_cnot_count": []
    }
    best_fitness = -float('inf')
    generations_without_improvement = 0
    all_time_champion_individual = None
    
    if config.RESUME_TRAINING and os.path.exists(checkpoint_path):
        print(f"--- Resuming training from checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        start_generation = checkpoint.get('generation', 0) + 1
        performance_history = checkpoint.get('performance_history', performance_history)
        best_fitness = checkpoint.get('best_fitness', -float('inf'))
        generations_without_improvement = checkpoint.get('generations_without_improvement', 0)
        all_time_champion_individual = checkpoint.get('all_time_champion_individual')
        if saboteur_agent and 'saboteur_agent_state' in checkpoint:
            saboteur_agent.load_state(checkpoint['saboteur_agent_state'])
        if 'architect_agent_state' in checkpoint:
            architect_agent.load_state(checkpoint['architect_agent_state'])
    
    final_generation = start_generation - 1
    try:
        for generation in range(start_generation, config.GA_NUM_GENERATIONS):
            final_generation = generation
            print(f"\n{'='*20} Generation {generation + 1}/{config.GA_NUM_GENERATIONS} {'='*20}")

            population_tokens = architect_agent.propose_circuits_from_population(architect_base_helper)
            
            # Stage 1: Train clean circuits
            print(" > Stage 1: Training clean circuits and getting CNOT counts...")
            clean_train_worker = partial(train_clean_ansatz_qaoa_worker, hamiltonian=MAX_CUT_HAMILTONIAN, num_qubits=QAOA_NUM_QUBITS)
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                clean_results = pool.map(clean_train_worker, population_tokens)

            # Stage 2: Saboteur chooses actions
            sabotage_instructions_for_pop = [None] * len(population_tokens)
            if saboteur_agent:
                print(" > Stage 2: Saboteur choosing actions...")
                for i, (tokens, weights, baseline_exp_val, _) in enumerate(clean_results):
                    raw_state = (tokens, weights, baseline_exp_val) # Use positive exp_val for state
                    instruction = saboteur_agent.get_action_and_store_in_memory(raw_state)
                    sabotage_instructions_for_pop[i] = instruction

            # Stage 3: Evaluate attacked circuits
            print(" > Stage 3: Evaluating attacked circuits...")
            attack_eval_args = []
            for i, instruction in enumerate(sabotage_instructions_for_pop):
                tokens, weights, _, _ = clean_results[i]
                attack_eval_args.append((tokens, weights, instruction))
            
            attack_eval_worker = partial(evaluate_attacked_ansatz_qaoa_worker, hamiltonian=MAX_CUT_HAMILTONIAN, num_qubits=QAOA_NUM_QUBITS)
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                attacked_exp_vals = pool.starmap(attack_eval_worker, attack_eval_args)

            # Stage 4: Calculate fitness (with penalty) and rewards
            generation_fitness, generation_rewards, generation_cnot_counts = [], [], []
            for i in range(len(population_tokens)):
                _, _, baseline_exp_val, cnot_count = clean_results[i]
                attacked_exp_val = attacked_exp_vals[i]
                final_exp_val = attacked_exp_val if sabotage_instructions_for_pop[i] else baseline_exp_val
                
                cnot_penalty = 0.0
                if cnot_count > CNOT_BUDGET:
                    cnot_penalty = CNOT_PENALTY_STRENGTH * (cnot_count - CNOT_BUDGET)
                
                architect_fitness = final_exp_val - cnot_penalty
                saboteur_reward = baseline_exp_val - final_exp_val
                
                generation_fitness.append(architect_fitness)
                generation_rewards.append(saboteur_reward)
                generation_cnot_counts.append(cnot_count)

                if saboteur_agent and sabotage_instructions_for_pop[i]:
                    saboteur_agent.memory.rewards.append(saboteur_reward)
                    saboteur_agent.memory.is_terminals.append(True)

            if saboteur_agent and len(saboteur_agent.memory.actions) > 0:
                print(" > Updating Saboteur policy...")
                saboteur_agent.update()

            best_individual_of_generation = architect_agent.update_population(generation_fitness)
            current_best_fitness, avg_fitness = np.max(generation_fitness), np.mean(generation_fitness)
            avg_cnot = np.mean(generation_cnot_counts)
            
            performance_history["best_architect_fitness"].append(current_best_fitness)
            performance_history["avg_architect_fitness"].append(avg_fitness)
            performance_history["avg_saboteur_reward"].append(np.mean(generation_rewards))
            performance_history["avg_cnot_count"].append(avg_cnot)

            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Architect Fitness (<H> - Pen): Best={current_best_fitness:.6f}, Avg={avg_fitness:.6f}")
            print(f"  Circuit Complexity (CNOTs):    Avg={avg_cnot:.2f}")
            print(f"  Saboteur Reward (<H> Decrease):  Avg={np.mean(generation_rewards):.6f}")

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
                all_time_champion_individual = best_individual_of_generation
                print(f"  New best fitness observed: {best_fitness:.6f}")
            else:
                generations_without_improvement += 1

            if (generation + 1) % config.GA_CHECKPOINT_FREQUENCY == 0:
                print(f"\n--- Saving checkpoint at Generation {generation+1} ---")
                checkpoint_data = {
                    'generation': generation,
                    'performance_history': performance_history,
                    'best_fitness': best_fitness,
                    'generations_without_improvement': generations_without_improvement,
                    'all_time_champion_individual': all_time_champion_individual,
                    'architect_agent_state': architect_agent.save_state()
                }
                if saboteur_agent:
                    checkpoint_data['saboteur_agent_state'] = saboteur_agent.save_state()
                torch.save(checkpoint_data, checkpoint_path)

            if generations_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n--- EARLY STOPPING: No improvement for {config.EARLY_STOPPING_PATIENCE} generations. ---")
                break
    finally:
        print(f"\n--- QAOA Training Finished at Generation {final_generation + 1} ---")
        plot_performance_trends(performance_history, final_generation, config.GA_NUM_GENERATIONS, save_directory)
        if all_time_champion_individual:
            final_champion_path = os.path.join(save_directory, "qaoa_champion_circuit_final.json")
            with open(final_champion_path, 'w') as f:
                json.dump({"token_ids": all_time_champion_individual}, f, indent=4)
            if saboteur_agent:
                saboteur_model_path = os.path.join(save_directory, "qaoa_saboteur_model_final.pth")
                torch.save(saboteur_agent.save_state(), saboteur_model_path)
            print("Final QAOA artifacts saved.")
