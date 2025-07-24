# advanced_trainer_vqe.py (Corrected Saboteur Logic)

import numpy as np
import math
import os
import json
from typing import List, Dict, Optional, Tuple
import random
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='qiskit_ibm_provider.api.session')

import multiprocessing
from functools import partial

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

import config
from coevolution_agents import PPO_GNN_SaboteurAgent, GeneticArchitectAgent
from coevolution_utils import apply_sabotage_instruction
from Architect.architect import ArchitectAgent, create_quantum_circuit_from_tokens, QuantumToken
from trainer_utils import train_for_energy
from utils.quantum_chemistry_utils import get_h2_hamiltonian, calculate_energy_expectation
from utils.qiskit_utils import apply_physical_noise

def plot_performance_trends(performance_history: Dict[str, List[float]], current_generation: int, total_generations: int, save_dir: str):
    """Plots the performance history of the VQE training."""
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    history_path = os.path.join(save_dir, "vqe_performance_history.json")
    try:
        with open(history_path, 'w') as f: json.dump({k: np.array(v).tolist() for k, v in performance_history.items()}, f, indent=4)
    except Exception as e: print(f"Warning: Could not save performance history: {e}")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    if not performance_history.get("avg_architect_fitness"):
        print("Warning: Performance history is empty. Cannot generate plot.")
        return

    generations = range(1, len(performance_history["avg_architect_fitness"]) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    plt.title(f"VQE Co-Evolutionary Arms Race - Up to Gen {current_generation+1}/{total_generations}")
    
    color1 = 'blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Architect Best Fitness (-Energy)', color=color1)
    ax1.plot(generations, performance_history["best_architect_fitness"], color=color1, linestyle='--', label="Architect Best Fitness (-Energy)")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    ax2 = ax1.twinx()
    color2 = 'orangered'
    ax2.set_ylabel('Saboteur Avg Reward (Energy Increase)', color=color2)
    ax2.plot(generations, performance_history["avg_saboteur_reward"], color=color2, linestyle='-', label="Saboteur Avg Reward")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(0, color=color2, linestyle=':', linewidth=1)

    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(save_dir, f"vqe_arms_race_trend_gen{current_generation+1}.png"))
    plt.close(fig)

# --- MODIFICATION: New worker for Stage 1: Training clean circuits ---
def train_clean_ansatz_worker(
    individual_tokens: List[QuantumToken],
    hamiltonian: SparsePauliOp,
    num_qubits: int
) -> Tuple:
    """Worker to train a clean VQE ansatz and return its performance."""
    qc_logical = create_quantum_circuit_from_tokens(num_qubits, individual_tokens, 0)
    if not qc_logical or qc_logical.num_qubits == 0 or not qc_logical.parameters:
        return (individual_tokens, [], -100.0) # Return high energy for invalid circuits

    trained_weights, _ = train_for_energy(qc_logical, hamiltonian, spsa_iters=150)
    baseline_energy = calculate_energy_expectation(qc_logical, hamiltonian, trained_weights)
    
    return (individual_tokens, trained_weights, baseline_energy)

# --- MODIFICATION: New worker for Stage 2: Evaluating attacked circuits ---
def evaluate_attacked_ansatz_worker(
    individual_tokens: List[QuantumToken],
    trained_weights: List[float],
    sabotage_instruction: Dict,
    hamiltonian: SparsePauliOp,
    num_qubits: int
) -> float:
    """Worker to apply an attack and calculate the new energy."""
    # Recreate the logical circuit. It's cheap to do.
    qc_logical = create_quantum_circuit_from_tokens(num_qubits, individual_tokens, 0)
    if not qc_logical:
        return 0 # Return high energy if circuit is invalid

    # Apply the physical noise instruction from the saboteur
    qc_sabotaged = apply_physical_noise(qc_logical, sabotage_instruction)
    
    if qc_sabotaged:
        attacked_energy = calculate_energy_expectation(qc_sabotaged, hamiltonian, trained_weights)
    else:
        attacked_energy = 0 # Assign a high energy if sabotage fails for some reason
        
    return attacked_energy

if __name__ == '__main__':
    # --- FIX: Added local flags to control experiment settings ---
    SABOTEUR_ACTIVE = True
    RESUME_TRAINING = False
    EXPERIMENT_TAG = None # Optional tag for the save directory
    EVALUATION_ENVIRONMENT = "noisy" # 'ideal' or 'noisy'
    
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

    VQE_NUM_QUBITS = 4
    H2_DISTANCE = 0.735
    H2_HAMILTONIAN = get_h2_hamiltonian(H2_DISTANCE)
    NUM_VARIATIONAL_PLACEHOLDERS = 10
    population_size = 15
    token_sequence_length=15
    print(f"--- Starting VQE Co-evolution for H2 molecule at {H2_DISTANCE}Å ({VQE_NUM_QUBITS} qubits) ---")
    
    architect_base_helper = ArchitectAgent(
        num_qubits=VQE_NUM_QUBITS,
        allowed_gates=config.ARCH_GATES_VOCAB,
        allowed_arguments=[f'w{i}' for i in range(NUM_VARIATIONAL_PLACEHOLDERS)],
        token_sequence_length=15
    )
    architect_agent = GeneticArchitectAgent(
        population_size=population_size,
        token_sequence_length=token_sequence_length,
        num_total_tokens_in_pool=len(architect_base_helper.global_token_pool),
        elitism_count=config.GA_ELITISM_COUNT,
        mutation_rate=config.GA_MUTATION_RATE,
        crossover_rate=config.GA_CROSSOVER_RATE
    )
    
    saboteur_agent = None
    saboteur_mode_str = "no_saboteur"
    if SABOTEUR_ACTIVE:
        print("Info: Configuring Saboteur for physical noise attacks.")
        saboteur_mode_str = 'saboteur'
        
        # --- FIX: Set the config attribute the agent class needs ---
        config.SABOTEUR_MODE = 'ppo_gnn_saboteur' 

        base_init_args = {
            'gate_vocab': config.ARCH_GATES_VOCAB,
            'num_qubits_circuit': VQE_NUM_QUBITS,
            'num_variational_placeholders': config.NUM_VARIATIONAL_PLACEHOLDERS,
            'noise_strength_multipliers': config.SABOTEUR_NOISE_STRENGTH_MULTIPLIERS,
            'base_noise_strength': config.SABOTEUR_NOISE_STRENGTH,
            'hw_error_rates': {'single_qubit': 0.001, 'two_qubit': 0.01} 
        }
        saboteur_agent = PPO_GNN_SaboteurAgent(**base_init_args, device_str="cpu")

    # Automatic directory naming
    save_directory = f"experiments/vqe/{EVALUATION_ENVIRONMENT}_{saboteur_mode_str}"
    if EXPERIMENT_TAG:
        save_directory = f"{save_directory}_{EXPERIMENT_TAG}"

    if not os.path.exists(save_directory): os.makedirs(save_directory)
    print(f"Results will be saved to: '{save_directory}'")
    checkpoint_path = os.path.join(save_directory, "vqe_training_checkpoint.pth")
    
    start_generation = 0
    performance_history = {"avg_saboteur_reward": [], "best_architect_fitness": [], "avg_architect_fitness": []}
    best_fitness = -float('inf')
    generations_without_improvement = 0
    all_time_champion_individual = None
    
    if RESUME_TRAINING and os.path.exists(checkpoint_path):
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
            
            # --- REFACTOR STAGE 1: Train all clean circuits first ---
            print(" > Stage 1: Training clean circuits in parallel...")
            clean_train_worker = partial(train_clean_ansatz_worker, hamiltonian=H2_HAMILTONIAN, num_qubits=VQE_NUM_QUBITS)
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                clean_results = pool.map(clean_train_worker, population_tokens)

            # --- REFACTOR STAGE 2: Saboteur chooses actions based on clean results ---
            sabotage_instructions_for_pop = [None] * len(population_tokens)
            if saboteur_agent:
                print(" > Stage 2: Saboteur choosing actions with full information...")
                for i, (tokens, weights, baseline_energy) in enumerate(clean_results):
                    # Create the state with the CORRECT trained weights and score
                    raw_state = (tokens, weights, -baseline_energy)
                    instruction = saboteur_agent.get_action_and_store_in_memory(raw_state)
                    sabotage_instructions_for_pop[i] = instruction

            # --- REFACTOR STAGE 3: Evaluate attacked circuits ---
            print(" > Stage 3: Evaluating attacked circuits in parallel...")
            attack_eval_args = []
            for i, instruction in enumerate(sabotage_instructions_for_pop):
                tokens, weights, _ = clean_results[i]
                attack_eval_args.append((tokens, weights, instruction))
            
            attack_eval_worker = partial(evaluate_attacked_ansatz_worker, hamiltonian=H2_HAMILTONIAN, num_qubits=VQE_NUM_QUBITS)
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                attacked_energies = pool.starmap(attack_eval_worker, attack_eval_args)

            # --- REFACTOR STAGE 4: Calculate fitness and rewards ---
            generation_fitness, generation_rewards = [], []
            for i in range(len(population_tokens)):
                _, _, baseline_energy = clean_results[i]
                attacked_energy = attacked_energies[i]
                
                # If no attack was applied, attacked_energy is the same as baseline
                final_energy = attacked_energy if sabotage_instructions_for_pop[i] else baseline_energy
                
                architect_fitness = -final_energy
                saboteur_reward = final_energy - baseline_energy
                
                generation_fitness.append(architect_fitness)
                generation_rewards.append(saboteur_reward)

                if saboteur_agent and sabotage_instructions_for_pop[i]:
                    saboteur_agent.memory.rewards.append(saboteur_reward)
                    saboteur_agent.memory.is_terminals.append(True)

            # --- Agent Updates and History Logging (Unchanged) ---
            if saboteur_agent and len(saboteur_agent.memory.actions) > 0:
                print(" > Updating Saboteur policy...")
                saboteur_agent.update()

            best_individual_of_generation = architect_agent.update_population(generation_fitness)
            current_best_fitness, avg_fitness = np.max(generation_fitness), np.mean(generation_fitness)
            
            performance_history["best_architect_fitness"].append(current_best_fitness)
            performance_history["avg_architect_fitness"].append(avg_fitness)
            performance_history["avg_saboteur_reward"].append(np.mean(generation_rewards))

            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Architect Fitness (-Energy): Best={current_best_fitness:.6f}, Avg={avg_fitness:.6f}")
            print(f"  Saboteur Reward (Energy Shift): Avg={np.mean(generation_rewards):.6f}")

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
        print(f"\n--- VQE Training Finished at Generation {final_generation + 1} ---")
        plot_performance_trends(performance_history, final_generation, config.GA_NUM_GENERATIONS, save_directory)
        if all_time_champion_individual:
            final_champion_path = os.path.join(save_directory, "vqe_champion_circuit_final.json")
            with open(final_champion_path, 'w') as f:
                json.dump({"token_ids": all_time_champion_individual}, f, indent=4)
            if saboteur_agent:
                saboteur_model_path = os.path.join(save_directory, "vqe_saboteur_model_final.pth")
                torch.save(saboteur_agent.save_state(), saboteur_model_path)
            print("Final VQE artifacts saved.")
