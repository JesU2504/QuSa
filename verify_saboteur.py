# verify_saboteur.py

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Any

from qiskit import transpile
from qiskit.circuit.library import TwoLocal

import config
from coevolution_agents import PPO_GNN_SaboteurAgent
from coevolution_utils import apply_sabotage_instruction
from Architect.architect import ArchitectAgent, create_quantum_circuit_from_tokens, QuantumToken
from trainer_utils import train_for_mse
from utils.data_loader import load_txt_data
from utils.qiskit_utils import calculate_circuit_mse, apply_physical_noise
from sklearn.utils import shuffle


def analyze_attacks(tokens: List[QuantumToken], attack_instructions: List[Dict[str, Any]]):
    if not attack_instructions:
        return

    print("\n  --- Saboteur Attack Analysis ---")
    
    param_attack_indices = [inst['target_token_idx'] for inst in attack_instructions if inst and 'target_token_idx' in inst]
    physical_attacks = [(inst['target_qubit'], inst['type'], inst['probability']) for inst in attack_instructions if inst and 'target_qubit' in inst]

    if param_attack_indices:
        print("    --- Parameter Attacks (on Tokens) ---")
        attack_counts = Counter(param_attack_indices)
        print(f"    {'Token Index':<15} | {'Gate Info':<25} | {'Times Attacked':<15}")
        print(f"    {'-'*15} | {'-'*25} | {'-'*15}")
        for token_idx, count in attack_counts.most_common():
            if token_idx < len(tokens):
                token = tokens[token_idx]
                gate_info = f"{token.gate_type.upper()} on q{token.qubits}"
                print(f"    {token_idx:<15} | {gate_info:<25} | {count:<15}")
        print()

    if physical_attacks:
        print("    --- Physical Noise Attacks (on Qubits) ---")
        attack_counts = Counter([(q, t) for q, t, _ in physical_attacks])
        print(f"    {'Qubit Index':<15} | {'Attack Type':<20} | {'Times Attacked':<15}")
        print(f"    {'-'*15} | {'-'*20} | {'-'*15}")
        for (qubit_idx, attack_type), count in attack_counts.most_common():
            print(f"    {qubit_idx:<15} | {attack_type:<20} | {count:<15}")

    if not param_attack_indices and not physical_attacks:
        print("    No valid attacks were recorded for analysis.")
        
    print("  ----------------------------------")


def run_verification(champion_path, saboteur_agent, architect_base_helper, x_train_set, y_train_pm1, x_test, y_test_pm1, num_attacks=10):
    with open(champion_path, 'r') as f:
        champion_data = json.load(f)

    qc_logical, feature_params, var_params, tokens = None, [], [], None
    name = os.path.basename(os.path.dirname(champion_path))

    if champion_data.get('type') == 'TwoLocal':
        name = champion_data['name']
        print(f"\n--- Verifying Benchmark Champion: {name} ---")
        qc_logical = TwoLocal(
            num_qubits=champion_data['num_qubits'],
            rotation_blocks=champion_data['rotation_blocks'],
            entanglement_blocks=champion_data['entanglement_blocks'],
            entanglement='full',
            reps=champion_data['reps']
        ).decompose()
        var_params = sorted(qc_logical.parameters, key=lambda p: p.name)
    else:
        print(f"\n--- Verifying Evolved Champion: {name} ---")
        token_ids = champion_data['token_ids']
        tokens = [architect_base_helper.global_token_pool[i] for i in token_ids]
        qc_logical = create_quantum_circuit_from_tokens(config.NUM_QUBITS_CIRCUIT, tokens, 0)
        feature_params = sorted([p for p in qc_logical.parameters if p.name.startswith('p')], key=lambda p: int(p.name[1:]))
        var_params = sorted([p for p in qc_logical.parameters if p.name.startswith('w')], key=lambda p: int(p.name[1:]))

    try:
        fig = qc_logical.draw(output='mpl', style='iqp')
        filename = f"verified_circuit_{name}.png"
        fig.savefig(filename)
        plt.close(fig)
        print(f"  > Saved circuit plot to '{filename}'")
    except Exception as e:
        print(f"  > Could not plot circuit: {e}")

    print(f"  > Optimizing weights for '{name}'...")
    trained_weights, loss_history = train_for_mse(qc_logical, feature_params, var_params, x_train_set, y_train_pm1, config.SHOTS_CONFIG, spsa_iters=150)

    if loss_history:
        plt.figure()
        plt.plot(loss_history, marker='o')
        plt.title(f"SPSA Training Loss ({name})")
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.savefig(f"spsa_loss_{name}.png")
        plt.close()

    baseline_mse = calculate_circuit_mse(qc_logical, trained_weights, feature_params, var_params, x_test, y_test_pm1,
                                         config.SHOTS_CONFIG, config.CLASSIFICATION_QUBIT_INDEX,
                                         config.CLASSIFICATION_OBSERVABLE_TYPE)

    print(f"  > Baseline MSE: {baseline_mse:.4f}")

    if tokens is None and saboteur_agent.attack_type != 'physical_noise':
        print("  > Skipping sabotage (no tokens for parameter attack).")
        return name, baseline_mse, baseline_mse, None

    print(f"  > Applying {num_attacks} Saboteur attacks...")
    attacked_mse_runs = []
    attack_instructions = []
    raw_state = (tokens if tokens else [], trained_weights, 1.0 - baseline_mse)

    for _ in range(num_attacks):
        instruction = saboteur_agent.get_action_and_store_in_memory(raw_state)
        attack_instructions.append(instruction)

        qc_sabotaged = qc_logical
        if instruction:
            if saboteur_agent.attack_type == 'physical_noise':
                qc_sabotaged = apply_physical_noise(qc_logical, instruction) or qc_logical
            elif tokens:
                sabotaged_tokens = apply_sabotage_instruction(tokens, instruction)
                qc_sabotaged = create_quantum_circuit_from_tokens(config.NUM_QUBITS_CIRCUIT, sabotaged_tokens, 0)

        attacked_mse = calculate_circuit_mse(qc_sabotaged, trained_weights, feature_params, var_params, x_test,
                                             y_test_pm1, config.SHOTS_CONFIG, config.CLASSIFICATION_QUBIT_INDEX,
                                             config.CLASSIFICATION_OBSERVABLE_TYPE)
        attacked_mse_runs.append(attacked_mse)

    avg_attacked_mse = np.mean(attacked_mse_runs)
    print(f"  > Average Attacked MSE: {avg_attacked_mse:.4f}")

    analyze_attacks(tokens, attack_instructions)
    return name, baseline_mse, avg_attacked_mse, attack_instructions


if __name__ == '__main__':
    path = 'experiments/moons/'
    path_data = 'experiment_data/moons/'
    control_paths = [
        f"{path}ideal_no_saboteur/champion_circuit_final.json",
        f"{path}noisy_no_saboteur/champion_circuit_final.json",
        "benchmark_circuits/vqe_benchmark_champion.json"
    ]
    robust_path = f"{path}ideal_ppo_gnn_saboteur/champion_circuit_final.json"
    saboteur_model_path = f"{path}ideal_ppo_gnn_saboteur/saboteur_model_final.pth"
    all_champions = control_paths + [robust_path]

    x_train = load_txt_data(os.path.join(path_data, "x_train.txt"))
    y_train_raw = load_txt_data(os.path.join(path_data, "y_train.txt"), dtype=float)
    x_train, y_train_raw = shuffle(x_train, y_train_raw, random_state=42)
    x_test = load_txt_data(os.path.join(path_data, "x_test.txt"))
    y_test_raw = load_txt_data(os.path.join(path_data, "y_test.txt"), dtype=float)

    def convert_pm1(y): return np.where(np.argmax(y, axis=1), -1.0, 1.0) if y.ndim == 2 else np.where(y.astype(int).ravel() == 1, -1.0, 1.0)
    y_train_pm1 = convert_pm1(y_train_raw)
    y_test_pm1 = convert_pm1(y_test_raw)

    model_dir_name = os.path.basename(os.path.dirname(saboteur_model_path))
    config.SABOTEUR_MODE = 'ppo_gnn_saboteur' if 'ppo_gnn_saboteur' in model_dir_name else 'parameter_attack'
    print(f"\nInfo: Using Saboteur mode: {config.SABOTEUR_MODE}")

    architect_helper = ArchitectAgent(
        num_qubits=config.NUM_QUBITS_CIRCUIT,
        allowed_gates=config.ARCH_GATES_VOCAB,
        allowed_arguments=[f'p{i}' for i in range(x_train.shape[1])] + [f'w{i}' for i in range(config.NUM_VARIATIONAL_PLACEHOLDERS)],
        token_sequence_length=config.TOKEN_SEQUENCE_LENGTH
    )

    print(f"Loading Saboteur model from {saboteur_model_path}...")
    saboteur_agent = PPO_GNN_SaboteurAgent(
        gate_vocab=config.ARCH_GATES_VOCAB,
        num_qubits_circuit=config.NUM_QUBITS_CIRCUIT,
        num_variational_placeholders=config.NUM_VARIATIONAL_PLACEHOLDERS,
        noise_strength_multipliers=config.SABOTEUR_CURRICULUM['stage_1']['multipliers'],
        base_noise_strength=config.SABOTEUR_NOISE_STRENGTH,
        hw_error_rates=None,
        device_str="cpu"
    )

    # ⚠️ Manually override action space size
    saboteur_agent.action_space_size = 95
    saboteur_agent.policy.action_space_size = 95
    saboteur_agent.policy_old.action_space_size = 95

    checkpoint = torch.load(saboteur_model_path, map_location=saboteur_agent.device)
    saboteur_agent.load_state(checkpoint)
    saboteur_agent.policy.eval()
    saboteur_agent.policy_old.eval()

    results = {}
    for champion_file in all_champions:
        if not os.path.exists(champion_file):
            print(f"Warning: Champion file not found, skipping: {champion_file}")
            continue
        champ_name, baseline, attacked, _ = run_verification(
            champion_file, saboteur_agent, architect_helper,
            x_train, y_train_pm1, x_test, y_test_pm1
        )
        results[champ_name] = attacked - baseline

    print("\n\n" + "=" * 50)
    print("--- SABOTEUR VERIFICATION SUMMARY ---")
    print("=" * 50)
    print(f"{'Champion Architecture':<45} | {'Fragility (MSE Increase)':<25}")
    print(f"{'-'*45} | {'-'*25}")
    for name, fragility in results.items():
        clean_name = name.replace("performance_plots_", "").replace("_control_run", "").replace("_collaborative", "")
        print(f"{clean_name:<45} | {fragility:<25.4f}")
    print("=" * 50)
