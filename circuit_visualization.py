import os
import json
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap

# Assuming the following files are in the same directory or accessible
import config
from Architect.architect import ArchitectAgent, create_quantum_circuit_from_tokens
from utils.data_loader import load_txt_data


# visualize_circuits.py (Updated to include VQE Benchmark)

import os
import json
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library import TwoLocal # <-- Import TwoLocal

# Assuming the following files are in the same directory or accessible
import config
from Architect.architect import ArchitectAgent, create_quantum_circuit_from_tokens
from utils.data_loader import load_txt_data

dataset = 'vowel_2'

if __name__ == '__main__':
    # --- MODIFICATION: Add the VQE benchmark to the list ---
    experiments_to_visualize = [
        f"experiments/{dataset}/ideal_no_saboteur/champion_circuit_final.json",
        f"experiments/{dataset}/noisy_no_saboteur/champion_circuit_final.json",
        f"experiments/{dataset}/ideal_ppo_gnn_saboteur/champion_circuit_final.json",
        #f"experiments/{dataset}/noisy_ppo_gnn_saboteur/champion_circuit_final.json",
        f"benchmark_circuits/vqe_benchmark_champion.json" # <-- VQE benchmark added
    ]
    # --------------------------------------------------------
    
    output_dir = "circuit_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Load Hardware Configuration for Transpilation ---
    print("Loading hardware configuration from snapshot file...")
    SNAPSHOT_FILENAME = 'backend_files/backend_snapshot_test.json' 
    try:
        with open(SNAPSHOT_FILENAME, 'r') as f:
            snapshot_data = json.load(f)
        
        coupling_map = CouplingMap(couplinglist=snapshot_data['coupling_map'])
        basis_gates = snapshot_data['basis_gates']
        print(f"Loaded transpilation settings from '{snapshot_data['backend_name']}' snapshot.")
    except Exception as e:
        print(f"ERROR: Could not load snapshot file '{SNAPSHOT_FILENAME}'. Cannot transpile circuits. {e}")
        exit()

    # --- Initialize Architect Helper ---
    x_train_all = load_txt_data(os.path.join('experiment_data', dataset, "x_train.txt"))
    if x_train_all is None:
        print("Could not load data to determine number of features. Exiting.")
        exit()
    num_features = x_train_all.shape[1]
    
    architect_base_helper = ArchitectAgent(
        num_qubits=config.NUM_QUBITS_CIRCUIT,
        allowed_gates=config.ARCH_GATES_VOCAB,
        allowed_arguments=[f'p{i}' for i in range(num_features)] + [f'w{i}' for i in range(config.NUM_VARIATIONAL_PLACEHOLDERS)],
        token_sequence_length=config.TOKEN_SEQUENCE_LENGTH
    )
    print(f"Architect helper initialized with {num_features} features.")

    # --- Loop Through Champions to Gather Data ---
    all_results = {}
    
    for path in experiments_to_visualize:
        try:
            # --- MODIFICATION: More robust name handling ---
            short_name = ""
            logical_qc = None

            print(f"\nProcessing circuit from: {path}")
            
            with open(path, 'r') as f:
                champion_data = json.load(f)

            # --- MODIFICATION: Logic to handle both evolved and VQE benchmark circuits ---
            if champion_data.get('type') == 'TwoLocal':
                # This is our VQE benchmark
                short_name = champion_data['name']
                print(f" > Handling VQE benchmark: {short_name}")
                vqe_config = champion_data
                logical_qc = TwoLocal(
                    num_qubits=vqe_config['num_qubits'],
                    rotation_blocks=vqe_config['rotation_blocks'],
                    entanglement_blocks=vqe_config['entanglement_blocks'],
                    entanglement='full',
                    reps=vqe_config['reps']
                ).decompose() # Decompose into basic gates immediately
            
            elif 'token_ids' in champion_data:
                # This is an evolved champion
                name = os.path.basename(os.path.dirname(path))
                short_name = name.replace("performance_plots_", "")
                print(f" > Handling evolved champion: {short_name}")
                tokens = [architect_base_helper.global_token_pool[i] for i in champion_data['token_ids']]
                logical_qc = create_quantum_circuit_from_tokens(config.NUM_QUBITS_CIRCUIT, tokens, 0)
            
            else:
                print(f"  > Skipping unknown champion type in {path}.")
                continue
            # --------------------------------------------------------------------------

            if not logical_qc:
                print(f"  > Could not create logical circuit. Skipping.")
                continue

            physical_qc = transpile(
                logical_qc,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                optimization_level=3,
                seed_transpiler=42
            )
            
            all_results[short_name] = {
                'logical_depth': logical_qc.depth(),
                'physical_depth': physical_qc.depth(),
                'logical_gates': sum(logical_qc.count_ops().values()),
                'physical_gates': sum(physical_qc.count_ops().values()),
                'logical_cnots': logical_qc.count_ops().get('cx', 0),
                'physical_cnots': physical_qc.count_ops().get('cx', 0),
                'logical_rzzs': logical_qc.count_ops().get('rzz', 0),
                'physical_swaps': physical_qc.count_ops().get('swap', 0)
            }
            print(f"  > Processed successfully.")

        except FileNotFoundError:
            print(f"\nWarning: Could not find file {path}. Skipping.")
        except Exception as e:
            print(f"\nAn error occurred while processing {path}: {e}")

    # --- Print the final summary table ---
    print("\n\n" + "="*120)
    print("--- Transpilation Summary ---")
    print("="*120)
    
    header_format = "{:<45} | {:<35} | {:<35}"
    row_format =    "{:<45} | {:<35} | {:<35}"
    
    print(header_format.format("Champion", "Logical Circuit Metrics", "Physical Circuit Metrics"))
    print("-" * 120)

    for name, metrics in all_results.items():
        depth_cost = (metrics['physical_depth'] / metrics['logical_depth']) if metrics['logical_depth'] > 0 else float('inf')
        gate_cost = (metrics['physical_gates'] / metrics['logical_gates']) if metrics['logical_gates'] > 0 else float('inf')

        logical_str = f"D={metrics['logical_depth']}, G={metrics['logical_gates']} ({metrics['logical_cnots']} CX, {metrics['logical_rzzs']} RZZ)"
        physical_str = f"D={metrics['physical_depth']} ({depth_cost:.1f}x), G={metrics['physical_gates']} ({gate_cost:.1f}x) | {metrics['physical_cnots']} CX, {metrics['physical_swaps']} SWAP"
        
        print(row_format.format(
            name,
            logical_str,
            physical_str,
        ))
    print("="*120)