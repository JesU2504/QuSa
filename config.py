# config.py (Updated Version)

from typing import Dict

# -----------------------------------------------------------------------------
# --- PARAMETER SETS FOR DIFFERENT USE CASES ---
# -----------------------------------------------------------------------------


thesis_run_settings = {
    "TOKEN_SEQUENCE_LENGTH": 20,
    "GA_NUM_GENERATIONS": 150, 
    "GA_POPULATION_SIZE": 25,
    "SPSA_MAX_ITERATIONS_CLEAN": 30,
    "SHOTS_CONFIG": 1024, "EARLY_STOPPING_PATIENCE": 150,
    "RESUME_TRAINING": False
}

balanced_run_settings = {
    "TOKEN_SEQUENCE_LENGTH": 15,         
    "GA_NUM_GENERATIONS": 150,           
    "GA_POPULATION_SIZE": 20,            
    "SPSA_MAX_ITERATIONS_CLEAN": 20,     
    "SHOTS_CONFIG": 1024,                
    "EARLY_STOPPING_PATIENCE": 100,      
    "RESUME_TRAINING": False            
}

# -----------------------------------------------------------------------------
# --- MAIN CONFIGURATION ---
# -----------------------------------------------------------------------------
# MODIFIED: Using the balanced settings as a base for a more challenging run
active_settings = balanced_run_settings

# --- Experiment Control Settings ---
TRAIN_WITH_ALL_TO_ALL_CONNECTIVITY = False
RESUME_TRAINING = active_settings["RESUME_TRAINING"]

# --- Global Configuration & Constants ---
DATA_DIR = "experiment_data"
CHOSEN_DATASET = None
NUM_QUBITS_CIRCUIT = 5
NUM_VARIATIONAL_PLACEHOLDERS = 10
TOKEN_SEQUENCE_LENGTH = active_settings["TOKEN_SEQUENCE_LENGTH"]
GA_POPULATION_SIZE = active_settings["GA_POPULATION_SIZE"]

# --- Training & Evaluation Settings ---
# NOTE: NUM_TRAIN_SAMPLES and NUM_VALIDATION_SAMPLES are commented out
# to align with the strategy of using the full train/test sets from previous changes.
# If you wish to use sampling, uncomment these and the relevant lines in the trainer.
# NUM_TRAIN_SAMPLES = 150
# NUM_VALIDATION_SAMPLES = 50
SHOTS_CONFIG = active_settings["SHOTS_CONFIG"]
EARLY_STOPPING_PATIENCE = active_settings["EARLY_STOPPING_PATIENCE"]

# --- Architect (Genetic Algorithm) Settings ---
GA_NUM_GENERATIONS = active_settings["GA_NUM_GENERATIONS"]
GA_ELITISM_COUNT = 2
GA_MUTATION_RATE = 0.05
GA_CROSSOVER_RATE = 0.7
ARCH_GATES_VOCAB = ['rx', 'ry', 'rz', 'h', 'cx', 'rzz']

# --- Circuit & Optimizer Settings ---
SPSA_MAX_ITERATIONS_CLEAN = active_settings["SPSA_MAX_ITERATIONS_CLEAN"]
CLEAN_CIRCUIT_OPTIMIZER = "SPSA"
SPSA_LEARNING_RATE = 0.1
SPSA_PERTURBATION = 0.1
SPSA_RESAMPLINGS = 1
ADAM_MAX_ITERATIONS_CLEAN = 15
ADAM_LEARNING_RATE_CLEAN = 0.05

# --- Classification Settings ---
CLASSIFICATION_QUBIT_INDEX = 0
CLASSIFICATION_OBSERVABLE_TYPE = "Z"
LABEL_ASSIGNMENT_METHOD = "Z_expectation"
LABEL_ASSIGNMENT_THRESHOLD = 0.0
CLASS_0_IS_POSITIVE_Z_EXPECTATION = True

# --- Saboteur Settings ---
# MODIFIED: Reduced base strength to force the agent to learn to use multipliers effectively.
SABOTEUR_NOISE_STRENGTH = 0.001
SABOTEUR_NOISE_STRENGTH_MULTIPLIERS = [1.0, 2.0, 5] # Expanded range

PHYSICAL_NOISE_SETTINGS = {
    'types': ['bit_flip', 'phase_flip', 'depolarizing', 'amplitude_damping', 'phase_damping', 'two_qubit_depolarizing'],
    'combo_attacks': {
        'crosstalk_q1_q2_q3': [
            {'type': 'bit_flip', 'target_qubit': 1},
            {'type': 'depolarizing', 'target_qubit': 2},
            {'type': 'phase_flip', 'target_qubit': 3}
        ],
        'global_depolarizing': [
            {'type': 'depolarizing', 'target_qubit': i} for i in range(NUM_QUBITS_CIRCUIT)
        ],
        'correlated_damping': [
            {'type': 'amplitude_damping', 'target_qubit': 0},
            {'type': 'phase_damping', 'target_qubit': 2},
            {'type': 'amplitude_damping', 'target_qubit': 4},
        ],
        'dual_cnot_failure': [
            {'type': 'two_qubit_depolarizing', 'target_qubits': [0, 1]},
            {'type': 'two_qubit_depolarizing', 'target_qubits': [3, 4]}
        ]
        # 'global_readout_error': [
        #     {'type': 'bit_flip', 'target_qubit': i} for i in range(NUM_QUBITS_CIRCUIT)
        # ]
    }
}

# MODIFIED: The curriculum is now more aggressive to pose a greater challenge.
SABOTEUR_CURRICULUM = {
    'stage_1': {
        'end_gen': 20,
        'multipliers': [0.0, 1.0, 2.0], # Start with stronger multipliers
        'types': ['bit_flip', 'phase_flip', 'depolarizing'], # Introduce depolarizing early
        'combo_attacks': [] 
    },
    'stage_2': {
        'end_gen': 60,
        'multipliers': [0.0, 2.0, 5.0],
        'types': ['bit_flip', 'phase_flip', 'depolarizing', 'amplitude_damping', 'phase_damping'], # All single-qubit errors
        'combo_attacks': ['crosstalk_q1_q2_q3', 'global_depolarizing'] # Introduce combos
    },
    'stage_3': {
        'multipliers': [0.0, 5.0, 10.0],
        'types': list(PHYSICAL_NOISE_SETTINGS['types']), # All single types
        'combo_attacks': list(PHYSICAL_NOISE_SETTINGS['combo_attacks'].keys()) # All combo attacks
    }
}

# --- PPO Hyperparameters ---
PPO_LEARNING_RATE = 0.0003
PPO_GAMMA = 0.99
PPO_K_EPOCHS = 5
PPO_EPS_CLIP = 0.4
PPO_ENTROPY_COEFF = 0.05

# --- DQN Hyperparameters ---
DQN_REPLAY_BUFFER_CAPACITY = 10000
DQN_BATCH_SIZE = 64
DQN_TARGET_UPDATE_FREQ = 20
DQN_LEARNING_RATE = 1e-4

# --- Save & Plotting Settings ---
GA_PLOT_FREQUENCY = 10
GA_CHECKPOINT_FREQUENCY = 10
PLAYOFF_NUM_CONTENDERS = 15
