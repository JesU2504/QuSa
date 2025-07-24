# coevolution_agents.py

import numpy as np
import random
from typing import List, Dict, Optional, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.distributions.categorical import Categorical

# PyTorch Geometric must be installed: pip install torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from Architect.architect import QuantumToken, Parameter, ParameterExpression, create_quantum_circuit_from_tokens
from Saboteur.saboteur import QuantumSaboteur
import config


class GeneticArchitectAgent:
    def __init__(self, population_size: int, token_sequence_length: int, num_total_tokens_in_pool: int,
                 elitism_count: int, mutation_rate: float, crossover_rate: float):
        self.population_size = population_size
        self.token_sequence_length = token_sequence_length
        self.num_total_tokens_in_pool = num_total_tokens_in_pool
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # Population stores tuples of ([token_ids], [weights] or None)
        self.population: List[Tuple[List[int], Optional[List[float]]]] = self._create_initial_population()
        # This will store the token IDs of the population before the update, for the final match
        self.population_before_update: List[List[int]] = []

    def save_state(self) -> Dict:
        return {'population': self.population}

    def load_state(self, state: Dict):
        self.population = state['population']
        print("Architect agent state (population with weights) loaded successfully.")

    def _create_initial_population(self) -> List[Tuple[List[int], None]]:
        # All individuals start with None for weights, signaling they need training.
        return [
            ([random.randrange(self.num_total_tokens_in_pool) for _ in range(self.token_sequence_length)], None)
            for _ in range(self.population_size)
        ]

    def propose_circuits_from_population(self, architect_base_helper: 'ArchitectAgent') -> List[Tuple[List[QuantumToken], Optional[List[float]]]]:
        # Proposes circuits for the main evaluation loop.
        proposed_circuits = []
        for individual_ids, weights in self.population:
            tokens = [architect_base_helper.global_token_pool[i] for i in individual_ids]
            proposed_circuits.append((tokens, weights))
        return proposed_circuits

    def _selection(self, individuals_with_results: List[Tuple[float, List[int], Optional[List[float]]]], num_parents: int) -> List[Tuple[List[int], Optional[List[float]]]]:
        """Performs tournament selection to choose parents."""
        parents = []
        for _ in range(num_parents):
            p1_idx, p2_idx = random.sample(range(len(individuals_with_results)), 2)
            
            p1_fitness = individuals_with_results[p1_idx][0] if individuals_with_results[p1_idx][0] != float('-inf') else -1e9
            p2_fitness = individuals_with_results[p2_idx][0] if individuals_with_results[p2_idx][0] != float('-inf') else -1e9

            winner_idx = p1_idx if p1_fitness >= p2_fitness else p2_idx
            
            # The winner is the tuple (fitness, tokens, weights)
            winner_tokens = individuals_with_results[winner_idx][1]
            winner_weights = individuals_with_results[winner_idx][2]
            parents.append((winner_tokens, winner_weights))
        return parents

    def _crossover(self, parent1_tokens: List[int], parent2_tokens: List[int]) -> List[int]:
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.token_sequence_length - 1)
            return parent1_tokens[:point] + parent2_tokens[point:]
        return random.choice([parent1_tokens, parent2_tokens])

    def _mutate(self, individual_tokens: List[int]) -> List[int]:
        return [
            random.randrange(self.num_total_tokens_in_pool) if random.random() < self.mutation_rate else token_id
            for token_id in individual_tokens
        ]

    def get_population_before_update(self) -> List[List[int]]:
        """Helper to get the tokens of the last evaluated population for the championship match."""
        return self.population_before_update

    def update_population(self, fitness_scores: List[float], generation_weights: List[List[float]]) -> Tuple[List[int], List[float]]:
        # Get the architectures that were just evaluated. Their order matches the incoming results.
        architectures_evaluated = [p[0] for p in self.population]
        
        # 1. Combine architectures with their new evaluation results (fitness and trained weights)
        individuals_with_results = list(zip(fitness_scores, architectures_evaluated, generation_weights))
        
        # 2. Sort the combined data by fitness (best to worst)
        sorted_individuals = sorted(individuals_with_results, key=lambda x: x[0] if x[0] != float('-inf') else -1e9, reverse=True)
        
        # 3. Save the sorted token lists for the championship match logic later
        self.population_before_update = [tokens for _, tokens, _ in sorted_individuals]

        # 4. Identify the best individual of this generation
        best_individual_tokens, best_individual_weights = sorted_individuals[0][1], sorted_individuals[0][2]

        # 5. Elitism: The best individuals carry over their architecture AND trained weights
        new_population = [(tokens, weights) for _, tokens, weights in sorted_individuals[:self.elitism_count]]
        
        # 6. Crossover and Mutation to fill the rest of the population
        while len(new_population) < self.population_size:
            # Select two parent individuals via tournament
            selected_parents = self._selection(individuals_with_results, 2)
            parent1_tokens = selected_parents[0][0]
            parent2_tokens = selected_parents[1][0]

            # Create a new child architecture
            child_tokens = self._crossover(parent1_tokens, parent2_tokens)
            mutated_child_tokens = self._mutate(child_tokens)

            # 7. THE FIX: New children have their weights RESET to None.
            # This forces them to be retrained from scratch and guarantees no mismatch.
            new_population.append((mutated_child_tokens, None))
        
        self.population = new_population
        return best_individual_tokens, best_individual_weights


# --- LEGACY DQN SABOTEUR AGENT AND COMPONENTS ---

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: Optional[np.ndarray], done: bool):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_tensor = torch.as_tensor([action], dtype=torch.long)
        reward_tensor = torch.as_tensor([reward], dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32) if next_state is not None else None
        self.buffer.append((state_tensor, action_tensor, reward_tensor, next_state_tensor, done))

    def sample(self, batch_size: int):
        state_b, action_b, reward_b, next_state_b, done_b = zip(*random.sample(self.buffer, batch_size))
        valid_next_states = [s for s in next_state_b if s is not None]
        next_state_stack = torch.stack(valid_next_states) if valid_next_states else None
        non_final_mask = torch.tensor([s is not None for s in next_state_b], dtype=torch.bool)
        return torch.stack(state_b), torch.stack(action_b), torch.stack(reward_b), next_state_stack, torch.tensor(done_b, dtype=torch.float32).unsqueeze(1), non_final_mask

    def __len__(self) -> int:
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim1: int = 256, hidden_dim2: int = 128, dropout_rate: float = 0.2):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, num_actions)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        return self.fc3(x)

class DQNSaboteurAgent:
    def __init__(self, gate_vocab: List[str], num_qubits_circuit: int, token_sequence_length: int,
                 num_variational_placeholders: int, max_feature_param_idx: int, max_eligible_targets_to_consider: int,
                 noise_strength_multipliers: List[float], base_noise_strength: float,
                 hw_error_rates: Optional[Dict], device_str: str = "cpu"):

        self.mode = 'dqn_saboteur'
        self.gate_vocab = sorted(list(set(gate_vocab + ["<PAD>", "<UNK>"])))
        self.gate_to_idx = {gate: i for i, gate in enumerate(self.gate_vocab)}
        self.num_gate_types = len(self.gate_vocab)
        self.num_qubits = num_qubits_circuit
        self.token_seq_len = token_sequence_length
        self.num_var_placeholders = num_variational_placeholders
        self.arg_type_vocab = ['is_none', 'is_float', 'is_param_p', 'is_param_w', 'is_param_expr']
        self.arg_type_to_idx = {name: i for i, name in enumerate(self.arg_type_vocab)}
        self.num_arg_types = len(self.arg_type_vocab)
        self.features_per_token = self.num_gate_types + 2 + self.num_arg_types + 1
        self.token_features_dim = self.features_per_token * self.token_seq_len
        self.weights_features_dim = self.num_var_placeholders
        self.global_features_dim = 8
        self.state_dim = self.token_features_dim + self.weights_features_dim + self.global_features_dim

        self.max_eligible_targets = max_eligible_targets_to_consider
        self.noise_multipliers = sorted(list(set(noise_strength_multipliers)))
        self.base_noise_strength = base_noise_strength
        self.num_noise_options_per_direction = len(self.noise_multipliers)
        self.num_directions = 2
        self.num_target_slots = self.max_eligible_targets
        self.num_actions = self.num_target_slots * self.num_directions * self.num_noise_options_per_direction
        
        self.gamma, self.epsilon, self.epsilon_decay, self.epsilon_min = 0.90, 1.0, 0.9995, 0.1
        self.batch_size = config.DQN_BATCH_SIZE
        self.target_update_freq = config.DQN_TARGET_UPDATE_FREQ
        self.total_steps_or_episodes = 0
        self.device = torch.device(device_str)
        self.policy_net = DQNNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.DQN_LEARNING_RATE, amsgrad=True)
        self.memory = ReplayBuffer(config.DQN_REPLAY_BUFFER_CAPACITY)
        self.saboteur_logic = QuantumSaboteur()
        self.current_eligible_target_details: List[Dict] = []
        self.last_dqn_action_idx_for_learning: int = -1

    def save_state(self) -> Dict:
        return {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory_buffer': list(self.memory.buffer),
            'epsilon': self.epsilon,
            'total_steps_or_episodes': self.total_steps_or_episodes,
        }

    def load_state(self, state: Dict):
        self.policy_net.load_state_dict(state['policy_net_state_dict'])
        self.target_net.load_state_dict(state['target_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.memory.buffer = deque(state['memory_buffer'], maxlen=self.memory.buffer.maxlen)
        self.epsilon = state['epsilon']
        self.total_steps_or_episodes = state['total_steps_or_episodes']
        print("DQN Saboteur agent state loaded successfully.")

    def _one_hot_encode(self, idx: int, vocab_size: int) -> np.ndarray:
        vec = np.zeros(vocab_size, dtype=np.float32)
        if 0 <= idx < vocab_size: vec[idx] = 1.0
        return vec

    def _featurize_state(self, raw_state: Tuple[List[QuantumToken], List[float], float]) -> Optional[np.ndarray]:
        selected_tokens, trained_weights, clean_score = raw_state
        token_features_list = []
        for i in range(self.token_seq_len):
            if i < len(selected_tokens):
                token = selected_tokens[i]
                gate_idx = self.gate_to_idx.get(token.gate_type.lower(), self.gate_to_idx["<UNK>"])
                gate_one_hot = self._one_hot_encode(gate_idx, self.num_gate_types)
                token_features_list.extend(gate_one_hot)
                q_indices = token.qubits if isinstance(token.qubits, list) else [token.qubits]
                q1_norm = (q_indices[0] / self.num_qubits) if len(q_indices) > 0 and self.num_qubits > 0 else 0.0
                q2_norm = (q_indices[1] / self.num_qubits) if len(q_indices) > 1 and self.num_qubits > 0 else 0.0
                token_features_list.extend([q1_norm, q2_norm])
                arg = token.argument
                arg_val_norm, arg_type_idx = 0.0, self.arg_type_to_idx['is_none']
                if isinstance(arg, (float, int)):
                    arg_type_idx, arg_val_norm = self.arg_type_to_idx['is_float'], float(arg) / (2 * np.pi)
                elif isinstance(arg, Parameter):
                    arg_type_idx = self.arg_type_to_idx['is_param_p'] if arg.name.startswith('p') else self.arg_type_to_idx['is_param_w']
                elif isinstance(arg, ParameterExpression): arg_type_idx = self.arg_type_to_idx['is_param_expr']
                arg_type_one_hot = self._one_hot_encode(arg_type_idx, self.num_arg_types)
                token_features_list.extend(arg_type_one_hot); token_features_list.append(np.clip(arg_val_norm, -1.0, 1.0))
            else:
                token_features_list.extend(self._one_hot_encode(self.gate_to_idx["<PAD>"], self.num_gate_types))
                token_features_list.extend([0.0, 0.0]); token_features_list.extend(self._one_hot_encode(self.arg_type_to_idx['is_none'], self.num_arg_types)); token_features_list.append(0.0)
        
        weights_features = np.zeros(self.num_var_placeholders, dtype=np.float32)
        if trained_weights:
            for i in range(min(len(trained_weights), self.num_var_placeholders)): weights_features[i] = trained_weights[i] / (2 * np.pi)
        
        num_cnot = sum(1 for t in selected_tokens if t.gate_type.lower() == 'cx')
        temp_qc = create_quantum_circuit_from_tokens(self.num_qubits, selected_tokens, 0)
        total_gates = len(selected_tokens)
        
        global_features = np.array([
            float(clean_score),
            (len(trained_weights) if trained_weights else 0) / self.num_var_placeholders if self.num_var_placeholders > 0 else 0.0,
            num_cnot / self.token_seq_len if self.token_seq_len > 0 else 0.0,
            len(self.saboteur_logic.get_tokens_with_any_argument(selected_tokens)) / self.token_seq_len if self.token_seq_len > 0 else 0.0,
            (temp_qc.depth() if temp_qc else 0) / self.token_seq_len if self.token_seq_len > 0 else 0.0,
            sum(1 for t in selected_tokens if len(t.qubits) == 1) / total_gates if total_gates > 0 else 0.0,
            sum(1 for t in selected_tokens if len(t.qubits) == 2) / total_gates if total_gates > 0 else 0.0,
            total_gates / self.token_seq_len if self.token_seq_len > 0 else 0.0,
        ], dtype=np.float32)

        final_state_vector = np.concatenate((np.array(token_features_list, dtype=np.float32), weights_features, global_features))
        if len(final_state_vector) != self.state_dim: raise ValueError(f"State dim mismatch! Expected {self.state_dim}, got {len(final_state_vector)}")
        return final_state_vector

    def _map_action_to_instruction(self, dqn_action_idx: int) -> Optional[Dict]:
        if dqn_action_idx < 0 or dqn_action_idx >= self.num_actions: return None
        if not self.current_eligible_target_details: return None
        actions_per_target_slot = self.num_directions * self.num_noise_options_per_direction
        target_slot_idx = dqn_action_idx // actions_per_target_slot
        if target_slot_idx >= len(self.current_eligible_target_details): return None
        remainder = dqn_action_idx % actions_per_target_slot
        direction_idx = remainder // self.num_noise_options_per_direction
        strength_multiplier_idx = remainder % self.num_noise_options_per_direction
        actual_token_original_idx = self.current_eligible_target_details[target_slot_idx]['original_idx']
        strength_multiplier = self.noise_multipliers[strength_multiplier_idx]
        direction = 1.0 if direction_idx == 0 else -1.0
        noise_val = direction * strength_multiplier * self.base_noise_strength
        return {'target_token_idx': actual_token_original_idx, 'noise_val': noise_val}

    def get_action(self, raw_state: Tuple[List[QuantumToken], List[float], float]) -> Optional[Dict]:
        self.last_dqn_action_idx_for_learning = -1
        selected_tokens_clean, _, _ = raw_state
        raw_eligible_targets = self.saboteur_logic.get_tokens_with_any_argument(selected_tokens_clean)
        self.current_eligible_target_details = [{'original_idx': idx, 'argument_obj': arg} for idx, arg in raw_eligible_targets[:self.max_eligible_targets]]
        num_valid_target_slots = len(self.current_eligible_target_details)
        if num_valid_target_slots == 0: return None
        num_valid_actions = num_valid_target_slots * self.num_directions * self.num_noise_options_per_direction
        
        if num_valid_actions == 0: return None
        featurized_state_np = self._featurize_state(raw_state)
        if featurized_state_np is None: return None
        
        state_tensor = torch.tensor(featurized_state_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        if random.random() < self.epsilon:
            dqn_action_idx = random.randrange(num_valid_actions)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                mask = torch.full_like(q_values, float('-inf')); mask[0, :num_valid_actions] = 0.0
                dqn_action_idx = (q_values + mask).max(1)[1].item()
            self.policy_net.train()
        
        instruction = self._map_action_to_instruction(dqn_action_idx)
        self.last_dqn_action_idx_for_learning = dqn_action_idx
        if instruction is None and num_valid_actions > 0:
            new_valid_action_idx = random.randrange(num_valid_actions)
            instruction = self._map_action_to_instruction(new_valid_action_idx)
            self.last_dqn_action_idx_for_learning = new_valid_action_idx
        return instruction
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size: return None
        self.policy_net.train()
        states_b, actions_b, rewards_b, next_states_stack, dones_b, non_final_mask = self.memory.sample(self.batch_size)
        states_b, actions_b, rewards_b, dones_b = states_b.to(self.device), actions_b.to(self.device), rewards_b.to(self.device), dones_b.to(self.device)
        if next_states_stack is not None: next_states_stack = next_states_stack.to(self.device)
        
        current_q_values = self.policy_net(states_b).gather(1, actions_b)
        next_q_values_target = torch.zeros(self.batch_size, device=self.device)
        if next_states_stack is not None:
             with torch.no_grad(): next_q_values_target[non_final_mask] = self.target_net(next_states_stack).max(1)[0]
        
        target_q_values = rewards_b + (self.gamma * next_q_values_target.unsqueeze(1) * (1 - dones_b))
        loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0); self.optimizer.step()
        return loss.item()

    def learn(self, raw_state: Tuple[List[QuantumToken], List[float], float], reward: float, next_raw_state: Optional[Tuple[List[QuantumToken], List[float], float]], done: bool):
        if self.last_dqn_action_idx_for_learning == -1: return None
        featurized_state_np = self._featurize_state(raw_state)
        if featurized_state_np is None: return None
        featurized_next_state_np = self._featurize_state(next_raw_state) if next_raw_state and not done else None
        if featurized_next_state_np is None: done = True
        
        self.memory.push(featurized_state_np, self.last_dqn_action_idx_for_learning, reward, featurized_next_state_np, done)
        if len(self.memory) >= self.batch_size:
            self.optimize_model()
        
        self.total_steps_or_episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.total_steps_or_episodes % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# --- ADVANCED PPO-GNN AGENT AND COMPONENTS ---

def _circuit_to_graph(
    tokens: List[QuantumToken],
    gate_vocab: List[str],
    arg_type_vocab: List[str],
    num_qubits: int,
    num_var_placeholders: int,
    trained_weights: Optional[List[float]] = None,
    clean_score: float = 0.0
) -> Optional[Data]:
    if not tokens: return None
    gate_to_idx = {gate: i for i, gate in enumerate(gate_vocab)}
    arg_type_to_idx = {name: i for i, name in enumerate(arg_type_vocab)}
    node_features = []
    for token in tokens:
        gate_idx = gate_to_idx.get(token.gate_type.lower(), gate_to_idx["<UNK>"])
        gate_one_hot = F.one_hot(torch.tensor(gate_idx), num_classes=len(gate_vocab))
        q_indices = token.qubits if isinstance(token.qubits, list) else [token.qubits]
        q1_norm = (q_indices[0] / num_qubits) if len(q_indices) > 0 and num_qubits > 0 else 0.0
        q2_norm = (q_indices[1] / num_qubits) if len(q_indices) > 1 and num_qubits > 0 else 0.0
        qubit_feats = torch.tensor([q1_norm, q2_norm], dtype=torch.float)
        arg = token.argument
        arg_val_norm, arg_type_idx = 0.0, arg_type_to_idx['is_none']
        if isinstance(arg, (float, int)):
            arg_type_idx, arg_val_norm = arg_type_to_idx['is_float'], float(arg) / (2 * np.pi)
        elif isinstance(arg, Parameter):
            arg_type_idx = arg_type_to_idx['is_param_p'] if arg.name.startswith('p') else arg_type_to_idx['is_param_w']
        elif isinstance(arg, ParameterExpression):
            arg_type_idx = arg_type_to_idx['is_param_expr']
        arg_type_one_hot = F.one_hot(torch.tensor(arg_type_idx), num_classes=len(arg_type_vocab))
        arg_feats = torch.cat([arg_type_one_hot.float(), torch.tensor([np.clip(arg_val_norm, -1.0, 1.0)], dtype=torch.float)])
        node_features.append(torch.cat([gate_one_hot.float(), qubit_feats, arg_feats]))
    x = torch.stack(node_features)
    edge_list = []
    qubit_last_use = [-1] * num_qubits
    for i, token in enumerate(tokens):
        for qubit_idx in token.qubits:
            if qubit_last_use[qubit_idx] != -1:
                edge_list.append([qubit_last_use[qubit_idx], i])
            qubit_last_use[qubit_idx] = i
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    weights_features = np.zeros(num_var_placeholders, dtype=np.float32)
    if trained_weights:
        for i in range(min(len(trained_weights), num_var_placeholders)):
            weights_features[i] = trained_weights[i] / (2 * np.pi)
    global_features = np.array([
        float(clean_score),
        len(trained_weights) / num_var_placeholders if num_var_placeholders > 0 else 0.0,
        sum(1 for t in tokens if t.gate_type.lower() == 'cx') / len(tokens) if tokens else 0.0,
        len(tokens) / config.TOKEN_SEQUENCE_LENGTH,
    ], dtype=np.float32)
    u = torch.tensor(np.concatenate([global_features, weights_features]), dtype=torch.float).unsqueeze(0)
    return Data(x=x, edge_index=edge_index, u=u)

class PPO_Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCriticGNN(nn.Module):
    def __init__(self, node_feature_dim, gnn_hidden_dim, global_feature_dim, num_actions):
        super(ActorCriticGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)
        combined_dim = gnn_hidden_dim + global_feature_dim
        self.actor_fc1 = nn.Linear(combined_dim, gnn_hidden_dim)
        self.actor_fc2 = nn.Linear(gnn_hidden_dim, num_actions)
        self.critic_fc1 = nn.Linear(combined_dim, gnn_hidden_dim)
        self.critic_fc2 = nn.Linear(gnn_hidden_dim, 1)

    def forward(self, data: Data):
        x, edge_index, batch, u = data.x, data.edge_index, data.batch, data.u
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        graph_embedding = global_mean_pool(x, batch)
        combined_features = torch.cat([graph_embedding, u], dim=1)
        action_logits = self.actor_fc2(F.relu(self.actor_fc1(combined_features)))
        state_value = self.critic_fc2(F.relu(self.critic_fc1(combined_features)))
        return action_logits, state_value


# coevolution_agents.py (Showing only the modified PPO_GNN_SaboteurAgent class)

# ... (all other classes and imports remain the same) ...

class PPO_GNN_SaboteurAgent:
    def __init__(self, gate_vocab: List[str], num_qubits_circuit: int, num_variational_placeholders: int,
                 noise_strength_multipliers: List[float], base_noise_strength: float,
                 hw_error_rates: Optional[Dict], device_str: str = "cpu"):
        self.device = torch.device(device_str)
        self.gamma = config.PPO_GAMMA
        self.eps_clip = config.PPO_EPS_CLIP
        self.k_epochs = config.PPO_K_EPOCHS
        self.saboteur_logic = QuantumSaboteur()
        self.memory = PPO_Memory()
        self.gate_vocab = sorted(list(set(gate_vocab + ["<PAD>", "<UNK>"])))
        self.arg_type_vocab = ['is_none', 'is_float', 'is_param_p', 'is_param_w', 'is_param_expr']
        self.num_qubits = num_qubits_circuit
        self.num_var_placeholders = num_variational_placeholders
        self.hw_error_rates = hw_error_rates
        self.base_noise_strength = base_noise_strength
        
        self.noise_strength_multipliers: List[float] = []
        self.single_attack_types: List[str] = []
        self.combo_attack_names: List[str] = []
        self.num_actions: int = 0
        self.num_single_actions: int = 0
        
        node_feature_dim = len(self.gate_vocab) + 2 + len(self.arg_type_vocab) + 1
        global_feature_dim = 4 + self.num_var_placeholders
        
        self.policy: Optional[ActorCriticGNN] = None
        self.policy_old: Optional[ActorCriticGNN] = None
        self.optimizer: Optional[torch.optim.Adam] = None
        self.mse_loss = nn.MSELoss()

        initial_stage_settings = config.SABOTEUR_CURRICULUM['stage_1']
        self.update_curriculum(initial_stage_settings)

    def update_curriculum(self, stage_settings: Dict[str, Any]):
        new_single_types = stage_settings['types']
        new_multipliers = stage_settings['multipliers']
        new_combo_names = stage_settings.get('combo_attacks', [])

        new_num_single_actions = self.num_qubits * len(new_single_types) * len(new_multipliers)
        new_num_combo_actions = len(new_combo_names)
        new_num_actions = new_num_single_actions + new_num_combo_actions
        
        if new_num_actions != self.num_actions:
            print(f"  > Saboteur Curriculum Change: Updating action space size to {new_num_actions} ({new_num_single_actions} single + {new_num_combo_actions} combo).")
            self.single_attack_types = new_single_types
            self.noise_strength_multipliers = new_multipliers
            self.combo_attack_names = new_combo_names
            self.num_actions = new_num_actions
            self.num_single_actions = new_num_single_actions

            node_feature_dim = len(self.gate_vocab) + 2 + len(self.arg_type_vocab) + 1
            global_feature_dim = 4 + self.num_var_placeholders
            
            self.policy = ActorCriticGNN(node_feature_dim, 128, global_feature_dim, self.num_actions).to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.PPO_LEARNING_RATE)
            self.policy_old = ActorCriticGNN(node_feature_dim, 128, global_feature_dim, self.num_actions).to(self.device)
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.noise_strength_multipliers = new_multipliers

    def _map_action_to_instruction(self, action_idx: int) -> Optional[Union[Dict, List[Dict]]]:
        if not self.single_attack_types and not self.combo_attack_names: return None

        # Check if the action is a SINGLE attack
        if action_idx < self.num_single_actions:
            actions_per_qubit = len(self.single_attack_types) * len(self.noise_strength_multipliers)
            if actions_per_qubit == 0: return None
            
            target_qubit = action_idx // actions_per_qubit
            remainder = action_idx % actions_per_qubit
            noise_type_idx = remainder // len(self.noise_strength_multipliers)
            strength_idx = remainder % len(self.noise_strength_multipliers)
            
            noise_type = self.single_attack_types[noise_type_idx]
            strength_multiplier = self.noise_strength_multipliers[strength_idx]

            base_error = self.hw_error_rates.get('two_qubit' if 'two_qubit' in noise_type else 'single_qubit', 0.01)
            prob = min(1.0, base_error * strength_multiplier)

            if 'two_qubit' in noise_type:
                neighbor_qubit = (target_qubit + 1) % self.num_qubits 
                return {'type': noise_type, 'target_qubits': [target_qubit, neighbor_qubit], 'probability': prob}
            else:
                return {'type': noise_type, 'target_qubit': target_qubit, 'probability': prob}
        
        # Otherwise, the action is a COMBO attack
        else:
            combo_idx = action_idx - self.num_single_actions
            if combo_idx >= len(self.combo_attack_names): return None # Should not happen

            combo_name = self.combo_attack_names[combo_idx]
            combo_instructions_template = config.PHYSICAL_NOISE_SETTINGS['combo_attacks'][combo_name]
            
            # Create a new list of instructions with calculated probabilities
            final_combo_instructions = []
            for instruction_template in combo_instructions_template:
                # Use a default strength multiplier for combos or make it part of the config
                strength_multiplier = self.noise_strength_multipliers[-1] # Use the strongest multiplier for combos
                base_error_type = 'two_qubit' if 'two_qubit' in instruction_template['type'] else 'single_qubit'
                base_error = self.hw_error_rates.get(base_error_type, 0.01)
                prob = min(1.0, base_error * strength_multiplier)
                
                final_instruction = instruction_template.copy()
                final_instruction['probability'] = prob
                final_combo_instructions.append(final_instruction)

            return final_combo_instructions

    def get_action_and_store_in_memory(self, raw_state: Tuple):
        tokens, trained_weights, clean_score = raw_state
        state_data = _circuit_to_graph(
            tokens,
            self.gate_vocab,
            self.arg_type_vocab,
            self.num_qubits,
            self.num_var_placeholders,
            trained_weights,
            clean_score
        )
        if state_data is None: return None
        state_data = state_data.to(self.device)
        with torch.no_grad():
            action_logits, _ = self.policy_old.forward(state_data)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        self.memory.states.append(state_data)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)
        return self._map_action_to_instruction(action.item())

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states_batch = Batch.from_data_list(self.memory.states).to(self.device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).to(self.device).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).to(self.device).detach()
        for _ in range(self.k_epochs):
            action_logits, state_values = self.policy.forward(old_states_batch)
            dist = Categorical(logits=action_logits)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - config.PPO_ENTROPY_COEFF * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def save_state(self):
        return {'policy_state_dict': self.policy.state_dict()}

    def load_state(self, state):
        self.policy.load_state_dict(state['policy_state_dict'])
        self.policy_old.load_state_dict(state['policy_state_dict'])