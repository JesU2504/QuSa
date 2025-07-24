import random
import math
import copy 
from typing import List, Union, Optional, Dict, Tuple

# Assuming QuantumToken and Parameter/ParameterExpression are defined elsewhere
# For type hinting, we can do this:
try:
    from qiskit.circuit import Parameter, ParameterExpression
    # If your QuantumToken is in architect.py:
    # from Architect.architect import QuantumToken # Adjust path if needed
    # For now, let's define a placeholder if not found, for linting
    class QuantumToken: # Placeholder
        def __init__(self, gate_type: str, qubits: Union[int, List[int]], argument: Optional[Union[float, str, Parameter, ParameterExpression]] = None):
            self.gate_type = gate_type
            if isinstance(qubits, int): self.qubits = [qubits]
            else: self.qubits = qubits
            self.argument = argument
except ImportError:
    print("Warning: Qiskit or Architect components not found for full type hinting in saboteur.py")
    Parameter = type("Parameter", (), {})
    ParameterExpression = type("ParameterExpression", (), {})
    class QuantumToken: # Placeholder
        def __init__(self, gate_type: str, qubits: Union[int, List[int]], argument: Optional[Union[float, str, Parameter, ParameterExpression]] = None):
            self.gate_type = gate_type
            if isinstance(qubits, int): self.qubits = [qubits]
            else: self.qubits = qubits
            self.argument = argument


class QuantumSaboteur:
    """
    Applies modifications to quantum circuit designs to introduce fragility
    or test robustness. Operates on a list of QuantumToken objects.
    """

    def __init__(self):
        """Initializes the QuantumSaboteur."""
        # Gates known to take a single numerical angle as their primary argument
        self.eligible_rotational_gates = {'rx', 'ry', 'rz', 'u1', 'p', 
                                          'rzz', 'crx', 'cry', 'crz', 'cu1', 'cp'}
        #print("QuantumSaboteur initialized, ready to make circuits fragile.")

    def get_eligible_numerical_angle_tokens(
        self,
        original_tokens: List[QuantumToken]
    ) -> List[Tuple[int, QuantumToken]]:
        """
        Identifies tokens that are rotational gates with fixed numerical angles.

        Args:
            original_tokens (List[QuantumToken]): The input list of QuantumToken objects.

        Returns:
            List[Tuple[int, QuantumToken]]: A list of tuples, where each tuple is
                                             (original_index, token_object) for eligible tokens.
        """
        eligible = []
        for i, token in enumerate(original_tokens):
            if token.gate_type.lower() in self.eligible_rotational_gates and \
               isinstance(token.argument, (float, int)):
                eligible.append((i, token))
        return eligible
    
    def get_tokens_with_any_argument(self, circuit_tokens: List['QuantumToken']) -> List[Tuple[int, any]]:
        """
        Identifies all tokens in the circuit that have a non-None argument.
        Returns a list of tuples: (original_token_index, token_argument_value_or_object).
        """
        eligible_for_arg_sabotage = []
        for i, token in enumerate(circuit_tokens):
            if token.argument is not None:
                eligible_for_arg_sabotage.append((i, token.argument))
        return eligible_for_arg_sabotage

    
    def apply_targeted_noise_to_angles(
        self,
        original_tokens: List[QuantumToken],
        sabotage_instructions: Dict[int, float], # {token_index_to_sabotage: noise_value_to_add}
        verbose: bool = False
    ) -> List[QuantumToken]:
        """
        Applies specific noise values to the numerical angles of specified tokens.

        Args:
            original_tokens (List[QuantumToken]): The input list of QuantumToken objects.
            sabotage_instructions (Dict[int, float]): A dictionary where keys are the
                original indices of tokens in `original_tokens` to be sabotaged,
                and values are the noise values to add to their angles.
            verbose (bool): If True, prints details of applied sabotage.

        Returns:
            List[QuantumToken]: A new list of QuantumToken objects with specified noise applied.
        """
        # Create deep copies of tokens to ensure original list and its objects are not modified
        sabotaged_tokens = [copy.deepcopy(token) for token in original_tokens]
        num_sabotaged_actually = 0

        for token_index, noise_to_add in sabotage_instructions.items():
            if 0 <= token_index < len(sabotaged_tokens):
                token_to_modify = sabotaged_tokens[token_index] # This is now a deep copy

                if token_to_modify.gate_type.lower() in self.eligible_rotational_gates and \
                   isinstance(token_to_modify.argument, (float, int)):
                    
                    original_angle = float(token_to_modify.argument)
                    sabotaged_angle = original_angle + noise_to_add
                    
                    # Update the argument of the copied token
                    token_to_modify.argument = sabotaged_angle
                    num_sabotaged_actually +=1
                    if verbose:
                        print(f"Saboteur: Applied noise to token at index {token_index} "
                              f"(Gate: {token_to_modify.gate_type}, Qubits: {token_to_modify.qubits}). "
                              f"Original angle: {original_angle:.4f}, Noise: {noise_to_add:.4f}, "
                              f"New angle: {sabotaged_angle:.4f}")
                elif verbose:
                    print(f"Saboteur: Token at index {token_index} (Gate: {token_to_modify.gate_type}) "
                          "was targeted but is not eligible for numerical angle noise (e.g., wrong type or symbolic arg).")
            elif verbose:
                print(f"Saboteur: Invalid token index {token_index} in sabotage_instructions. Skipping.")
        
        if verbose:
            print(f"Saboteur: Applied targeted noise to {num_sabotaged_actually} gate angles.")
        return sabotaged_tokens

    # Previous random sabotage method (can be kept for comparison or other uses)
    def add_random_noise_to_numerical_angles(
        self,
        original_tokens: List[QuantumToken],
        noise_strength: float = 0.05, 
        sabotage_probability: float = 0.1, 
        noise_type: str = "uniform" 
    ) -> List[QuantumToken]:
        if not (0.0 <= sabotage_probability <= 1.0):
            raise ValueError("Sabotage probability must be between 0.0 and 1.0.")
        if noise_strength < 0:
            raise ValueError("Noise strength cannot be negative.")

        sabotaged_tokens: List[QuantumToken] = []
        num_sabotaged = 0
        for original_token in original_tokens:
            new_token_arg = original_token.argument
            is_sabotaged_this_token = False
            gate_type_lower = original_token.gate_type.lower()
            is_rotational = gate_type_lower in self.eligible_rotational_gates

            if is_rotational and isinstance(original_token.argument, (float, int)):
                if random.random() < sabotage_probability:
                    numerical_angle = float(original_token.argument)
                    noise = 0.0
                    if noise_type == "uniform": noise = random.uniform(-noise_strength, noise_strength)
                    elif noise_type == "normal": noise = random.normalvariate(0, noise_strength)
                    else: noise = random.uniform(-noise_strength, noise_strength)
                    new_token_arg = numerical_angle + noise
                    is_sabotaged_this_token = True
                    num_sabotaged += 1
            
            sabotaged_tokens.append(
                QuantumToken(gate_type=original_token.gate_type, qubits=list(original_token.qubits), argument=new_token_arg)
            )
        # if num_sabotaged > 0: print(f"Saboteur (Random): Added noise to {num_sabotaged} angles.")
        return sabotaged_tokens


if __name__ == '__main__':
    print("--- Testing QuantumSaboteur (Targeted Noise) ---")
    saboteur = QuantumSaboteur()

    if 'Parameter' not in globals(): # Mock for standalone testing
        class Parameter:
            def __init__(self, name): self.name = name
            def __repr__(self): return f"Parameter({self.name})"
        class ParameterExpression: # Simplified mock
            def __init__(self, expr_str): self.expr_str = expr_str; self.parameters = []
            def __repr__(self): return f"ParameterExpression({self.expr_str})"

    p0 = Parameter('p0'); w0 = Parameter('w0')
    original_tokens_test = [
        QuantumToken(gate_type='h', qubits=0),
        QuantumToken(gate_type='rx', qubits=0, argument=1.5708), # Eligible
        QuantumToken(gate_type='ry', qubits=1, argument=p0),    
        QuantumToken(gate_type='rz', qubits=0, argument=0.5),   # Eligible
        QuantumToken(gate_type='cx', qubits=[0, 1]),
        QuantumToken(gate_type='p', qubits=1, argument=-0.25),  # Eligible
        QuantumToken(gate_type='rx', qubits=0, argument=w0) 
    ]
    
    print("\nOriginal Tokens:")
    for i, t in enumerate(original_tokens_test): print(f"  Idx {i}: {t.gate_type} on {t.qubits}, arg: {t.argument}")

    eligible_for_sabotage = saboteur.get_eligible_numerical_angle_tokens(original_tokens_test)
    print("\nEligible for numerical angle sabotage (index, token):")
    for idx, token in eligible_for_sabotage:
        print(f"  Index {idx}: Gate {token.gate_type}, Qubits {token.qubits}, Angle {token.argument}")

    # Example: Saboteur's "learning algorithm" decides to hit token at index 1 with +0.1 noise
    # and token at index 3 with -0.05 noise.
    sabotage_plan = {
        1: 0.1,  # Add 0.1 to angle of token at original_tokens_test[1] (the RX gate)
        3: -0.05 # Add -0.05 to angle of token at original_tokens_test[3] (the RZ gate)
    }
    print(f"\nApplying sabotage plan: {sabotage_plan}")
    
    sabotaged_list = saboteur.apply_targeted_noise_to_angles(original_tokens_test, sabotage_plan, verbose=True)

    print("\nFinal (Sabotaged) Token List:")
    for i, t_sab in enumerate(sabotaged_list):
        change_indicator = ""
        if i in sabotage_plan and isinstance(original_tokens_test[i].argument, (float,int)) \
           and isinstance(t_sab.argument, (float,int)) \
           and not math.isclose(float(original_tokens_test[i].argument), float(t_sab.argument)):
            change_indicator = f"  <--- MODIFIED (orig: {original_tokens_test[i].argument:.4f})"
        
        arg_display = t_sab.argument
        if isinstance(t_sab.argument, Parameter): arg_display = t_sab.argument.name
        elif isinstance(t_sab.argument, ParameterExpression): arg_display = str(t_sab.argument)
        elif isinstance(t_sab.argument, float): arg_display = f"{t_sab.argument:.4f}"


        print(f"  Idx {i}: {t_sab.gate_type} on {t_sab.qubits}, arg: {arg_display}{change_indicator}")

