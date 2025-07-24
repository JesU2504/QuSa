from typing import List, Dict
# Assuming QuantumToken is accessible (as above)
from Architect.architect import QuantumToken
from qiskit.circuit import Parameter, ParameterExpression

def apply_sabotage_instruction(
    original_tokens: List[QuantumToken],
    sabotage_instruction: Dict # Expected: {'target_token_idx': int, 'noise_val': float}
    ) -> List[QuantumToken]:
    """
    Applies the sabotage instruction (noise) to a copy of the tokens.
    Can now add noise to float, int, Parameter, or ParameterExpression arguments.
    """
    sabotaged_tokens = [token.copy() for token in original_tokens]

    target_original_idx = sabotage_instruction['target_token_idx']
    noise = sabotage_instruction['noise_val'] # This is a float

    #print(f"DEBUG (apply_sabotage): Attempting to modify token at original_idx {target_original_idx} with noise {noise:.3f}.")

    if 0 <= target_original_idx < len(sabotaged_tokens):
        token_to_modify = sabotaged_tokens[target_original_idx]
        
        # print(f"DEBUG (apply_sabotage): Token to modify (before): "
        #       f"type={token_to_modify.gate_type}, "
        #       f"qubits={token_to_modify.qubits}, "
        #       f"arg_type={type(token_to_modify.argument)}, "
        #       f"arg_value='{token_to_modify.argument}'")

        if token_to_modify.argument is not None:
            original_arg = token_to_modify.argument
            if isinstance(original_arg, (float, int, Parameter, ParameterExpression)):
                # Qiskit Parameter and ParameterExpression support addition with floats,
                # resulting in a new ParameterExpression.
                try:
                    token_to_modify.argument = original_arg + noise
                    # print(f"DEBUG (apply_sabotage): SUCCESS - Modified token argument. "
                    #       f"Original arg: '{original_arg}', New arg: '{token_to_modify.argument}'")
                except Exception as e:
                    pass
                    # print(f"DEBUG (apply_sabotage): FAILED during modification - {e}. "
                    #       f"Arg type: {type(original_arg)}, Value: '{original_arg}'")
            # else:
                # print(f"DEBUG (apply_sabotage): SKIPPED - Token argument is not a modifiable type (float, int, Parameter, ParameterExpression). "
                #       f"Type: {type(original_arg)}, Value: '{original_arg}'")
        # else:
            # print(f"DEBUG (apply_sabotage): SKIPPED - Token has no argument to modify.")
    # else:
        # print(f"DEBUG (apply_sabotage): FAILED - Target original index {target_original_idx} out of bounds.")
    return sabotaged_tokens