# In Architect/architect.py

import dataclasses
import math
import random 
from typing import Optional, Union, List, Tuple, Dict 
import copy

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterExpression 
from qiskit.circuit.library import (
    HGate, XGate, YGate, ZGate, SGate, TGate,
    RXGate, RYGate, RZGate, PhaseGate, UGate, 
    CXGate, CYGate, CZGate, CRXGate, CRYGate, CRZGate, CPhaseGate, CUGate, 
    SwapGate, CSwapGate, RZZGate
)

@dataclasses.dataclass
class QuantumToken:
    gate_type: str
    qubits: Union[int, List[int]]
    argument: Optional[Union[float, str, Parameter, ParameterExpression]] = None

    def __post_init__(self):
        if isinstance(self.qubits, int): self.qubits = [self.qubits]
        elif not isinstance(self.qubits, list): raise TypeError("Qubits must be an int or list.")

    def copy(self): 
        """Creates a deep copy of this QuantumToken."""
        # Using copy.deepcopy is robust for various attribute types,
        # including ParameterExpression.
        return copy.deepcopy(self)
    
    @property
    def is_single_qubit(self) -> bool: return len(self.qubits) == 1
    @property
    def is_double_qubit(self) -> bool: return len(self.qubits) == 2
    @property
    def control_qubit(self) -> Optional[int]: return self.qubits[0] if self.is_double_qubit else None
    @property
    def target_qubit(self) -> Optional[int]:
        if self.is_double_qubit: return self.qubits[1]
        elif self.is_single_qubit: return self.qubits[0]
        return None


def create_quantum_circuit_from_tokens(
    num_qubits: int, tokens: List[QuantumToken],
    num_classical_bits: Optional[int] = 0, verb: bool = False
) -> Optional[QuantumCircuit]:
    if num_qubits <= 0: return QuantumCircuit() 
    qr = QuantumRegister(num_qubits, 'q')
    cr = None
    if num_classical_bits is not None and num_classical_bits > 0:
        cr = ClassicalRegister(num_classical_bits, 'c'); qc = QuantumCircuit(qr, cr)
    else: qc = QuantumCircuit(qr)
    gate_map = {
        'h': qc.h, 'x': qc.x, 'y': qc.y, 'z': qc.z, 's': qc.s, 't': qc.t,
        'rx': qc.rx, 'ry': qc.ry, 'rz': qc.rz, 'u1': qc.p, 
        'measure': qc.measure, 'cx': qc.cx, 'cy': qc.cy, 'cz': qc.cz,
        'crx': qc.crx, 'cry': qc.cry, 'crz': qc.crz, 'cu1': qc.cp, 
        'swap': qc.swap, 'rzz': qc.rzz, 'cswap': qc.cswap
    }
    for i, token in enumerate(tokens):
        gate_func = gate_map.get(token.gate_type.lower())
        if not gate_func: 
            if verb: print(f"Verbose: Unknown gate '{token.gate_type}' (idx {i}). Skip.")
            continue
        valid_qubits = all(0 <= q_idx < num_qubits for q_idx in token.qubits)
        if not valid_qubits: 
            if verb: print(f"Verbose: Qubit idx out of range for gate '{token.gate_type}' (idx {i}). Skip.")
            continue
        try:
            num_token_qubits = len(token.qubits)
            if token.gate_type.lower() == 'measure':
                if cr is None: 
                    if verb: print(f"Verbose: 'measure' token (idx {i}), no clbits. Skip."); 
                    continue
                for qb_idx in token.qubits: # Assuming token.qubits contains indices of qbits to measure
                    if 0 <= qb_idx < cr.size: # And map to classical bit of same index
                        qc.measure(qr[qb_idx], cr[qb_idx])
            elif token.argument is not None: 
                if num_token_qubits == 1: gate_func(token.argument, qr[token.qubits[0]])
                elif num_token_qubits == 2: gate_func(token.argument, qr[token.qubits[0]], qr[token.qubits[1]])
                else: gate_args = [token.argument] + [qr[q] for q in token.qubits]; gate_func(*gate_args)
            else: 
                if num_token_qubits == 1: gate_func(qr[token.qubits[0]])
                elif num_token_qubits == 2: gate_func(qr[token.qubits[0]], qr[token.qubits[1]])
                elif num_token_qubits == 3 and token.gate_type.lower() == 'cswap':
                    gate_func(qr[token.qubits[0]], qr[token.qubits[1]], qr[token.qubits[2]])
                else: gate_args = [qr[q] for q in token.qubits]; gate_func(*gate_args)
        except Exception as e:
            if verb: print(f"Verbose Error applying gate '{token.gate_type}' (idx {i}): {e}")
            continue
    return qc

def _parse_angle_string(angle_str: str) -> float:
    if not isinstance(angle_str, str): raise TypeError(f"Expected str, got {type(angle_str)}.")
    lower_str = angle_str.lower().strip(); is_negative = False
    if lower_str.startswith('-'): is_negative = True; lower_str = lower_str[1:]
    if 'pi' in lower_str:
        val = 0.0
        if '/' in lower_str:
            parts = lower_str.split('/');
            if len(parts)==2 and parts[0].strip()=='pi':
                try: den = float(parts[1].strip()); val = math.pi/den if den!=0 else float('inf')
                except: raise ValueError(f"Invalid pi expr: '{angle_str}'.")
            else: raise ValueError(f"Unrec pi expr: '{angle_str}'.")
        elif lower_str=='pi': val=math.pi
        else: raise ValueError(f"Unrec pi str: '{angle_str}'.")
        return -val if is_negative else val
    try: return float(angle_str)
    except: raise ValueError(f"Unrec angle str: '{angle_str}'.")

def generate_token_pool(
    gates: List[str], args_in: List[Union[float, str, Parameter, ParameterExpression]], num_q: int
) -> List[QuantumToken]:
    pool: List[QuantumToken] = []; 
    if num_q <= 0: return pool
    p_sgl={'rx','ry','rz','u1'}; np_sgl={'h','x','y','z','s','t'}
    p_two={'rzz','crx','cry','crz','cu1'}; np_two={'cx','cy','cz','swap'}
    np_three={'cswap'}
    cache: Dict[str, Parameter]={}; 
    def get_p(name:str)->Parameter: 
        if name not in cache: cache[name]=Parameter(name)
        return cache[name]
    proc_args: List[Union[float,Parameter,ParameterExpression]]=[]
    for arg_i in args_in:
        if isinstance(arg_i,(Parameter,ParameterExpression)): proc_args.append(arg_i)
        elif isinstance(arg_i,str):
            neg=arg_i.startswith('-'); name=arg_i[1:] if neg else arg_i
            if name.replace('_','').isalnum() and not name.isnumeric() and not name.replace('.','',1).isdigit():
                p_obj=get_p(name); proc_args.append(-p_obj if neg else p_obj)
            else: 
                try: proc_args.append(_parse_angle_string(arg_i))
                except: pass
        elif isinstance(arg_i,(float,int)): proc_args.append(float(arg_i))
    for gt in gates:
        gtl=gt.lower()
        if gtl in p_sgl or gtl in np_sgl:
            for q_idx in range(num_q):
                if gtl in p_sgl: [pool.append(QuantumToken(gtl,q_idx,arg)) for arg in proc_args]
                else: pool.append(QuantumToken(gtl,q_idx))
        elif gtl in p_two or gtl in np_two:
            if num_q<2:continue
            for q1 in range(num_q):
                for q2 in range(num_q):
                    if q1==q2:continue
                    if gtl in p_two: [pool.append(QuantumToken(gtl,[q1,q2],arg)) for arg in proc_args]
                    else: pool.append(QuantumToken(gtl,[q1,q2]))
        elif gtl in np_three:
            if num_q<3:continue
            for q1 in range(num_q):
                for q2 in range(num_q):
                    for q3 in range(num_q):
                        if q1==q2 or q1==q3 or q2==q3:continue
                        pool.append(QuantumToken(gtl,[q1,q2,q3]))
        elif gtl=='measure': [pool.append(QuantumToken(gtl,q_idx)) for q_idx in range(num_q)]
    return pool

class ArchitectAgent:
    def __init__(self, num_qubits: int, allowed_gates: List[str],
                 allowed_arguments: List[Union[float, str, Parameter, ParameterExpression]],
                 token_sequence_length: int, backend=None, device_properties=None): 
        self.num_qubits = num_qubits
        self.token_sequence_length = token_sequence_length
        self.global_token_pool = generate_token_pool(allowed_gates, allowed_arguments, num_qubits)
        self.num_total_tokens_in_pool = len(self.global_token_pool)
        if self.num_total_tokens_in_pool == 0 and self.num_qubits > 0 : 
            raise ValueError("ArchitectAgent: Global token pool is empty.")
        print(f"Architect initialized: {self.num_total_tokens_in_pool} ops in pool for {self.num_qubits} qubits.")

    def _generate_random_identity_list(self) -> List[int]:
        if self.num_total_tokens_in_pool == 0: return []
        return random.choices(range(self.num_total_tokens_in_pool), k=self.token_sequence_length)

    def propose_circuit_from_identities(
        self, identity_list: List[int], add_measurements: bool = False
    ) -> Tuple[Optional[QuantumCircuit], List[QuantumToken], List[Union[Parameter, ParameterExpression]], List[float]]: # MODIFIED return signature
        """
        Proposes a Qiskit QuantumCircuit based on token identities.
        Returns circuit, the list of selected QuantumToken objects, 
        its Qiskit Parameter objects, and random initial values for them.
        """
        selected_tokens: List[QuantumToken] = []
        for identity in identity_list:
            if 0 <= identity < len(self.global_token_pool):
                selected_tokens.append(self.global_token_pool[identity])
        
        num_cl_bits_for_creation = self.num_qubits if add_measurements and self.num_qubits > 0 else 0
        qc = create_quantum_circuit_from_tokens(
            self.num_qubits, selected_tokens, num_classical_bits=num_cl_bits_for_creation, verb=False # Set verb=True for debugging token application
        )
        if qc is None: return None, [], [], []

        if add_measurements and self.num_qubits > 0: 
            all_qubits_measured = False
            if qc.num_clbits >= self.num_qubits:
                measured_q_indices = set()
                for inst in qc.data:
                    if inst.operation.name == 'measure':
                        for qbit_obj in inst.qubits: 
                            try: measured_q_indices.add(qc.find_bit(qbit_obj).index)
                            except: pass 
                if len(measured_q_indices) == self.num_qubits: all_qubits_measured = True
            if not all_qubits_measured:
                measure_all_creg_name = 'c_measure_all_arch' 
                existing_creg = next((reg for reg in qc.cregs if reg.name == measure_all_creg_name and reg.size == self.num_qubits), None)
                if not existing_creg:
                    qc.cregs = [reg for reg in qc.cregs if reg.name != measure_all_creg_name] 
                    cr_measure = ClassicalRegister(self.num_qubits, measure_all_creg_name)
                    qc.add_register(cr_measure)
                # Ensure measurement is to the correct register
                target_measure_reg = next(reg for reg in qc.cregs if reg.name == measure_all_creg_name and reg.size == self.num_qubits)
                qc.measure(range(self.num_qubits), target_measure_reg)


        circuit_parameters_objects: List[Union[Parameter, ParameterExpression]] = [] # Should be List[Parameter]
        unique_params_in_circuit = set() 
        for token in selected_tokens: 
            if isinstance(token.argument, Parameter):
                unique_params_in_circuit.add(token.argument)
            elif isinstance(token.argument, ParameterExpression):
                for p_expr_component in token.argument.parameters: 
                    unique_params_in_circuit.add(p_expr_component)
        
        circuit_parameters_objects = sorted(list(unique_params_in_circuit), key=lambda p: p.name)
        initial_param_values = [random.uniform(0, 2 * math.pi) for _ in circuit_parameters_objects]
        
        return qc, selected_tokens, circuit_parameters_objects, initial_param_values

    def propose_random_circuit(
        self, add_measurements: bool = False
    ) -> Tuple[Optional[QuantumCircuit], List[QuantumToken], List[Union[Parameter, ParameterExpression]], List[float]]: # MODIFIED return signature
        random_identity_list = self._generate_random_identity_list()
        return self.propose_circuit_from_identities(random_identity_list, add_measurements=add_measurements)

if __name__ == '__main__':
    print("--- Testing ArchitectAgent (Return Tokens) ---")
    num_q_test = 2
    allowed_g_test = ['rx', 'ry', 'h', 'cx', 'u1', 'measure'] 
    feature_p_names_test = [f'p{i}' for i in range(num_q_test)] 
    var_w_names_test = [f'w{i}' for i in range(2)]    
    allowed_a_test = feature_p_names_test + var_w_names_test + [math.pi / 2, "-pi/4", "pi"] 
    token_len_test = 6

    architect_test = ArchitectAgent(num_q_test, allowed_g_test, allowed_a_test, token_len_test)
    
    print(f"\nGlobal token pool (first 10 tokens if available):")
    for i, token_item in enumerate(architect_test.global_token_pool[:10]):
        arg_str_item = token_item.argument
        if isinstance(token_item.argument, Parameter): arg_str_item = token_item.argument.name
        elif isinstance(token_item.argument, ParameterExpression): arg_str_item = str(token_item.argument)
        print(f"  Token {i}: type={token_item.gate_type}, qubits={token_item.qubits}, arg={arg_str_item}")

    print("\nProposing a random circuit (no measurements by architect):")
    qc_rand_test, selected_tokens_test, qc_params_test, initial_vals_test = architect_test.propose_random_circuit(add_measurements=False)
    if qc_rand_test:
        print("Circuit:")
        print(qc_rand_test.draw(output='text'))
        print("\nSelected Tokens:")
        for t in selected_tokens_test: print(f"  {t.gate_type} on {t.qubits}, arg: {t.argument}")
        print(f"\nParameters in this circuit: {[p.name for p in qc_params_test]}")
        print(f"Suggested initial values for these params: {initial_vals_test}")

    print("\nProposing a random circuit WITH measurements added by architect:")
    qc_rand_meas_test, _, _, _ = architect_test.propose_random_circuit(add_measurements=True)
    if qc_rand_meas_test:
        print(qc_rand_meas_test.draw(output='text'))
