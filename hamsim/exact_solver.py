import numpy as np
from scipy.sparse.linalg import expm_multiply
from ket import *

pI = np.array([[1, 0], [0, 1]])
pX = np.array([[0, 1], [1, 0]])
pY = np.array([[0,-1j], [+1j,0]])
pZ = np.array([[1, 0], [0, -1]])
pauli_map = {'I': pI, 'X': pX, 'Y': pY, 'Z': pZ}

def pauli_strings(hamiltonian: Hamiltonian, num_qubits: int):
    coef = []
    pauli_str = []

    for term in hamiltonian.terms:
        p = ["I"] * num_qubits
        for qubit, pauli in term.map.items():
            p[qubit] = pauli
        coef.append(term.coef)
        pauli_str.append("".join(p))
    return pauli_str, coef

def operator_from_paulistr(paulistr):
    if len(paulistr) == 2:
        return np.kron(pauli_map[paulistr[0]],pauli_map[paulistr[1]])
    return np.kron(pauli_map[paulistr[0]], operator_from_paulistr(paulistr[1:]))

def get_observable_Z(state):
    num_qubits = len(state)
    # queremos a magnetização média = média do valor esperado do observável Z em cada qubit = ZII... + IZI... + ... + ...IIZ
    generic_paulistr = 'I' * num_qubits
    z_paulistr = [operator_from_paulistr(generic_paulistr[:i] + 'Z' + generic_paulistr[i+1:]) for i in range(num_qubits)]
    return sum(z_paulistr)

def get_exact_unitary(hamiltonian: Hamiltonian, t: float, state: Quant=[]):
    # se quiser salvar processamento vc passa o estado já e assumimos que ele tenha a mesma dimensionalidade do hamiltoniano
    if state:
        num_qubits = len(state)
    else:
        num_qubits = max([max(term.map.keys()) for term in hamiltonian.terms]) + 1

    pauli_str, coef = pauli_strings(hamiltonian, num_qubits)
    H = np.zeros((2**num_qubits,2**num_qubits))
    for i, paulistr in enumerate(pauli_str):
        H += coef[i] * operator_from_paulistr(paulistr)
    # H = hamiltoniano

    # U = operador unitário para evolução do hamiltoniano em t    
    U = expm(-1j * H * t)
    return U

def get_quantum_vector(state: Quant):
    state_dict = dump(state).states
    state_vector = np.zeros(2**len(state), dtype=complex)
    state_vector[list(state_dict.keys())] = list(state_dict.values())
    return state_vector

def solve_tfim(hamiltonian: Hamiltonian, t: float, state: Quant):
    '''Exact mean magnetization solver of the TFIM'''
    num_qubits = len(state)

    state_vector = get_quantum_vector(state)
    # state_vector = estado inicial

    pauli_str, coef = pauli_strings(hamiltonian, num_qubits)
    H = np.zeros((2**num_qubits,2**num_qubits))
    for i, paulistr in enumerate(pauli_str):
        H += coef[i] * operator_from_paulistr(paulistr)
    # H = hamiltoniano

    final_state = expm_multiply(-1j * H * t, state_vector)

    # Valor esperado
    observable_Z = get_observable_Z(state)
    total_expval = np.vdot(final_state, observable_Z @ final_state)
    magmed = np.real(total_expval)/num_qubits
    return final_state, magmed

if __name__ == '__main__':
    pr2 = Process()
    q2 = pr2.alloc(2)
    with obs():
        H2 = -1.0 * Z(q2[0]) * Z(q2[1]) - 0.5 * X(q2[0]) - 0.5 * X(q2[1])
    print(solve_tfim(H2, 1, q2))