from ket import *
import numpy as np
from itertools import combinations

################ DISPONÍVEIS NO KET #########################
_ROT_MAP = {
    "X": H,
    "Y": RX(np.pi / 2),
    "Z": I,
}

def _rot(string, qubits):
    for p, q in zip(string, qubits):
        _ROT_MAP[p](q)


def _rzg(angle, qubits):
    if len(qubits) == 1:
        RZ(angle, qubits)
    else:
        with around(CNOT, qubits[0], qubits[1]):
            _rzg(angle, qubits[1:])


_RGATE = {"X": RX, "Y": RY, "Z": RZ}
#############################################################

def evolve_directly(hamiltonian: Hamiltonian, t: float = 1, reverse=False):
    """Time evolution :math:`e^{-iHt}` for an unitary or commutative 
    Hamiltonian.
    
    For each Pauli term :math:`c_k P_k` in the Hamiltonian, applies the
    corresponding single-step Trotter evolution gate
    :math:`e^{-i c_k P_k}` to the qubits in that term.
    """
    process = hamiltonian.ket_process

    terms = hamiltonian.terms[:]
    if reverse: terms.reverse()
    for term in terms:
        gates_qubits = [
            (gate, qi) for qi, gate in term.map.items() if gate != "I"
        ]

        if not gates_qubits:
            continue

        gates, qubits = zip(*gates_qubits)
        qubits = Quant(qubits=list(qubits), process=process)

        if len(gates) == 1:
            # Se for 1-local, apenas aplique a porta de rotação correspondente
            _RGATE[gates[0]](2 * term.coef * t, qubits)
        else:
            # Se for 2+-local, aplique uma porta de transformação correspondente para formular a porta de rotação de maior localidade
            with around(_rot, gates, qubits):
                _rzg(2 * term.coef * t, qubits)

def commutative(h: Hamiltonian):
    '''Verifies if the hamiltonian terms are commutative'''
    #OBS: Vamos considerar que todo Hamiltonian é uma combinação linear de strings de Pauli
    #Considerando isso, a verificação de comutatividade é tranquila, visto que portas de Pauli comutam consigo e com I, e anticomutam com as demais
    #Se o número de anticomutações for par, as strings comutam.
    gates_dict = [term.map for term in h.terms]
    for term1, term2 in combinations(gates_dict, 2):
        common_qubit = term1.keys() & term2.keys() # Interseção de chaves
        anticommutatives = sum(1 for k in common_qubit if term1[k] != term2[k]) # Número de interações anticomutativas
        if anticommutatives%2 != 0:
            return False
    return True

def evolve_trotter_suzuki(h: Hamiltonian, t: float, order: int, steps: int, error: float):
    '''Evolves a Hamiltonian by Trotter-Suzuki's `order`
    product formula with `steps` steps or a tolerated
    `error` error'''
    if steps == 0:
        # Sendo legal comigo mesmo, vou estimar um bom número de passos considerando norma-1
        # Tem um jeito mais preciso que envolve as comutações, mas é meio bizarrinho
        # O norma-1 leva em consideração a soma do módulo dos coeficientes para que o
        # número de passos ótimo seja em torno de: 
        steps = int(np.ceil(((t*sum(abs(term.coef) for term in h.terms))**(1+(1/order)))/(error**(1/order))))
        print(f'Defined {steps} Trotter steps')
    ts = t/(order*steps)
    for _ in range(steps): 
        evolve_directly(h, ts)
        if order == 2: evolve_directly(h, ts, reverse=True)

def evolve_hamiltonian(h: Hamiltonian, t: float, order=1, steps=0, error=0.01):
    '''Time evolution :math:`e^{-iHt}` for a Hamiltonian.
    
    Identifies unitary and comutative terms hamiltonians 
    and applies directly Pauli's string exponentiation in 
    these cases.
    If the hamiltonian's terms are not commutative, a 
    trotterization needs to be applied to execute the 
    approximate evolution.

    Params:
    h: Hamiltonian -> Hamiltonian of the evolution
    qubits: Quant -> State to be evoluted
    t: float -> Time of evolution

    (For non-commutative simulation)
    order: int = 1 -> Defines the product formula's 
    order. Can be 1 or 2.
    steps: int = 0 -> The number of Trotter steps. If 0, 
    we calculate a theoretically optimal number based on 
    `error`and 1-norm.
    error: float = 0.01 -> maximum error tolerated for 
    the simulation.
    '''
    if commutative(h):
        evolve_directly(h, t)
    else:
        evolve_trotter_suzuki(h, t, order=order, steps=steps, error=error)