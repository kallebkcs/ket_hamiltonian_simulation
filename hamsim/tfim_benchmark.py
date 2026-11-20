from ket import *
from ket.qulib import gate_count
from ket.ibm import IBMDevice
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh as Provider
from hamsim import *
from exact_solver import *
'''
BENCHMARK DO MÓDULO DE EVOLUÇÃO HAMILTONIANA DO KET
Modelo: TFIM

- Seguindo a proposta de benchmark de https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10949677:
Para uma rede unidimensional de 2 a 10 spins, se calculará a magnetização média obtida após um segundo de simulação utilizando um
solver exato (que monta a matriz de evolução e resolve por álgebra linear), o módulo de evolução utilizando a primeira e a segunda ordem.
Parâmetros serão fixos:
m: 5
t: 1 u.a.
J: 0.1 u.a.
h: 1 u.a.

O objetivo é comparar a fidelidade dos resultados obtidos pelos solvers quânticos em relação ao solver exato.
'''

def create_1d_tfim(J: float, h: float, qubits: Quant) -> Hamiltonian:
    '''Considerando que já temos qubits de um Process alocados, retorna o hamiltoniano do TFIM'''
    n_spins = len(qubits)
    with obs():
        H = sum([-J * Z(qubits[i]) * Z(qubits[i+1]) for i in range(n_spins-1)])
        H += sum([-h * X(qubits[i]) for i in range(n_spins)])
    return H

def magnetizacao_media(state: Quant):
    '''Após a evolução, calcula a magnetização média do estado (média do valor esperado de Z em cada qubit)'''
    with obs():
        Z_spin = Z(state[0])
        for spin in state[1:]:
            Z_spin += Z(spin)
    return exp_value(Z_spin).get()/len(state)

def quantum_fidelity(state1, state2):
    '''Determina a fidelidade entre estados quânticos, dado pelo módulo ao quadrado do produto interno entre os dois vetores.
    Os estados aceitos são arrays unidimensionais do numpy'''
    return np.abs(np.conj(state2) @ state1)**2

def tfim_trotterization(J, h, t, m, order, q):
    H = create_1d_tfim(J, h, q)
    evolve_hamiltonian(H, t, order=order, steps=m)

def create_and_evolve(q, J, h, t, m, order):
    hamiltonian = create_1d_tfim(J, h, q)
    evolve_hamiltonian(hamiltonian, t, order=order, steps=m)

# Noisy simulation
def ibm_process():
    backend = Provider()
    simulator = AerSimulator.from_backend(backend)
    device = IBMDevice(simulator)
    return Process(device)

def simulacao_noisy(J: float, h: float, t: float, m: int, n_spins: int, print_results=False):
    # Auxiliar para exato
    auxp = Process()
    auxq = auxp.alloc(n_spins)
    H = create_1d_tfim(J, h, auxq)

    _, mag_exata = solve_tfim(H, t, auxq)
    results = (mag_exata,)

    for order in [1,2]:
        p = ibm_process()
        q = p.alloc(n_spins)
        H = create_1d_tfim(J, h, q)
        evolve_hamiltonian(H, t, order=order, steps=m)
        mag_sim = magnetizacao_media(q)
        cnots, depth = gate_count(tfim_trotterization, n_spins, (J, h, t, m, order))
        results += (mag_sim, cnots['cnot'], depth, '')

    # Retornos: magmed exata, [de 1 e de 2]: magmed simulada, número de cnots, profundidade de circuito, fidelidade de estado
    if print_results:
        print(f'''RESULTADOS DA SIMULAÇÃO
    Parâmetros:
        Número de spins: {n_spins};
        J: {J};
        h: {h};
        Tempo: {t};
        Passos de Trotter (m): {m}.
    
    Magnetizações:
        Exata: {results[0]:.3f}
        Primeira ordem: {results[1]:.3f}
        Segunda ordem: {results[5]:.3f}
    
    Resultados para primeira ordem:
        CNOTs: {results[2]};
        Profundidade: {results[3]};

    Resultados para segunda ordem:
        CNOTs: {results[6]};
        Profundidade: {results[7]};
    ''')
    return results

def simulacao(J: float, h: float, t: float, m: int, n_spins: int, print_results=False, noise=False):
    '''Gera o hamiltoniano com o número de spins especificado, executa a simulação exata e a quântica na ordem especificada e retorna a
    fidelidade'''
    if noise: 
        results = simulacao_noisy(J, h, t, m, n_spins, print_results)
        return results
    
    p = Process()
    q = p.alloc(n_spins)
    
    H = create_1d_tfim(J, h, q)

    estado_exato, mag_exata = solve_tfim(H, t, q)
    results = (mag_exata,)
    for order in [1,2]:
        p = Process()
        q = p.alloc(n_spins)
        H = create_1d_tfim(J, h, q)
        evolve_hamiltonian(H, t, order=order, steps=m)
        mag_sim = magnetizacao_media(q)
        cnots, depth = gate_count(tfim_trotterization, n_spins, (J, h, t, m, order))
        results += (mag_sim, cnots['cnot'], depth, quantum_fidelity(estado_exato, get_quantum_vector(q)))

    # Retornos: magmed exata, [de 1 e de 2]: magmed simulada, número de cnots, profundidade de circuito, fidelidade de estado
    if print_results:
        print(f'''RESULTADOS DA SIMULAÇÃO
    Parâmetros:
        Número de spins: {n_spins};
        J: {J};
        h: {h};
        Tempo: {t};
        Passos de Trotter (m): {m}.
    
    Magnetizações:
        Exata: {results[0]:.3f}
        Primeira ordem: {results[1]:.3f}
        Segunda ordem: {results[5]:.3f}
    
    Resultados para primeira ordem:
        CNOTs: {results[2]};
        Profundidade: {results[3]};
        Fidelidade de estado: {results[4]:.3f}

    Resultados para segunda ordem:
        CNOTs: {results[6]};
        Profundidade: {results[7]};
        Fidelidade de estado: {results[8]:.3f}
''')
    return results

def visualize(n_spins, J, h, t, m, order):
    p = Process()
    q = p.alloc(n_spins)
    visu = partial(create_and_evolve, J=J, h=h, t=t, m=m, order=order)
    fig = qulib.draw(visu, n_spins, output='mpl')
    fig.savefig(f'circuits/circuito_{J}{h}{t}{m}{n_spins}{order}.pdf')
    print(f'Imagem `circuito_{J}{h}{t}{m}{n_spins}{order}.pdf` gerada com sucesso.')