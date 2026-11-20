# Ket Hamiltonian Simulation

Este repositório contém: 
- Arquivo _Jupyter Notebook_ referente à experimentos associados à simulação quântica do hamiltoniano de Ising de campo tranverso realizados com a plataforma de programação quântica Ket
- Diretório `hamsim`, que contém o código que generaliza a simulação hamiltoniana com a sintaxe de Ket, bem como os arquivos associados ao benchmarking realizado para sua validação.

## Instruções de Uso
- Instale as bibliotecas requeridas: `pip install -r requirements.txt`
- Simulação rápida: `py sim.py n_spins h t m order` ou `py sim.py` e siga as instruções
- Geração de CSV: Altere as variáveis globais `PLAN_FILE`, `RESULT_FILE` e `PARAMS` (essa principalmente) como quiser e, então? `py csv_gen.py`

**Autor**: Kalleb C. Santos
