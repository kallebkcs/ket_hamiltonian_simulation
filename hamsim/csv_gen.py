import csv
import os
import random
import math
from tfim_benchmark import *
from itertools import product
from tqdm import tqdm

PLAN_FILE = "plan.csv"
RESULT_FILE = "simulation_results.csv"

# Parâmetro fixo
J = 1
# Parâmetros variados
list_h = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6]
list_m = list(range(2,21,2)) + list(range(25,55,5)) + list(range(55,105,10)) + list(range(120,240,20)) + list(range(250, 550, 50)) 
list_t = [0.5] + list(range(1,21)) + list(range(22,52,2))
list_spins = range(2,11)

PARAMS = {
    "h": list_h,
    "t": list_t,
    "m": list_m,
    "n_spins": list_spins,
    "noise": [0] # incluir 1 apenas em simulações reduzidas
}

def generate_plan(plan_file, params):
    keys = list(params.keys())
    combs = list(product(*params.values()))

    if not os.path.exists(plan_file):
        random.shuffle(combs)
        with open(plan_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(combs)
        print(f'Plano criado do zero com {len(combs)} combinações.')
        return

    planned = set()
    with open(plan_file, 'r') as f:
        reader = csv.reader(f)
        old_header = next(reader)
        for row in reader:
            planned.add(tuple(row))

    new_comb = []
    for comb in combs:
        comb_str = tuple(str(x) for x in comb)
        if comb_str[:len(old_header)] not in planned:
            new_comb.append(comb)

    if not new_comb:
        print('Nenhuma simulação nova encontrada.')
        return

    random.shuffle(new_comb)
    with open(plan_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_comb)

    print(f'Plano atualizado com {len(new_comb)} novas combinações.')

def count_results(result_file):
    if not os.path.exists(result_file):
        return 0
    with open(result_file, 'r') as f:
        return sum(1 for _ in f) - 1

def cleaner(row):
    clear_row = {}
    for k, v in row.items():
        try:
            clear_row[k] = float(v) if '.' in v else int(v)
        except ValueError:
            clear_row[k] = v
    return clear_row


def batch_simulation(plan_file, result_file, params):
    result_header = ["magnetizacao_exata", "magnetizacao_simulada_1", "cnots_1", "profundidade_1", "fidelidade_1", "magnetizacao_simulada_2", "cnots_2", "profundidade_2", "fidelidade_2"]
    
    keys = list(params.keys())
    header = keys + result_header
    generate_plan(plan_file, params)
    already_used = count_results(result_file)
    
    write_mode = 'a' if already_used > 0 else 'w'
    with open(result_file, write_mode, newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        
        if already_used == 0:
            writer.writeheader()

        total_rows = math.prod(len(v) for v in params.values())
        
        with open(plan_file, 'r') as f_in:
            reader = csv.DictReader(f_in)
            bar = tqdm(total=total_rows, initial=already_used, desc="Progresso")
            for i, row in enumerate(reader):
                if i < already_used:
                    continue 
                try:
                    clean_row = cleaner(row)
                    result = simulacao(J, **clean_row)
                    result_row = {**row} | {r: result[i] for i, r in enumerate(result_header)}
                    writer.writerow(result_row)
                    f_out.flush()
                    os.fsync(f_out.fileno())
                    bar.update(1)
                    
                except Exception as err:
                    print(f"Simulação interrompida em {row}: {err}")
            bar.close()

if __name__ == "__main__":
    batch_simulation(PLAN_FILE, RESULT_FILE, PARAMS)
    print("We did it...")