import os
from multiprocessing import Pool
from flatland_rl_policy_benchmark.trainer.train_dddqn import train
from flatland_rl_policy_benchmark.trainer.plot_utils import plot_evolution_results, plot_combined_results

# Configurazione
BASE_SAVE_DIR = "results"
ROUNDS_PER_LEVEL = 10
N_AGENTS = 4
GRID_SIZE = 25

def run_training(args):
    """Esegue il training per un singolo branch (per parallel processing)"""
    level, branch, parent_branch = args
    branch_name = f"L{level}_B{branch}"
    parent_path = os.path.join(BASE_SAVE_DIR, f"best_model_L{level-1}_B{parent_branch}.pt") if parent_branch is not None else None
    
    print(f"\nTraining {branch_name} (Parent: {parent_path or 'None'})")
    
    rewards = train(
        round_start=level * ROUNDS_PER_LEVEL,
        n_rounds=ROUNDS_PER_LEVEL,
        branch_name=branch_name,
        parent_path=parent_path,
        save_dir=BASE_SAVE_DIR,
        level=level,
        width=GRID_SIZE,
        height=GRID_SIZE,
        n_agents=N_AGENTS
    )
    return (level, branch, rewards)

def learning_speed(reward_list, window=100):
    """Calcola la velocità di apprendimento come derivata delle reward"""
    if len(reward_list) < window:
        return -float('inf')
    return (reward_list[-1] - reward_list[0]) / window  # Pendenza media

def main():
    """Flusso principale dell'esperimento evolutivo"""
    print("\n⭐️ Avvio esperimento evolutivo ⭐️\n")
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # Configurazione dell'albero evolutivo
    tree_structure = {
        0: [('0', None)],               # Livello 0 (radice)
        1: [('0', '0'), ('1', '0')],    # Livello 1 (2 figli)
        2: [('0', '0'), ('1', '0'),     # Livello 2 (4 figli)
            ('2', '1'), ('3', '1')]     
    }

    # Esecuzione parallela dei livelli
    with Pool(processes=4) as pool:  # 4 processi in parallelo
        results = []
        
        # Livello 0 (radice)
        print("\n=== LIVELLO 0 - Training radice ===")
        level_results = pool.map(run_training, [(0, '0', None)])
        results.extend(level_results)
        
        # Livello 1 (2 branches)
        print("\n=== LIVELLO 1 - Training 2 branches ===")
        level_results = pool.map(run_training, [(1, '0', '0'), (1, '1', '0')])
        results.extend(level_results)
        
        # Trova il migliore del livello 1
        best_l1 = max(level_results, key=lambda x: learning_speed(x[2]))[1]
        print(f"\nMiglior branch livello 1: {best_l1}")
        
        # Livello 2 (4 branches)
        print("\n=== LIVELLO 2 - Training 4 branches ===")
        parent_branches = ['0', '1']  # Branches del livello 1
        level_args = []
        
        for parent in parent_branches:
            level_args.extend([(2, str(i), parent) for i in range(2)])  # 2 figli per parent
        
        level_results = pool.map(run_training, level_args)
        results.extend(level_results)

    # Genera i grafici finali
    print("\nGenerazione dei grafici...")
    plot_evolution_results(BASE_SAVE_DIR)
    plot_combined_results(BASE_SAVE_DIR)
    
    print("\n⭐️ Esperimento completato con successo! ⭐️")

if __name__ == "__main__":
    main()