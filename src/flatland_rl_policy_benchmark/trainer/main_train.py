import os
from flatland_rl_policy_benchmark.trainer.train_dddqn import train
from flatland_rl_policy_benchmark.trainer.plot_branch import plot_branch

BASE_SAVE_DIR = "results"
ROUNDS_PER_LEVEL = 10

def get_model_path(level, branch):
    return os.path.join(BASE_SAVE_DIR, f"model_L{level}_B{branch}.pt")

def run_training(level, branch, parent_branch=None):
    branch_name = f"L{level}_B{branch}"
    parent_path = get_model_path(level - 1, parent_branch) if parent_branch is not None else ""

    print(f"\n Training {branch_name} (Parent: {parent_path or 'None'})")
    if parent_path:
        print(f"✔️ Branch {branch_name} inizializzato con pesi ereditati da {parent_path}")

    rewards = train(
        round_start=level * ROUNDS_PER_LEVEL,
        n_rounds=ROUNDS_PER_LEVEL,
        branch_name=branch_name,
        parent_path=parent_path,
        save_dir=BASE_SAVE_DIR
    )
    plot_branch(branch_name, save_dir=BASE_SAVE_DIR)
    return rewards

def learning_speed(reward_list, window=100):
    y = reward_list[-window:]
    diffs = [y[i+1] - y[i] for i in range(len(y)-1)]
    return sum(diffs) / len(diffs) if diffs else -float("inf")

if __name__ == "__main__":
    # LIVELLO 0 (root)
    run_training(0, "0")

    # LIVELLO 1 (2 figli del root)
    rewards_l1 = {}
    for b in ["0", "1"]:
        rewards_l1[b] = run_training(1, b, parent_branch="0")

    # Selezione automatica del migliore tra B0 e B1 tramite derivata
    best_l1 = max(rewards_l1, key=lambda b: learning_speed(rewards_l1[b]))
    print(f"\n Miglior branch livello 1 (più veloce ad apprendere): B{best_l1}")

    # LIVELLO 2 (4 figli del migliore del livello 1)
    for b in ["0", "1", "2", "3"]:
        run_training(2, b, parent_branch=best_l1)

    print("\n⭐️ Esperimento evolutivo completato")
