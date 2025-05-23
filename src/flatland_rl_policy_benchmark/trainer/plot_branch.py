import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_branch(branch_name, save_dir="results"):
    log_path = os.path.join(save_dir, f"log_{branch_name}.csv")
    if not os.path.exists(log_path):
        print(f" File non trovato: {log_path}")
        return

    df = pd.read_csv(log_path)
    df['episode_global'] = (df['round'] - 1) * 1000 + df['episode']
    df['reward_smooth'] = df['reward'].rolling(window=50, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='episode_global', y='reward', label='Reward', alpha=0.3)
    sns.lineplot(data=df, x='episode_global', y='reward_smooth', label='Smoothed (window=50)')
    plt.title(f"Reward Trend – {branch_name}")
    plt.xlabel("Global Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"reward_curve_{branch_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f" Grafico salvato: {out_path}")

if __name__ == "__main__":
    # ESEMPIO USO
    plot_branch("L2_B3")
