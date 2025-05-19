
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dddqn_trend(csv_path="tournament_results.csv", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f" File non trovato: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    dddqn_df = df[df["policy"] == "DDDQN"].sort_values(by="round")
    dddqn_df["reward_moving_avg"] = dddqn_df["total_reward"].rolling(window=3, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(dddqn_df["round"], dddqn_df["total_reward"], label="Reward", alpha=0.5, marker='o', color='orange')
    plt.plot(dddqn_df["round"], dddqn_df["reward_moving_avg"], label="Media mobile (3)", linewidth=2, color='red')
    plt.title("Andamento della ricompensa DDDQN per round")
    plt.xlabel("Round")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_file = os.path.join(output_dir, "plot_dddqn_trend.png")
    plt.savefig(output_file)
    plt.close()

    print(f" Grafico salvato in '{output_file}'")

if __name__ == "__main__":
    plot_dddqn_trend()
