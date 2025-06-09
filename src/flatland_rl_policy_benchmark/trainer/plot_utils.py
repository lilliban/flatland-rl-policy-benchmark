import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# questa funzione fa un grafico per ogni file evolution_levelX.csv nella directory
def plot_evolution_results(save_dir):
    # Trova tutti i file evolution_levelX.csv nella directory
    files = [f for f in os.listdir(save_dir) if f.startswith("evolution_level") and f.endswith(".csv")]
    if not files:
        print("Nessun file di evoluzione trovato nella directory.")
        return

    for file in files:
        csv_path = os.path.join(save_dir, file)
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        # pulisce la colonna reward da evntuali NaN o valori non numerici
        df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
        df.dropna(subset=['reward'], inplace=True)

        # Liscia la curva
        df['raw_episode_reward'] = pd.to_numeric(df['raw_episode_reward'], errors='coerce')
        df.dropna(subset=['raw_episode_reward'], inplace=True)
        if 'shaped_episode_reward' in df.columns:
                df['smoothed_reward'] = df['shaped_episode_reward'].rolling(window=60, min_periods=1).mean()
                
                plt.figure(figsize=(10, 6))
                plt.plot(df['episode'], df['smoothed_reward'], label='Smoothed Shaped Reward')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title(f'Evolution - {file.replace(".csv", "")}')
                plt.grid(True)
                plt.tight_layout()
                output_path = os.path.join(save_dir, file.replace(".csv", ".png"))
                plt.savefig(output_path)
                plt.close()
                print(f"Grafico salvato in: {output_path}")
        else:
                print(f"⚠️  Il file {file} non contiene 'shaped_episode_reward'. Skipping.")
                continue
        
      


def plot_combined_results(save_dir):
    plot_combined_results_generic(save_dir, target_column='raw_episode_reward', suffix='raw')
    plot_combined_results_generic(save_dir, target_column='shaped_episode_reward', suffix='shaped')


def plot_combined_results_generic(save_dir, target_column='raw_episode_reward', suffix='raw'):
    best_path = os.path.join(save_dir, f"combined_parallel_training_{suffix}.png")
    if os.path.exists(best_path):
        os.remove(best_path)

    all_data = []
    for file in os.listdir(save_dir):
        if file.startswith("evolution_level") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(save_dir, file))
            df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
            df.dropna(subset=['reward'], inplace=True)
            df['level'] = file.split('_')[1].replace('level', '')

            # Verifica se il target_column esiste
            if target_column not in df.columns:
                print(f"⚠️ Il file {file} non contiene '{target_column}'. Skipping.")
                continue

            df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
            df.dropna(subset=[target_column], inplace=True)

            all_data.append(df)

    if not all_data:
        print(f"Nessun file valido trovato per la combinazione per '{target_column}'.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['global_episode'] = combined_df.groupby(['level', 'branch']).cumcount() + 1

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    palette = sns.color_palette("Set2", combined_df['level'].nunique())
    line_styles = {
        'B0': '-', 'B1': '--', 'B2': '-.', 'B3': ':', 'B4': (0, (5, 10)),
    }

    for i, (level, group) in enumerate(combined_df.groupby('level')):
        for branch in group['branch'].unique():
            branch_df = group[group['branch'] == branch]
            branch_key = branch.split('_')[-1]

            plt.plot(branch_df['global_episode'],
                     branch_df[target_column].rolling(60, min_periods=1).mean(),
                     color=palette[i],
                     linestyle=line_styles.get(branch_key, '-'),
                     label=f"Level {level} - {branch}")

    plt.title(f'Combined Evolutionary Training Progress ({target_column})', fontsize=14)
    plt.xlabel('Global Episode Number', fontsize=12)
    plt.ylabel(f'Smoothed {target_column} (window=60)', fontsize=12)
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(best_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafico combinato generato: {best_path}")
