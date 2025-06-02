import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_evolution_results(save_dir):
    """Genera grafici per tutti i file CSV nella directory"""
    for file in os.listdir(save_dir):
        if file.startswith("evolution_level") and file.endswith(".csv"):
            csv_path = os.path.join(save_dir, file)
            output_path = os.path.join(save_dir, f"plot_{file.replace('.csv', '.png')}")
            
            try:
                df = pd.read_csv(csv_path)
                
                # Prepara i dati
                df['episode_num'] = df.groupby(['level', 'branch']).cumcount() + 1
                
                # Crea figura
                plt.figure(figsize=(12, 6))
                sns.set_style("whitegrid")
                
                # Plot per ogni branch
                for branch in df['branch'].unique():
                    branch_df = df[df['branch'] == branch]
                    plt.plot(branch_df['episode_num'], 
                            branch_df['reward'].rolling(10, min_periods=1).mean(),
                            label=f"{branch} (smoothed)")
                
                plt.axhline(0, color='red', linestyle='--', alpha=0.5)
                plt.title('Evolutionary DDDQN Training Progress')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.legend()
                plt.tight_layout()
                
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f" Grafico generato: {output_path}")
                
            except Exception as e:
                print(f" Errore con {file}: {str(e)}")

def plot_combined_results(save_dir):
    """Grafico combinato di tutti i livelli"""
    all_data = []
    
    for file in os.listdir(save_dir):
        if file.startswith("evolution_level") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(save_dir, file))
            all_data.append(df)
    
    if not all_data:
        return
        
    combined_df = pd.concat(all_data)
    combined_df['episode_num'] = combined_df.groupby(['level', 'branch']).cumcount() + 1
    
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # Colori diversi per ogni livello
    palette = sns.color_palette("husl", combined_df['level'].nunique())
    
    for i, (level, group) in enumerate(combined_df.groupby('level')):
        for branch in group['branch'].unique():
            branch_df = group[group['branch'] == branch]
            plt.plot(branch_df['episode_num'], 
                    branch_df['reward'].rolling(20, min_periods=1).mean(),
                    color=palette[i],
                    linestyle='--' if 'B1' in branch else '-',
                    label=f"L{level}_{branch.split('_')[-1]}")
    
    plt.title('Combined Evolutionary Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, "combined_training_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Grafico combinato generato: {output_path}")