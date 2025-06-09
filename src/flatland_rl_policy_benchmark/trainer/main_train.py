import os
import logging
import pandas as pd
import glob
import traceback
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from flatland_rl_policy_benchmark.trainer.train_dddqn import train
from flatland_rl_policy_benchmark.trainer.plot_utils import plot_evolution_results, plot_combined_results

#crea la struttura ad albero per il training e per ogni nodo si va a chiamre la funzione train()


BASE_SAVE_DIR = "results" #salva i risultati nella cartella result
ROUNDS_PER_LEVEL = 10 #quanti round vengono eseguiti per ogni livello
N_AGENTS = 4
GRID_SIZE = 25
MAX_PROCESSES = 4  # Numero di processi paralleli

# scrive messaggi di log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #fornisce un file dentro log con tutti i messaggi
        logging.FileHandler(os.path.join(BASE_SAVE_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) #inizializza il logger

#STRUTTURA DATI CHE RAPPRESENTA I NODI DELL'ALBERO, dove con il costruttore crei un nodo al livello n e con add lo aggiungi al padre
class TrainingNode:
    def __init__(self, level, branch, parent_branch=None):
        self.level = level # livello dell'albero
        self.branch = branch #identifica il nodo corrente
        self.parent_branch = parent_branch # identifica il nodo padre
        self.children = [] # lista dei figli del nodo corrente
        self.rewards = [] #salva le reward ottenute in ogni episodio
    
    def add_child(self, child_node):
        self.children.append(child_node)


def get_level_params(level):
    if level == 0:
        return {"n_agents": 2, "width": 25, "height": 25, "max_depth": 2}
    elif level == 1:
        return {"n_agents": 3, "width": 30, "height": 25, "max_depth": 3}
    elif level == 2:
        return {"n_agents": 4, "width": 35, "height": 30, "max_depth": 3}
    else:
        return {"n_agents": 4, "width": 40, "height": 35, "max_depth": 3}

# Costruisce la struttura ad albero iniziale, solo il padre e i preimi due figli perchè gli altri dipendo dal migliore
def build_tree_structure():
    #il livello lo scrivi a numero, il branch lo scrivi come stringa e hai due numeri nella stringa perchè uno rappresenta il label giusto per chiarezza
    root = TrainingNode(0, '0')
    # Livello 1 - sempre due rami iniziali
    node_0 = TrainingNode(1, '0', '0') # ramo '0' dal padre '0'
    node_1 = TrainingNode(1, '1', '0') # ramo '1' dal padre '0'
    root.children = [node_0, node_1]
    # Livello 2 - non creiamo i rami di livello 2 finché non conosciamo il miglior branch di L1
    return root

def find_best_branch(level):
    """Trova il branch con la reward media più alta per un dato livello"""
   
   #cerca tutti i file cvs che iniziano con evolution_levelX_ e terminano con .csv 
    pattern = os.path.join(BASE_SAVE_DIR, f"evolution_level{level}_*.csv")
    files = glob.glob(pattern)
   #se non ci sono file, ritorna 0 per evitare problemi
    if not files:
        return '0'  # Default se non esiste
    #Apre ogni CSV, lo carica come DataFrame, e lo mette in all_data
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            logger.error(f"Errore leggendo {file}: {str(e)}")
            continue
    # Se non ci sono dati, ritorna 0  
    if not all_data:
        return '0'
    
    # Combina tutti i DataFrame in uno solo per analizzarli
    df_combined = pd.concat(all_data, ignore_index=True)
    # Estrai il nome base del branch (es. "0" da "L1_B0")
    df_combined['base_branch'] = df_combined['branch'].str.extract(r'B(\d+)$')
    #calcola la media delle ricompense per ogni branch e forzi la rewar a essere numerica
    df_combined['reward'] = pd.to_numeric(df_combined['reward'], errors='coerce')
    avg_rewards = df_combined.groupby('base_branch')['reward'].mean()
    #restituisce il branch con la media più alta
    return str(avg_rewards.idxmax()) if not avg_rewards.empty else '0'



def run_training(node):
    """Esegue il training per un nodo specifico"""
    try:
        #da un nome al ramo
        branch_name = f"L{node.level}_B{node.branch}"
        
        #se il nodo ha un genitore allora costruisce il path file del modello padre
        parent_path = None
        if node.parent_branch is not None:
            parent_path = os.path.join(BASE_SAVE_DIR, f"best_model_L{node.level-1}_B{node.parent_branch}.pt")
            if not os.path.exists(parent_path):
                logger.warning(f"File del modello padre non trovato: {parent_path}. Utilizzo il default.")
                parent_path = None

        #recupare i parametri del livello
        level_params = get_level_params(node.level)

        logger.info(f"Inizio training {branch_name} (Parent: {node.parent_branch or 'None'})")
        logger.debug(f"Config: {level_params}")
        logger.info(f"Parametri per {branch_name}: "
                    f"Agenti={level_params['n_agents']}, "
                    f"Dimensione={level_params['width']}x{level_params['height']}")

        try:
            #allena l'agente
            rewards = train(
                round_start=node.level * ROUNDS_PER_LEVEL,
                n_rounds=ROUNDS_PER_LEVEL,
                branch_name=branch_name,
                parent_path=parent_path,
                save_dir=BASE_SAVE_DIR,
                level=node.level,
                width=level_params["width"],
                height=level_params["height"],
                
            )   
        except RuntimeError as e:
            if "Cannot fit more than one city" in str(e):
                logger.warning("Problema generazione ambiente, riduco dimensioni...")
                
                #SALVA LE REWARD DEL NODO 
                rewards = train(
                    round_start=node.level * ROUNDS_PER_LEVEL,
                    n_rounds=ROUNDS_PER_LEVEL,
                    branch_name=branch_name,
                    parent_path=parent_path,
                    save_dir=BASE_SAVE_DIR,
                    level=node.level,
                    width=25,
                    height=25,
                    
                )
            else:
                raise

        node.rewards = rewards
        logger.info(f"Completato training {branch_name}")
        return node

    except Exception as e:
        logger.error(f"Errore nel nodo {node.level}-{node.branch}: {str(e)}", exc_info=True)
        return None


def parallel_tree_training(root):
    """Esegue l'addestramento selezionando solo il miglior percorso"""
    #CREA UN POOL DI PROCESSI PARALLELI
    with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        # Livello 0
        future_root = executor.submit(run_training, root)
        root_result = future_root.result()
        # controllo se il training del nodo radice fallisce, interrompi l'esecuzione       
        if root_result is None:
            logger.error("Training del nodo radice fallito, interruzione.")
            return 
        # Livello 1
        futures_level1 = [executor.submit(run_training, child) for child in root_result.children]
        results_level1 = [f.result() for f in as_completed(futures_level1)]
      
        # Filtra nodi falliti
        results_level1 = [res for res in results_level1 if res is not None]
        if not results_level1:
            logger.error("Nessun nodo di livello 1 completato, interruzione.")
            return
      
        # Trova il miglior branch di livello 1
        best_branch = find_best_branch(1)
        logger.info(f"Miglior branch al livello 1: L1_B{best_branch}")
        # Identifica il nodo corrispondente al branch migliore
        best_node = None
        for node in results_level1:
            if node.branch == best_branch:
                best_node = node
                break
        if best_node is None:
            logger.error("Nessun nodo migliore trovato per il livello 1, interruzione.")
            return
        # Crea due figli a livello 2 dal modello migliore di livello 1
        best_node.children = [TrainingNode(2, '0', best_branch), TrainingNode(2, '1', best_branch)]
        # Avvia in parallelo l'addestramento per i due figli del livello 2
        futures_level2 = [executor.submit(run_training, child) for child in best_node.children]
        # Attendi completamento dei processi di livello 2
        for future in as_completed(futures_level2):
            _ = future.result()

def main():
    try:
        logger.info("Avvio esperimento evolutivo ad albero")
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)
        # Costruisci l'albero
        root = build_tree_structure()
        # Esegui l'addestramento parallelo
        parallel_tree_training(root)
        # Genera i grafici alla fine di tutto, cioè dopo che che tutto l'albero è stato processato
        csv_files = [f for f in os.listdir(BASE_SAVE_DIR) if f.startswith("evolution_level") and f.endswith(".csv")]
        if csv_files:
            logger.info("Generazione dei grafici...")
            plot_evolution_results(BASE_SAVE_DIR)
            plot_combined_results(BASE_SAVE_DIR)
            logger.info("Esperimento completato con successo!")
        else:
            logger.error("Nessun risultato trovato, impossibile generare grafici")
    except Exception as e:
        logger.critical(f"Errore critico: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()