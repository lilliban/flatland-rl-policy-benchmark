import numpy as np

def flatten_obs(obs, max_depth=3, num_features=11):
    # Serve per trasformare l’osservazione che ricevi dall’ambiente in un vettore (array) di numeri, in modo che tu possa darlo in input alla rete neurale
    # Le reti neurali lavorano su array di numeri (tensori) → quindi se l’osservazione è una struttura complessa (tipo albero o dizionario), la devi trasformare in un array → questo si chiama flattening
    # Se è già un array, restituiscilo ci sta nel caso in cui l'osservazione è stata già gestita precedentemente
    if isinstance(obs, np.ndarray):
        return obs

    # se l'osservazione è un dizionario allora prendi tutti i valori e li concateni uno dopo l'altro ottenendo un array unico
    if isinstance(obs, dict):
        return np.concatenate([v for v in obs.values()])
    
    # Se l’osservazione è un albero
    
    #calcoli quanti nodi ci sono nell'albero fino a max_depth
    # NOTA:  fai 4**i (4 alla i) perché ogni nodo ha al massimo 4 figli (Nord, Sud, Est, Ovest)
    num_nodes = sum([4 ** i for i in range(max_depth + 1)])
    # vai a creare l'array finale 
    flat_obs = np.zeros(num_nodes * num_features, dtype=np.float32)

    # funzione ricorsiva che visita l'albero per riempire il vettore flat_obs
    #pende il nodo correnre (node) e l'indice corrente (idx) in flat_obs
    def fill_obs(node, idx):
        if node is None or 'value' not in node or node['value'] is None:
            return
        flat_obs[idx * num_features:(idx + 1) * num_features] = node['value']
        
        #per ogni figlio del nodo corrente, chiama ricorsivamente fill_obs
        for i, child in enumerate(node.get('children', [])):
            fill_obs(child, 4 * idx + i + 1)

    fill_obs(obs, 0)
    return flat_obs
