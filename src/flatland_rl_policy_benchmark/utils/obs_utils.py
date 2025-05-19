import numpy as np

def flatten_obs(obs, max_depth=3, num_features=11):
    """
    Appiattisce un'osservazione ad albero in un vettore NumPy di dimensione fissa.
    """
    num_nodes = sum([4 ** i for i in range(max_depth + 1)])
    flat_obs = np.zeros(num_nodes * num_features, dtype=np.float32)

    def fill_obs(node, idx):
        if node is None or 'value' not in node or node['value'] is None:
            return
        flat_obs[idx * num_features:(idx + 1) * num_features] = node['value']
        for i, child in enumerate(node.get('children', [])):
            fill_obs(child, 4 * idx + i + 1)

    fill_obs(obs, 0)
    return flat_obs
