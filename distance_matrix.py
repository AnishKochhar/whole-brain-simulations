# Python script to produce distance matrices for each structural connectivity matrix

import os
import numpy as np
import networkx as nx
import scipy.io

def compute_distance_matrix(sc_matrix, eps=1e-6):
    """
    From an SC adjacency (node x node), produce:
      - normalised log-SC
      - all-pairs shortest-path proxy distances
    """
    # Symmetrise, log-transform, normalise
    sc_sym = 0.5 * (sc_matrix + sc_matrix.T)
    log_sc = np.log1p(sc_sym)
    norm_val = np.linalg.norm(log_sc)
    sc_norm = log_sc / (norm_val + eps)

    # Build directed graph with weight = 1 / connection_strength
    G = nx.from_numpy_array(sc_norm, create_using=nx.DiGraph())
    for u, v, d in G.edges(data=True):
        d['weight'] = 1.0 / (sc_norm[u, v] + eps)

    # All-pairs shortest-path lengths
    dist_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    # Assemble into matrix
    N = sc_matrix.shape[0]
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i, row in dist_dict.items():
        for j, length in row.items():
            dist_matrix[i, j] = length

    return sc_norm, dist_matrix

def main():
    dti_filename = "HCP Data/DTI Fibers HCP.mat"
    output_dir   = "HCP Data/distance_matrices"
    os.makedirs(output_dir, exist_ok=True)

    dti_mat = scipy.io.loadmat(dti_filename)
    dti_data = dti_mat["DTI_fibers_HCP"]  # shape (num_subjects, 1)
    num_subjects = dti_data.shape[0]

    for subject in range(num_subjects):
        sc_raw = dti_data[subject, 0]                           # (node, node)
        sc_norm, dist_matrix = compute_distance_matrix(sc_raw)
        np.save(os.path.join(output_dir, f"subj{subject}.npy"), dist_matrix)
        np.save(os.path.join(output_dir, f"sc_norm_subj{subject}.npy"), sc_norm)

    print(f"[distance_matrix.py] Saved {num_subjects} matrices to '{output_dir}/'")

def test(): 
    from matplotlib.pyplot import plt

    sc_dummy = np.array([
        [0.0, 10.0, 20.0,  0.0],
        [10.0, 0.0,  5.0,  2.0],
        [20.0, 5.0,  0.0,  1.0],
        [ 0.0, 2.0,  1.0,  0.0]
    ], dtype=np.float32)

    sc_norm_dummy, dist_dummy = compute_distance_matrix(sc_dummy)

    # Plot normalized SC
    plt.figure(figsize=(5,4))
    plt.imshow(sc_norm_dummy, vmin=0, vmax=sc_norm_dummy.max(), cmap='viridis')
    plt.title("Normalised log-SC (dummy)")
    plt.colorbar(label='Normalized log strength')
    plt.xlabel("Node j")
    plt.ylabel("Node i")
    plt.show()

    # Plot distance matrix
    plt.figure(figsize=(5,4))
    plt.imshow(dist_dummy, vmin=0, vmax=dist_dummy.max(), cmap='magma')
    plt.title("Distance Matrix (dummy)")
    plt.colorbar(label='Shortest-path proxy distance')
    plt.xlabel("Node j")
    plt.ylabel("Node i")
    plt.show()


if __name__ == "__main__":
    main()
