# Python script to produce distance matrices for each structural connectivity matrix

import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def compute_normalised_sc(sc_matrix, eps=1e-6):
    """
    From an SC adjacency (node x node), produce:
      - normalised log-SC
      - Euclidean distance matrix
    """
    # Symmetrise, log-transform
    sc_sym = 0.5 * (sc_matrix + sc_matrix.T)
    log_sc = np.log1p(sc_sym)

    max_val  = log_sc.max()
    sc_norm = log_sc / (max_val + eps)

    return sc_norm

def compute_distance_matrix(coords):
    dist_matrix = squareform(pdist(coords, metric="euclidean")).astype(np.float32)

    return dist_matrix

def main():
    dti_filename = "HCP Data/DTI Fibers HCP.mat"
    coords_file  = "schaefer100_node_centroids.csv"
    output_dir   = "HCP Data/distance_matrices"
    os.makedirs(output_dir, exist_ok=True)

    coords_df = pd.read_csv(coords_file)
    coords = coords_df[['R', 'A', 'S']].values

    distance_matrix = compute_distance_matrix(coords)
    np.save(os.path.join(output_dir, f"schaefer100_dist.npy"), distance_matrix)

    dti_mat = scipy.io.loadmat(dti_filename)
    dti_data = dti_mat["DTI_fibers_HCP"]  # shape (num_subjects, 1)
    num_subjects = dti_data.shape[0]

    for subject in range(num_subjects):
        sc_raw = dti_data[subject, 0]                           # (node, node)
        sc_norm = compute_normalised_sc(sc_raw)
        np.save(os.path.join(output_dir, f"sc_norm_subj{subject}.npy"), sc_norm)

    print(f"[distance_matrix.py] Saved {num_subjects} SC & distance matrices to '{output_dir}/'")

def validate_subject(subject=0,
                    dti_filename="HCP Data/DTI Fibers HCP.mat",
                    coords_file="schaefer100_node_centroids.csv",
                    save_path="validation_subj0.png"):
    """
    Compute and plot SC and distance matrix for one subject.
    """
    # Load DTI data for one subject
    dti_mat = scipy.io.loadmat(dti_filename)
    sc_raw = dti_mat["DTI_fibers_HCP"][subject, 0]

    # Load coordinates
    coords_df = pd.read_csv(coords_file)
    coords = coords_df[['R', 'A', 'S']].values

    dist_matrix = compute_distance_matrix(coords)
    sc_norm = compute_normalised_sc(sc_raw)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(sc_norm, vmin=0, vmax=1, cmap='viridis')
    axes[0].set_title("Normalised log-SC")
    axes[0].set_xlabel("Node j")
    axes[0].set_ylabel("Node i")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(dist_matrix, vmin=0, vmax=dist_matrix.max(), cmap='magma')
    axes[1].set_title("Euclidean Distance")
    axes[1].set_xlabel("Node j")
    axes[1].set_ylabel("Node i")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(f"Subject {subject}: SC & Distance Matrices", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"[validate_subject] Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
