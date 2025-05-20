# Python script to produce distance matrices for each structural connectivity matrix

import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def compare_log_transform(sc_raw):
    log_sc = np.log1p(sc_raw)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(sc_raw.flatten(), bins=100, color='blue', alpha=0.7)
    axes[0].set_title("Raw SC histogram")
    axes[0].set_yscale("log")
    axes[1].hist(log_sc.flatten(), bins=100, color='orange', alpha=0.7)
    axes[1].set_title("Log(1+SC) histogram")
    axes[1].set_yscale("log")
    plt.tight_layout()
    plt.savefig("val_test.png")

def compute_normalised_sc(sc_matrix):
    """
    From an SC adjacency (node x node), produce:
      - normalised log-SC
      - Euclidean distance matrix
    """
    # Symmetrise, log-transform
    sc_sym = 0.5 * (sc_matrix + sc_matrix.T)
    # norm = np.linalg.norm(sc_sym, ord='fro')
    # sc_norm = sc_sym / norm if norm > 0 else sc_sym

    # log_sc = np.log1p(sc_sym)
    log_sc = sc_sym

    max_val  = log_sc.max()
    sc_norm = log_sc / max_val

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
    compare_log_transform(sc_raw)
    
    row_sum = sc_norm.sum(axis=1)
    print(f"[validate_subject] Subject {subject}")
    print(f"  nodes            : {sc_norm.shape[0]}")
    print(f"  row-sum  min     : {row_sum.min():.4f}")
    print(f"  row-sum  mean    : {row_sum.mean():.4f}")
    print(f"  row-sum  max     : {row_sum.max():.4f}")
    print(f"  row-sum  std     : {row_sum.std():.4f}")

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

def validate_fastdmf_sc(csv_path="HCP Data/fiber_consensus.csv", 
                        figure_path="consensus_sc_diagnostics.png"):
    sc = np.loadtxt(csv_path, delimiter=",")
    sc_norm = sc / sc.max()                         # global-max scaling

    row_sum = sc_norm.sum(axis=1)                   # (N,)
    print("[validate_consensus_sc]")
    print(f"  nodes            : {sc.shape[0]}")
    print(f"  row-sum  min     : {row_sum.min():.4f}")
    print(f"  row-sum  mean    : {row_sum.mean():.4f}")
    print(f"  row-sum  max     : {row_sum.max():.4f}")
    print(f"  row-sum  std     : {row_sum.std():.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1.3, 1, 1]})

    # (A) heat-map
    im = ax[0].imshow(sc_norm, vmin=0, vmax=1, cmap="viridis")
    ax[0].set_title("Normalised consensus SC")
    ax[0].set_xlabel("Node j")
    ax[0].set_ylabel("Node i")
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    # (B) histogram of row-sums
    ax[1].hist(row_sum, bins=30, color="steelblue", edgecolor="k")
    ax[1].set_title("Row-sum histogram")
    ax[1].set_xlabel("Σ Cᵢⱼ")
    ax[1].set_ylabel("# nodes")

    # (C) sorted row-sum curve
    ax[2].plot(np.sort(row_sum), ".-")
    ax[2].set_title("Sorted row-sums")
    ax[2].set_xlabel("Rank")
    ax[2].set_ylabel("Σ Cᵢⱼ")

    fig.suptitle("Consensus SC diagnostics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"  figure saved to  : {figure_path}")


if __name__ == "__main__":
    main()
    # validate_subject()
