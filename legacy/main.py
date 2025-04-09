# Anish Kochhar, Imperial College London, March 2025

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from costs import *
from trainer import *
from model import *

def load_BOLD_and_SC(fmri_filename="HCP Data/BOLD Timeseries HCP.mat", dti_filename="HCP Data/DTI Fibers HCP.mat", subject_index=None):
    """
    Loads BOLD fMRI data and Structural Connectivity data from given files.
    If subject_index is given the returns particular subject, otherwise returns entire 100 participant dataset.
    Normalises and Log-Transforms the Structural Connectivity

    Returns:
        BOLD timeseries (participants, 1) or (nodes, time_steps)
    """
    print("Loading .mat fMRI and DTI data")
    bold_mat = scipy.io.loadmat(fmri_filename)
    dti_mat = scipy.io.loadmat(dti_filename)

    bold_data = bold_mat["BOLD_timeseries_HCP"]
    dti_data  = dti_mat["DTI_fibers_HCP"] 

    print('BOLD data shape: ', bold_data.shape) # (100, 1) - 100 participants
    print('DTI data shape: ', dti_data.shape) # (100, 1)

    if subject_index is None:
        return bold_data, dti_data

    else:
        bold_subject = bold_data[subject_index, 0]
        dti_subject  = dti_data[subject_index, 0]

        print(f"BOLD subject {subject_index} shape:", bold_subject.shape)  # (100, 1189)
        print(f"DTI subject {subject_index} shape:", dti_subject.shape)    # (100, 100)

        SC = 0.5 * (dti_subject.T + dti_subject) # Make symmetric
        SC = np.log1p(SC) / np.linalg.norm(np.log1p(SC)) # Log-Transform and Normalize

        return bold_subject, SC

def main():
    subject_index = 0
    BOLD, SC = load_BOLD_and_SC(subject_index=subject_index)

    # Heatmap for SC
    plt.figure(figsize=(8,6))
    sns.heatmap(SC, cmap='viridis')
    plt.title(f"SC Heatmap (Subject {subject_index})")
    plt.xlabel("Time points")
    plt.ylabel("ROI index")
    plt.show()

    # Constants
    node_size = SC.shape[0]
    assert node_size == SC.shape[1], "SC matrix not square"
    assert node_size == BOLD.shape[0], f"BOLD time series with {BOLD.shape[0]} nodes, SC with {node_size} nodes"

    params = ModelParams()

    dt = 0.05
    tr = 0.75
    batch_time = int(tr / dt) # Number of hidden steps per BOLD output
    T = BOLD.shape[1]
    delays_dummy = np.full((node_size, node_size), 3, dtype=np.int64) # MARK

    costs = Costs()
    dmf_model = DMF(SC, delays_dummy, params, dt, batch_time)
    balloon_model = Balloon(params, dt, batch_time)

    trainer = Trainer(dmf_model, balloon_model, costs, BOLD, batch_time=batch_time, plot=True, verbose=True)

    loss_history = trainer.train()

    simulated_bold = trainer.test()


if __name__ == '__main__':
    main()