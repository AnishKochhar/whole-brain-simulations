## Data loader class for BOLD + SC matrices
##  Used for WholeBrainModel and 

import os, random
import numpy as np
import torch
import scipy.io
from wbm.utils import DEVICE

class BOLDDataLoader:
    def __init__(self, fmri_filename: str, dti_filename: str, distance_matrices_path: str, chunk_length: int = 50):
        """
        Loads fMRI (BOLD) time series, Structural Connectivity matrices, and distance (delay) matrices, and splits BOLD time series into chunks
        """
        self.fmri_filename = fmri_filename
        self.dti_filename = dti_filename
        self.distance_matrices_path = distance_matrices_path
        self.chunk_length = chunk_length
        self.all_bold = []      # list of BOLD arrays, each shape (node_size, num_TRs)
        self.all_SC = []        # list of SC matrices, each shape (node_size, node_size)
        self.all_distances = [] # list of dist_matrix, each shape (node_size, node_size)
        self.bold_chunks = []   # list of dicts: {'subject': int, 'bold': array (node_size, chunk_length)}
        
        self._load_data()
        self._split_into_chunks()

    def get_node_size(self):
        if len(self.all_SC) == 0: 
            return 0
        return self.all_SC[0].shape[0]

    def _load_data(self):
        fmri_mat = scipy.io.loadmat(self.fmri_filename)
        bold_data = fmri_mat["BOLD_timeseries_HCP"]    # shape (100, 1)
        # dti_mat = scipy.io.loadmat(self.dti_filename)
        # dti_data = dti_mat["DTI_fibers_HCP"]           # shape (100, 1)
        num_subjects = bold_data.shape[0]
        
        for subject in range(num_subjects):
            bold_subject = bold_data[subject, 0]  # shape (100, 1189)
            # dti_subject = dti_data[subject, 0]    # shape (100, 100)
            self.all_bold.append(bold_subject)
            
            # SC pre-processed: symmetric, log-transform, normalise
            sc_path = os.path.join(self.distance_matrices_path, f"sc_norm_subj{subject}.npy")
            sc_norm = np.load(sc_path)
            self.all_SC.append(sc_norm)

            dist_path = os.path.join(self.distance_matrices_path, f"subj{subject}.npy")
            dist_matrix = np.load(dist_path)
            self.all_distances.append(dist_matrix)
            
        print(f"[DataLoader] Loaded {num_subjects} subjects.")

    def _split_into_chunks(self):
        self.bold_chunks = []
        for subject, bold_subject in enumerate(self.all_bold):
            num_TRs = bold_subject.shape[1]
            num_chunks = num_TRs // self.chunk_length
            for i in range(num_chunks):
                chunk = bold_subject[:, i*self.chunk_length:(i+1)*self.chunk_length]
                self.bold_chunks.append({"subject": subject, "bold": chunk})
        print(f"[DataLoader] Created {len(self.bold_chunks)} chunks (chunk length = {self.chunk_length}).")

    def batched_dataset_length(self, batch_size: int):
        return min(len(self.bold_chunks) // batch_size, 4)

    def load_all_ground_truth(self, builder):
        """
        Loads all chunked ground truth data as PyTorchGeometric.Data for discriminator training 
        Parameters:
            builder: GraphBuilder, defined inside discriminator package
        """
        gt_graphs = []
        for subject, _bold in enumerate(self.all_bold):
            sc = self.all_SC[subject]
            gt_graphs.extend(
                [builder.build_graph(torch.tensor(chunk["bold"], dtype=torch.float32, device=DEVICE), torch.tensor(sc, dtype=torch.float32, device=builder.device), label=1.0) for chunk in self.bold_chunks if chunk["subject"] == subject]
            )

        return gt_graphs

    def sample_minibatch(self, batch_size: int):
        sampled = random.sample(self.bold_chunks, batch_size)
        batched_bold = []
        batched_SC = []
        batched_laplacians = []
        batched_dist = []
        batch_subjects = []

        for batch_element in sampled:
            batched_bold.append(batch_element["bold"]) # (node_size, chunk_length)
            subject = batch_element["subject"]
            batch_subjects.append(subject)

            # NOTE: Test with non-Laplacian SC
            sc_norm = self.all_SC[subject]
            batched_SC.append(sc_norm)

            degree_matrix = np.diag(np.sum(sc_norm, axis=1))
            laplacian = degree_matrix - sc_norm
            batched_laplacians.append(laplacian)

            distance_matrix = self.all_distances[subject]
            
            batched_dist.append(distance_matrix)

            # Plotter.plot_laplacian(subject, laplacian)
            # Plotter.plot_distance_matrix(subject, distance_matrix)

        # Stack BOLD
        batched_bold = np.stack(batched_bold, axis=-1) # (node_size, chunk_length, batch_size)
        batched_bold = torch.tensor(batched_bold, dtype=torch.float32, device=DEVICE)

        # Stack batched laplacians
        batched_laplacians = np.stack(batched_laplacians, axis=0) # (batch_size, node_size, node_size)
        batched_laplacians = torch.tensor(batched_laplacians, dtype=torch.float32, device=DEVICE)

        # Stack batched SC
        batched_SC = np.stack(batched_SC, axis=0)
        batched_SC = torch.tensor(batched_SC, dtype=torch.float32, device=DEVICE)

        # Stack distance matrices
        batched_dist = np.stack(batched_dist, axis=0)
        batched_dist = torch.tensor(batched_dist, dtype=torch.float32, device=DEVICE)

        batch_subjects = torch.tensor(batch_subjects, dtype=torch.int32, device=DEVICE)

        return batched_bold, batched_SC, batched_laplacians, batched_dist, batch_subjects

