from wbm.utils import DEVICE
from wbm.data_loader import BOLDDataLoader
from wbm.fastdmf import FastDMFParams, WholeBrainFastDMF
from wbm.costs import Costs
from wbm.model_fitting import ModelFitting


fmri_filename = "./HCP Data/BOLD Timeseries HCP.mat"
dti_filename = "./HCP Data/DTI Fibers HCP.mat"
sc_path = "./HCP Data/distance_matrices/"
distance_matrix_path = "./HCP Data/schaefer100_dist.npy"
encoder_path = "checkpoints/encoder.pt"
discriminator_path = "checkpoints/discriminator.pt"

data_loader = BOLDDataLoader(fmri_filename, dti_filename, sc_path, distance_matrix_path, chunk_length=15)

## Model Settings
batch_size = 4                          # Minibatch size
node_size = data_loader.get_node_size() # 100
dt = 0.1                                # ms
dtt = 1.0                               # ms
tr = 750.0                              # ms
delayed_based_feedback = True           # Toggle between delayed feedback (up to `delays_max` timesteps) or last step of E activity
inhibitory_gain_scalar = False          # Toggle between 
input_size = 6                          # Number of noise channels
delays_max = 1000                       # Maximum delay time

trainer_lr               = 1e-2
trainer_epochs           = 5
trainer_smoothing_window = 1
trainer_batch_iters      = 4            # (optional) limit minibatch iterations per epoch

in_dim, hidden_dim, latent_dim = node_size, 64, 32  # From discriminator.ipynb
finetune_lr              = 1e-4
finetune_steps           = 5                        # Number of steps taken in each finetune
finetune_batch           = 32


# params = ModelParams()
params = FastDMFParams(dt=dt, dtt=dtt, tr=tr, use_delay_based=delayed_based_feedback, inhibitory_gain_scalar=inhibitory_gain_scalar, verbose=True)

distance_matrix = data_loader.get_distance_matrix()

model = WholeBrainFastDMF(params, distance_matrix, node_size, input_size, batch_size, delays_max).to(DEVICE)

costs = Costs()

trainer = ModelFitting(model, data_loader, num_epochs=trainer_epochs, lr=trainer_lr, cost_function=costs, \
                       smoothing_window=trainer_smoothing_window, finetune_steps=finetune_steps, finetune_batch=finetune_batch, log_state=False)

trainer.train(delays_max, batch_size)

print("[Main] Training complete")