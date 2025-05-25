import argparse, json, os, datetime, torch
from pathlib import Path
from wbm.fastdmf import FastDMFParams, WholeBrainFastDMF
from wbm.data_loader import BOLDDataLoader
from wbm.costs import Costs
from wbm.model_fitting import ModelFitting
from wbm.plotter import Plotter
from wbm.utils import DEVICE


def parse():
    p = argparse.ArgumentParser()
    # grid search
    p.add_argument('--g',          type=float, default=2.0)
    p.add_argument('--g_EE',       type=float, default=1.4)
    p.add_argument('--g_IE',       type=float, default=0.75)
    p.add_argument('--JN',         type=float, default=0.15)
    p.add_argument('--sigma',      type=float, default=0.02)    # on both E/I/BOLD
    p.add_argument('--V0',         type=float, default=0.20)
    p.add_argument('--delay',      action='store_true')          # use_delay_based
    p.add_argument('--inhgain',    action='store_true')          # inhibitory_gain_scalar

    p.add_argument('--tag',        type=str,   default=None)     # run ID / folder
    p.add_argument('--epochs',     type=int,   default=3)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--batch-size', type=int,   default=4)
    p.add_argument('--chunk-size', type=int,   default=30)
    p.add_argument('--batch_iter', type=int,   default=4)
    p.add_argument('--verbose',    action='store_true')
    return p.parse_args()


def main():
    args = parse()
    delays_max = 1000
    print(f"[MAIN] Batch size: {args.batch_size}")

    ts   = datetime.datetime.now().strftime('%d:%m_%H:%M:%S')
    tag  = args.tag or f"run_{ts}"
    run_dir = Path('runs')/tag
    run_dir.mkdir(parents=True, exist_ok=True)

    fmri_filename = "./HCP Data/BOLD Timeseries HCP.mat"
    dti_filename = "./HCP Data/DTI Fibers HCP.mat"
    sc_path = "./HCP Data/distance_matrices/"
    distance_matrix_path = "./HCP Data/schaefer100_dist.npy"
    encoder_path = "checkpoints/encoder.pt"
    discriminator_path = "checkpoints/discriminator.pt"

    data_loader = BOLDDataLoader(fmri_filename, dti_filename, sc_path, distance_matrix_path, chunk_length=args.chunk_size)

    Plotter.set_tag(tag)

    params = FastDMFParams(g=args.g, g_IE=args.g_IE, sigma_E=args.sigma, sigma_I=args.sigma,\
                            use_delay_based=args.delay, verbose=args.verbose)
    # params = FastDMFParams(
    #     g=args.g, g_EE=args.g_EE, g_IE=args.g_IE, JN=args.JN, 
    #     sigma_E=args.sigma, sigma_I=args.sigma, sigma_BOLD=args.sigma,
    #     V0=args.V0, use_delay_based=args.delay, inhibitory_gain_scalar=args.inhgain,
    #     verbose=args.verbose)


    model = WholeBrainFastDMF(
        params,
        distance_matrix=data_loader.get_distance_matrix(),
        node_size=data_loader.get_node_size(),
        input_size=6,
        batch_size=args.batch_size,
        delays_max=delays_max).to(DEVICE)

    trainer = ModelFitting(model, data_loader,
                           num_epochs=args.epochs,
                           lr=args.lr,
                           cost_function=Costs(),
                           smoothing_window=1,
                           batch_iters=args.batch_iter,
                           log_state=False)

    metrics = trainer.train(delays_max=delays_max, batch_size=args.batch_size)

    # Save results
    core_file = Path('runs')/'log.csv'
    header = not core_file.exists()
    with core_file.open('a') as f:
        if header:
            f.write(','.join(metrics.keys()) + '\n')
        f.write(','.join(map(str, metrics.values())) + '\n')

    with (run_dir/'run_summary.json').open('w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()