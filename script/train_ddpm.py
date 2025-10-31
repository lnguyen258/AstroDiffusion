import os
import yaml
import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset

from src.config import TrainConfig, DDPM_Config
from src.engine import DiffusionTrainer 
from src.model import Diffusion_Unet
from src.utils import Noise_Scheduler
from src.utils import setup_ddp, set_seed, cleanup_ddp
from src.utils import Galaxies_ML_Dataset
from src.utils import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DDPM on Galaxies_ML imgs")

    # Model and config args
    parser.add_argument('--config_path', type=str, default='./config/train_ddpm.yaml', help="Path to YAML config")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Checkpoint to restore trainer state")

    # Data loading args
    parser.add_argument('--train_data_dir', type=str, help="Train data folder")
    parser.add_argument('--val_data_dir', type=str, help="Validation data folder")

    return parser.parse_args()

def main(
    rank: int,
    world_size: int,
    config_path: str,
    checkpoint_path: str = None,
    train_data_dir: str = None,
    val_data_dir: str = None,   
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = DDPM_Config.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])

    # Initialize multi-GPU processing
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    if train_data_dir is None:
        train_data_dir = './data/train/galaxies_ml_train_set.hdf5'
    if val_data_dir is None:
        val_data_dir = './data/val/galaxies_ml_val_set.hdf5'

    train_dataset = Galaxies_ML_Dataset(train_data_dir)
    val_dataset = Galaxies_ML_Dataset(val_data_dir)

    subset_indices = list(range(1000))
    train_dataset = Subset(train_dataset, subset_indices)
    val_dataset = Subset(val_dataset, subset_indices)

    # Initialize the model
    model = Diffusion_Unet(Config=model_config).to(device)

    # Initialize the noise scheduler
    noise_scheduler = Noise_Scheduler()
    
    #Initialize the trainer
    trainer = DiffusionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=train_config,
        noise_scheduler=noise_scheduler           
    )
    
    # Resume from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")

        try:
            trainer.load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Train the model
    history, model = trainer.train()

    # Run inference
    trainer.inference()
    # Clean up distributed processing
    cleanup_ddp()

    # Save the training history plot
    output_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}.png") if train_config.save_fig else None
    plot_history(history, save_fig=output_path)


if __name__ == '__main__':
    
    # Parse command-line arguments
    args = parse_args()
    set_seed(42)

    # Multi-GPU processing
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args=(
                world_size,
                args.config_path,
                args.checkpoint_path,
                args.train_data_dir,
                args.val_data_dir
            ),
            nprocs=world_size
        )
    else:
        main(
            rank=0,
            world_size=1,
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            train_data_dir=args.train_data_dir,
            val_data_dir=args.val_data_dir
        )

