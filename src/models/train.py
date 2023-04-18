import os

import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import numpy as np
from tqdm import trange

from src.models.model import get_model
from src.data.dataloader import get_loaders

# plot style
plt.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 3
import seaborn as sns
sns.set(font_scale=1.5)

colors = ['C0', 'C1', 'C2', 'C3', 'C4']

def train(
    model_name: str, 
    experiment_name: str,
    epochs: int = 100, 
    checkpoint_every_epoch: int = 1,
    visualize_every_epoch: int = 5,
    lr: float = 1e-3,
    batch_size: int = 128,
    balancing_strategy: str = 'downsample',
    val_size: float = 0.2,
    seed: int = 42, 
    data_path: str = '../data',
    save_path: str = '../models',
    logs_path: str = '../models',
    figure_path: str = '../models',
):
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize logging + create save folders
    os.makedirs(f"{save_path}/{experiment_name}", exist_ok=True)
    os.makedirs(f"{logs_path}/{experiment_name}", exist_ok=True)
    os.makedirs(f"{figure_path}/reconstructions/{experiment_name}", exist_ok=True)
    writer = SummaryWriter(f"{logs_path}/{experiment_name}")

    # Run on GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dataloaders
    loaders = get_loaders(
        data_path = data_path, 
        balancing_strategy = balancing_strategy, 
        batch_size = batch_size,
        shuffle = True,
        val_size = val_size,
        seed=seed
    )
    
    # Get model, optimizer and criterion
    model, optimizer, criterion = get_model(device, lr = 1e-3)
    
    # Initialize loss storage and other
    current_best_loss = np.inf
    Ntrain, Nval = len(loaders['train']), len(loaders['val'])

    with trange(epochs) as t:
        for epoch in t:
            
            # Visualize selected reconstructions
            if epoch % visualize_every_epoch == 0:
                idx = 0
                nrows, ncols = 5, 2
                fig, axs = plt.subplots(nrows, ncols*2, sharex=True, sharey=True, figsize=(12, 8))

                selected_idxs = [2, 4, 1, 3, 9, 20, 5, 44, 10, 41]#, 17, 18]
                with torch.no_grad():
                    viz_batch = torch.stack([loaders['val'].dataset.__getitem__(i)['data'] for i in selected_idxs]).to(device)
                    viz_recon = model(viz_batch)['x_recon']

                for i in range(nrows):
                    for j in range(ncols):
                        axs[i, 2*j].imshow(viz_batch.cpu().squeeze(1)[idx].numpy())
                        axs[i, 2*j+1].imshow(viz_recon.cpu().squeeze(1)[idx].numpy())
                        axs[i, 2*j].set_title(f'Original, $x{selected_idxs[idx]}$')
                        axs[i, 2*j+1].set_title(f'Reconstruction, $\hat x{selected_idxs[idx]}$')
                        axs[i, 2*j].axis('off')
                        axs[i, 2*j+1].axis('off')

                        idx += 1

                fig.suptitle(f"EPOCH {epoch}")
                plt.tight_layout()
                plt.savefig(f'{figure_path}/reconstructions/{experiment_name}/epoch{epoch}.png')

            # Initialize training epoch
            running_loss_train, running_loss_val = 0, 0    
            model.train()
            
            # Train on batches
            for batch in iter(loaders['train']):
                
                # Get data
                x = batch['data'].to(device)

                # Reset parameter gradients
                optimizer.zero_grad()

                # Get reconstruction
                x_recon = model(x)['x_recon']
                
                # Compute loss and backpropagate gradient for update
                train_loss = criterion(x_recon, x)
                train_loss.backward()
                optimizer.step()

                # Add the mini-batch training loss to epoch loss
                running_loss_train += train_loss.item()
                
            # Compute the epoch training loss
            running_loss_train = running_loss_train / Ntrain

            # Evaluation
            model.eval()
            with torch.no_grad():
                for batch in iter(loaders['val']):
                    
                    # Get data
                    x = batch['data'].to(device)
                    # Get reconstruction
                    x_recon = model(x)['x_recon']

                    # Compute and store loss
                    val_loss = criterion(x_recon, x)
                    running_loss_val += val_loss.item()

                # compute the epoch validation loss
                running_loss_val = running_loss_val / Nval
            
            # Logging and progress bar update
            t.set_description(
                f"EPOCH [{epoch + 1}/{epochs}] --> Train loss: {running_loss_train:.4f} | Validation loss: {running_loss_val:.4f} | Progress: "
            )
            writer.add_scalar('loss/train', running_loss_train, epoch)
            writer.add_scalar('loss/validation', running_loss_val, epoch)

            # Store best epoch
            if running_loss_val < current_best_loss and epoch % checkpoint_every_epoch == 0:
                current_best_loss = running_loss_val
                
                # Create checkpoint
                checkpoint = {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "model": {
                        'name': model_name,
                    },
                    "training_parameters": {
                        "save_path": save_path,
                        "lr": lr,
                        "optimizer": optimizer,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "device": device,
                    },
                    "best_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                }

                # Save checkpoint
                torch.save(checkpoint, f"{save_path}/{experiment_name}/best.ckpt")

if __name__ == '__main__':

    train(
        # Setup
        model_name              = 'ConvAutoencoder', 
        experiment_name         = 'ConvAE.lr=1e-3.bz=32',
        data_path               = 'data',
        save_path               = 'models',
        logs_path               = 'logs',
        figure_path             = 'figures',

        # Data parameters
        balancing_strategy      = 'downsample',
        val_size                = 0.2,

        # Training parameters
        batch_size              = 32,
        lr                      = 1e-3,
        epochs                  = 100, 

        # Reproducibility parameters
        checkpoint_every_epoch  = 1,
        visualize_every_epoch   = 1,
        seed                    = 42,
    )