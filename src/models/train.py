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
    lambda_: float = 0.5,
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
    loaders, mu, sigma = get_loaders(
        data_path = data_path, 
        balancing_strategy = balancing_strategy, 
        batch_size = batch_size,
        shuffle = True,
        val_size = val_size,
        seed=seed
    )
    
    # Get model, optimizer and criterion
    model, optimizer, recon_criterion, predictive_criterion = get_model(model_name=model_name, lr=lr, device=device)
    
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
                plt.close()

            # Initialize training epoch
            running_loss_train, running_loss_val        = 0, 0
            running_MSELoss_train, running_MSELoss_val  = 0, 0    
            running_CELoss_train, running_CELoss_val    = 0, 0   
            running_acc_train, running_acc_val          = 0, 0 

            model.train()
            for batch in iter(loaders['train']):
                
                # Get data
                x       = batch['data'].to(device)
                targets = batch['label'].to(device).to(torch.long)

                # Reset parameter gradients
                optimizer.zero_grad()

                # Get model outputs
                outputs = model(x)               

                # Compute loss and backpropagate gradient for update
                recon_loss      = recon_criterion(outputs['x_recon'], x)
                predictive_loss = predictive_criterion(outputs['t_logits'], targets)

                train_loss = (1 - lambda_) * recon_loss + lambda_ * predictive_loss
                train_loss.backward()
                optimizer.step()

                # Add the mini-batch training loss to epoch loss
                running_loss_train += train_loss.item()
                running_MSELoss_train += recon_loss.item()
                running_CELoss_train += predictive_loss.item()

                # Compute accuracy
                preds = outputs['t_logits'].detach().argmax(axis=1)
                equals = preds.cpu().numpy() == targets.cpu().numpy()
                running_acc_train += equals.mean()
                
            # Evaluation
            model.eval()
            with torch.no_grad():
                for batch in iter(loaders['val']):
                    
                    # Get data
                    x       = batch['data'].to(device)
                    targets = batch['label'].to(device).to(torch.long)

                    # Get model output
                    outputs = model(x)

                    # Compute losses
                    recon_loss      = recon_criterion(outputs['x_recon'], x)
                    predictive_loss = predictive_criterion(outputs['t_logits'], targets)
                    val_loss = (1 - lambda_) * recon_loss + lambda_ * predictive_loss

                    # Store losses
                    running_loss_val += val_loss.item()
                    running_MSELoss_val += recon_loss.item()
                    running_CELoss_val += predictive_loss.item()
                
                    # Compute accuracy
                    preds = outputs['t_logits'].argmax(axis=1)
                    equals = preds.cpu().numpy() == targets.cpu().numpy()
                    running_acc_val += equals.mean()

            # Logging and progress bar update
            t.set_description(
                f"EPOCH [{epoch + 1}/{epochs}] --> Train loss: {running_loss_train / Ntrain:.4f} | Validation loss: {running_loss_val / Nval:.4f} | Progress: "
            )
            writer.add_scalar('A. Total Loss/train', running_loss_train / Ntrain, epoch)
            writer.add_scalar('A. Total Loss/validation', running_loss_val / Nval, epoch)
            writer.add_scalar('B. CrossEntropyLoss/train', running_CELoss_train / Ntrain, epoch)
            writer.add_scalar('B. CrossEntropyLoss/validation', running_CELoss_val / Nval, epoch)
            writer.add_scalar('C. MSELoss/train', running_MSELoss_train / Ntrain, epoch)
            writer.add_scalar('C. MSELoss/validation', running_MSELoss_val / Nval, epoch)
            writer.add_scalar('D. Accuracy/train', running_acc_train / Ntrain, epoch)
            writer.add_scalar('D. Accuracy/validation', running_acc_val / Nval, epoch)

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
                    "data": {
                        "mu_standardization": mu,
                        "sigma_standardization": sigma,
                        "val_size": val_size,
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

    # OBS:! REMEMBER TO CHANGE EXPERIMENT NAME BEFORE RUNNING - OTHERWISE IT WILL OVERWRITE !
    
    train(
        # Setup
        model_name              = 'PredictiveConvAutoencoder', 
        experiment_name         = 'PredConvAE.lr=1e-4.lambda=0.0.bz=64.seed=42.friday_original',
        data_path               = 'data',
        save_path               = 'models',
        logs_path               = 'logs',
        figure_path             = 'figures',

        # Data parameters
        balancing_strategy      = 'downsample',
        val_size                = 0.2,

        # Training parameters
        batch_size              = 64,
        lr                      = 1e-4,
        lambda_                 = 0.0,       # weights predictive loss mostly --> higher lambda favors CrossEntropy loss, lower lambda favors reconstruction loss
        epochs                  = 100, 

        # Reproducibility parameters
        checkpoint_every_epoch  = 1,
        visualize_every_epoch   = 1,
        seed                    = 42,
    )