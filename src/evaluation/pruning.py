import torch
import torch.nn.utils.prune as prune
import os

import numpy as np
import pandas as pd

from collections import OrderedDict

from src.data.dataloader import get_loaders
from src.models.model import get_model, PredictiveEncoder
from src.evaluation.figures import plot_marginal_frequency
# from src.evaluation.pruning import prune_encoder_, pruning_status
from src.models.test import predict, run_inference

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from codecarbon import EmissionsTracker
from memory_profiler import memory_usage

def pruning_status(encoder, global_only: bool = False):
    if not global_only:
        print(
            "Sparsity in block1.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block1.conv1.weight == 0))
                / float(encoder.encoder.block1.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in block2.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block2.conv1.weight == 0))
                / float(encoder.encoder.block2.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in block3.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block3.conv1.weight == 0))
                / float(encoder.encoder.block3.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in block4.conv1.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder.block4.conv1.weight == 0))
                / float(encoder.encoder.block4.conv1.weight.nelement())
            )
        )
        print(
            "Sparsity in encoder_fc.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.encoder_fc.weight == 0))
                / float(encoder.encoder_fc.weight.nelement())
            )
        )
        print(
            "Sparsity in latent_classifier.weight: {:.2f}%".format(
                100. * float(torch.sum(encoder.latent_classifier.weight == 0))
                / float(encoder.latent_classifier.weight.nelement())
            )
        )

    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(encoder.encoder.block1.conv1.weight == 0)
                + torch.sum(encoder.encoder.block2.conv1.weight == 0)
                + torch.sum(encoder.encoder.block3.conv1.weight == 0)
                + torch.sum(encoder.encoder.block4.conv1.weight == 0)
                + torch.sum(encoder.encoder_fc.weight == 0)
                + torch.sum(encoder.latent_classifier.weight == 0)
            )
            / float(
                encoder.encoder.block1.conv1.weight.nelement()
                + encoder.encoder.block2.conv1.weight.nelement()
                + encoder.encoder.block3.conv1.weight.nelement()
                + encoder.encoder.block4.conv1.weight.nelement()
                + encoder.encoder_fc.weight.nelement()
                + encoder.latent_classifier.weight.nelement()
            )
        )
    )


def prune_encoder_(encoder_, pruning_ratio):
    # Determine pruning parameters
    parameters_to_prune = (
        (encoder_.encoder.block1.conv1, 'weight'),
        (encoder_.encoder.block2.conv1, 'weight'),
        (encoder_.encoder.block3.conv1, 'weight'),
        (encoder_.encoder.block4.conv1, 'weight'),
        (encoder_.encoder_fc, 'weight'),
        (encoder_.latent_classifier, 'weight'),
    )

    # Run pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    pruning_status(encoder_, global_only=True)

from tqdm import tqdm

def run_inference_with_computational_metrics(loader, model, device):
    latent_representations, predictions, equals, all_targets, probs = [], [], [], [], []
    memory_usage_data = []

    try:
        os.remove("emissions.csv")
    except:
        None
        
    tracker = EmissionsTracker()
    
    with torch.no_grad():
        for batch in tqdm(iter(loader)):
        
            x       = batch['data'].to(device)
            targets = batch['label'].to(device).to(torch.long)

            tracker.start()
            outputs = model(x)
            memory_usage_data.append(memory_usage()[-1])
            tracker.stop()

            # Store latent representations
            latent_representations.append(outputs['z'].cpu())

            # Store predictions
            preds = outputs['t_logits'].argmax(axis=1)
            predictions.append(preds)
            equals.append(torch.tensor(preds.cpu().numpy() == targets.cpu().numpy()))

            # Store probabilities
            probs.append(outputs['t_logits'].softmax(dim=1)) 

            # Store targets
            all_targets.append(targets)

        computational_metrics = pd.read_csv('emissions.csv')[[
            'timestamp', 'project_name', 'duration', 
            'emissions', 'emissions_rate', 
            'cpu_power', 'gpu_power', 'ram_power', 
            'cpu_energy', 'gpu_energy', 'ram_energy', 'energy_consumed', 
            'os', 'python_version', 
            'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model', 
            'ram_total_size'
        ]]

        # Duration is continuously accumulated
        computational_metrics['duration'] = [computational_metrics['duration'][0]]+[computational_metrics['duration'][i+1] 
                                            - computational_metrics['duration'][i] for i in range(len(computational_metrics['duration']) - 1)]
        computational_metrics['memory_usage'] = memory_usage_data

    # Stack latent representations to one frame
    return (
        torch.vstack(latent_representations).cpu().numpy(), 
        torch.hstack(predictions).cpu().numpy(), 
        torch.vstack(probs).cpu().numpy(), 
        torch.hstack(all_targets).cpu().numpy(), 
        torch.hstack(equals).cpu().numpy(),
        computational_metrics
    )



def prune_eval(experiment_name = 'test', pruning_ratios = np.linspace(0.0, 1.0, 21), batch_size = None): #checkpoint['training_parameters']['batch_size']):

    # Load stored checkpoint
    checkpoint = torch.load(f'{os.getcwd()}/models/{experiment_name}/best.ckpt') #f'../../models/{experiment_name}/best.ckpt')
    encoder_state_dict = OrderedDict({layer_: weights_ for (layer_, weights_) in checkpoint['state_dict'].items() if 'decoder' not in layer_})

    device = str(checkpoint['training_parameters']['device'])

    if batch_size == None:
        batch_size = checkpoint['training_parameters']['batch_size']

    # for changing batch_size
    # Load data
    loaders, mu, sigma = get_loaders(
        data_path = f'{os.getcwd()}/data', #../../data',
        balancing_strategy='downsample',
        batch_size = batch_size,
        shuffle=True,
        val_size=0.2,
        seed= checkpoint['seed'],
    )
    
    accuracy = [] 
    balanced_acc = []
    comp_metrics = []

    pruning_ratios = pruning_ratios
    for pruning_ratio in pruning_ratios:

        # Load encoder
        encoder_ = PredictiveEncoder().to(device)
        encoder_.load_state_dict(encoder_state_dict)
        encoder_.eval()

        # Prune
        prune_encoder_(encoder_ = encoder_, pruning_ratio = pruning_ratio)

        # Get predictions
        _, pred_val, _, tval, equals_val, metrics  = run_inference_with_computational_metrics(loaders['val'], encoder_, device=device)

        accuracy.append(equals_val.sum() / equals_val.__len__())
        balanced_acc.append(balanced_accuracy_score(tval, pred_val))
        comp_metrics.append(metrics)


    # Average duration and memory_usage per batch   
    tmp = np.array([table[['duration', 'memory_usage', 'cpu_power', 'gpu_power', 'ram_power']].mean() for table in comp_metrics])

    # Creating table
    df = pd.DataFrame()
    df['pruning_ratio'] = pruning_ratios
    df['duration'] = tmp[:,0]
    df['memory_usage'] = tmp[:,1]
    df['accuracy'] = accuracy
    df['balanced_acc'] = balanced_acc
    
    return df

if __name__ == '__main__':

    df = prune_eval(experiment_name = 'test', pruning_ratios = np.linspace(0.0, 1.0, 2), batch_size = None)
    print(df)

