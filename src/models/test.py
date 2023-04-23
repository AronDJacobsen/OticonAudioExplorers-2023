from tqdm import tqdm
import torch

def predict(loader, model, device, output_data=False):
    with torch.no_grad():
        predictions = []
        data = []
        for batch in tqdm(iter(loader)):
            # Get data from batch
            x       = batch['data'].to(device)
            targets = batch['label'].to(device).to(torch.long)
            
            # Get model outputs
            logits = model(x)['t_logits']

            # Store predictions
            preds = logits.argmax(axis=1)
            predictions.append(preds)

            if output_data:
                # Store data
                data.append(x)

    # Stack latent representations to one frame
    return torch.hstack(predictions) if not output_data else torch.hstack(predictions), torch.vstack(data)


def run_inference(loader, model, device):
    latent_representations, predictions, equals, all_targets, probs = [], [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(iter(loader)):
            x       = batch['data'].to(device)
            targets = batch['label'].to(device).to(torch.long)
            outputs = model(x)

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

    # Stack latent representations to one frame
    return (
        torch.vstack(latent_representations).cpu().numpy(), 
        torch.hstack(predictions).cpu().numpy(), 
        torch.vstack(probs).cpu().numpy(), 
        torch.hstack(all_targets).cpu().numpy(), 
        torch.hstack(equals).cpu().numpy()
    )


def run_inference_with_computational_metrics(loader, model, device):
    latent_representations, predictions, equals, all_targets, probs = [], [], [], [], []
    memory_usage_data = []

    os.remove("emissions.csv")
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