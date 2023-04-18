from tqdm import tqdm
import torch

def predict(loader, model, device):
    with torch.no_grad():
        predictions = []
        for batch in tqdm(iter(loader)):
            # Get data from batch
            x       = batch['data'].to(device)
            targets = batch['label'].to(device).to(torch.long)
            
            # Get model outputs
            logits = model(x)['t_logits']

            # Store predictions
            preds = logits.argmax(axis=1)
            predictions.append(preds)

    # Stack latent representations to one frame
    return torch.hstack(predictions)


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