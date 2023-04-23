from tqdm import tqdm 

from src.data.dataloader import get_loaders
from src.models.test import predict
from src.models.model import get_model, PredictiveEncoder


import torch
from collections import OrderedDict

if __name__ == '__main__':

    # Run specifications
    experiment_name = 'final-model'
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load stored checkpoint
    checkpoint = torch.load(f'models/{experiment_name}/best.ckpt')
    encoder_state_dict = OrderedDict({layer_: weights_ for (layer_, weights_) in checkpoint['state_dict'].items() if 'decoder' not in layer_})

    # Load pre-trained model
    model, _, _, _ = get_model(model_name=checkpoint['model']['name'], device=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load encoder part
    encoder = PredictiveEncoder().to(device)
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    # Load testloader
    testloader = get_loaders(
        data_path = 'data',
        balancing_strategy='downsample',
        batch_size=64,
        shuffle=False,
        val_size=0.2,
        seed=0,
    )[0]['test']

    # Run predictions
    preds, data = predict(testloader, model, device, output_data=True)

    # Check that testloader is not shuffled
    assert torch.all(data.cpu() == testloader.dataset.data).item()

    # Write predictions
    N_test = len(preds.cpu())
    with open('test_predictions.txt', 'w') as f:
        for i, pred in enumerate(preds.cpu()):
            f.write(str(pred.item()))
            if i != N_test-1:
                f.write('\n')

    