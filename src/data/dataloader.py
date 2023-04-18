from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

from .utils import downsample, standardize

class OticonAudioExplorers(Dataset):
    
    def __init__(self, data, labels):
        self.data      = torch.tensor(data).unsqueeze(1)
        self.labels    = torch.tensor(labels)
                
        # Define number of classes
        self.n_classes = np.unique(self.labels).__len__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "data":  self.data[item, :], 
            "label": self.labels[item], 
        }
    

def get_loaders(
        data_path: str = '../data', 
        balancing_strategy: str = 'downsample', 
        batch_size: int = 32,
        shuffle: bool = True,
        val_size: float = 0.2,
        seed: int = 0
    ):

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset files
    data_path       = Path(data_path)
    train_data      = np.load(data_path / 'raw/npy/training.npy')
    train_labels    = np.load(data_path / 'raw/npy/training_labels.npy')
    
    # Load test data
    Xtest           = np.load(data_path / 'raw/npy/test.npy')
    ttest           = np.nan * np.ones(Xtest.shape[0])    
    
    # Get split indices
    train_idxs  = np.random.choice(np.arange(train_data.__len__()), size=int(train_data.__len__() * (1-val_size)), replace=False)
    val_idxs    = np.setdiff1d(np.arange(train_data.__len__()), train_idxs)
    assert train_data.__len__() == train_idxs.__len__() + val_idxs.__len__()

    # Split data set
    Xtrain, Xval = train_data[train_idxs, :, :], train_data[val_idxs, :, :]
    ttrain, tval = train_labels[train_idxs],     train_labels[val_idxs]

    if balancing_strategy == 'downsample':
        Xtrain, ttrain = downsample(Xtrain, ttrain)
    else:
        raise NotImplementedError("Not yet implemented...")
    
    # Standardize data
    Xtrain, mu, sigma   = standardize(Xtrain, dtype='train')
    Xval, _, _          = standardize(Xval, dtype='validation', mu=mu, sigma=sigma)
    Xtest, _, _         = standardize(Xtest, dtype='test', mu=mu, sigma=sigma)
    
    # Creating datasets
    train_dataset   = OticonAudioExplorers(Xtrain, ttrain)
    val_dataset     = OticonAudioExplorers(Xval, tval)
    test_dataset    = OticonAudioExplorers(Xtest, ttest)
    
    # Initializing loaders
    return {
        name_: torch.utils.data.DataLoader(
            dataset_, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        for name_, dataset_ in {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}.items()
    }, mu, sigma
