import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

from typing import Optional


def standardize(X, dtype: str, mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None):    
    if dtype == 'train':
        assert mu is None and sigma is None, "In training mode, the mean and standard deviation are estimated from the data."
        # Compute moments
        mu, sigma = np.mean(X, axis=0).reshape(1, X.shape[1], X.shape[2]), np.std(X, axis=0).reshape(1, X.shape[1], X.shape[2]), 
    else:
        assert mu is not None and sigma is not None, "Specify mean and standard deviation when running in evaluation mode."
    
    # Standardize
    X_ = (X - mu) / sigma
    return X_, mu, sigma


def downsample(Xtrain_orig, ttrain_orig):
    
    label_counts, _ = visualize_label_distribution(ttrain_orig, ttrain_orig, plot = False)

    # Naive downsampling to minority class size
    target_size = min(label_counts['train'].values())

    Xtrain_downsampled, ttrain_downsampled = [], []
    num_classes = len(np.unique(ttrain_orig))
    for i in range(num_classes):
        Xtrain_class    = Xtrain_orig[ttrain_orig == i]
        ttrain_class    = ttrain_orig[ttrain_orig == i]
        N_class         = Xtrain_class.__len__()

        # Randomly draw as many points as in minority class 
        keep_idxs   = np.random.choice(np.arange(N_class), size=target_size, replace=False)

        # Downsampled class data
        Xtrain_     = Xtrain_class[keep_idxs, :]
        ttrain_     = ttrain_class[keep_idxs]

        # Append class-selected data points
        Xtrain_downsampled.append(Xtrain_)
        ttrain_downsampled.append(ttrain_)

    # Concatenate
    Xtrain_downsampled = np.vstack(Xtrain_downsampled)
    ttrain_downsampled = np.concatenate(ttrain_downsampled)
    
    return Xtrain_downsampled, ttrain_downsampled


def visualize_label_distribution(ttrain, tval, plot = True):
    
    idx2label   = {0: 'Other', 1: 'Music', 2: 'Human voice', 3: 'Engine sounds', 4: 'Alarm'}
    label2idx   = {v: k for k, v in idx2label.items()}
    num_classes = len(np.unique(ttrain))
    
    N_train, N_val = len(ttrain), len(tval)
    
    label_dist = {'train': dict(sorted(Counter(ttrain).items(), key=lambda x: -x[1])), 'val': dict(sorted(Counter(tval).items(), key=lambda x: -x[1]))}

    label_counts = {dtype: {idx2label[k]: v for k, v in dict_.items()} for dtype, dict_ in label_dist.items()}
    label_freqs = {dtype: {idx2label[k]: v / (N_train if dtype == 'train' else N_val) for k, v in dict_.items()} for dtype, dict_ in label_dist.items()}

    if plot:
        # Plot pie chart
        fig = plt.figure(figsize=(10, 5), dpi=200)
        ax1 = plt.subplot2grid((2,2),(0,0))
        plt.pie(x=label_counts['train'].values(), labels=label_counts['train'].keys())
        plt.title('Train')

        ax1 = plt.subplot2grid((2,2), (0, 1))
        plt.pie(x=label_counts['val'].values(), labels=label_counts['val'].keys())
        plt.title('Validation')

        plt.tight_layout()
        plt.show()

    return label_counts, label_freqs
