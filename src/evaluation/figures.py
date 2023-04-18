import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot style
plt.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 3
sns.set(font_scale=1.5)

colors = ['C0', 'C1', 'C2', 'C3', 'C4']


def plot_marginal_frequency(X, t, num_classes):

    idx2label   = {0: 'Other', 1: 'Music', 2: 'Human voice', 3: 'Engine sounds', 4: 'Alarm'}
    label2idx   = {v: k for k, v in idx2label.items()}


    fig, axs = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(25, 8))    
    for i in range(num_classes):
        # Stack all spectrograms into 1
        Xtrain_class_h = np.hstack(X[t == i])

        # Plot marginal mean frequency per frequency band
        axs[0, i].barh(np.arange(32), np.mean(Xtrain_class_h, axis=1), alpha=0.5, color=colors[i])
        axs[0, i].errorbar(np.mean(Xtrain_class_h, axis=1), np.arange(32), xerr=np.std(Xtrain_class_h, axis=1) / np.sqrt(Xtrain_class_h.shape[1]), color='k')
        axs[0, i].set_title("$\mathbf{Mean} \\rightarrow$ " + f"{idx2label[i]}", loc='left')

        # Plot marginal median frequency per frequency band
        axs[1, i].barh(np.arange(32), np.median(Xtrain_class_h, axis=1), alpha=0.5, color=colors[i])
        axs[1, i].errorbar(np.median(Xtrain_class_h, axis=1), np.arange(32), xerr=np.std(Xtrain_class_h, axis=1) / np.sqrt(Xtrain_class_h.shape[1]), color='k')
        axs[1, i].set_title("$\mathbf{Median} \\rightarrow$ " + f"{idx2label[i]}", loc='left')

        axs[1, i].set_xlabel('Amplitude [dB]')

    axs[0, 0].set_ylabel('Frequency band index')
    axs[1, 0].set_ylabel('Frequency band index')

    fig.suptitle("Marginal amplitude per frequency band")
    return fig

def visualize_utility(ax, U, labels=None):
    
    num_classes = len(U)
    
    ax.imshow(U, cmap=plt.cm.Greys_r, alpha=0.5)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    
    if labels:
        ax.set_xticklabels(labels, rotation=25)
        ax.set_yticklabels(labels)
    
    ax.grid(False)
    
    for (j,i), val in np.ndenumerate(U):
        ax.text(i,j, val, ha='center', va='center', fontsize=16)
    ax.set_title('Utility matrix', fontweight='bold')
    
def visualize_confusion_matrix(fig, ax, cm_, reg_type, dtype):
    idx2label   = {0: 'Other', 1: 'Music', 2: 'Human voice', 3: 'Engine sounds', 4: 'Alarm'}

    # Plot confusion matrix as heatmap
    sns.heatmap(cm_, annot=True, cmap='Blues', ax=ax)

    # Set ticks
    ticks = list(zip(*[(i + 0.5, name_) for i, name_ in idx2label.items()]))
    ax.set_xticks(ticks[0], ticks[1], rotation=25)
    ax.set_yticks(ticks[0], ticks[1], rotation=0)

    # Set labels and title
    ax.set_ylabel('True class')
    ax.set_xlabel('Predicted class')
    ax.set_title(f"Confusion matrix: {dtype}", weight='bold')
    fig.suptitle(f'{reg_type}', weight='bold')
    return fig, ax