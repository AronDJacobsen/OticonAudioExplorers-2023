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
    plt.show()