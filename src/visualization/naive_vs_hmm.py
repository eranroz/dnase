# coding=utf-8
import os
import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
from config import MEAN_DNASE_DIR
from data_provider import SeqLoader
from data_provider.DiscreteTransformer import DiscreteTransformer

__author__ = 'eranroz'


def get_data():
    raw_data = SeqLoader.load_result_dict(os.path.join(MEAN_DNASE_DIR, "brain_fetal.mean.npz"))['chr7']
    raw_data = SeqLoader.down_sample(raw_data, 100 / 20)
    return raw_data
    raw_data = transform(raw_data)


def naive_approch():
    data = get_data()
    data = np.log(data + 1)
    size = 1000
    start = 12000
    data = data[start - size / 2:start + size / 2]

    x = np.arange(size)
    percentiles_values = np.percentile(data, q=[70, 100])
    transform = DiscreteTransformer([70])
    plt.figure(figsize=(16, 8))
    gs = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[6, 1, 1], hspace=0)
    ax = plt.subplot(gs[0])
    prev_v = 0
    colors = ['#ffff00', '#ff0000', '#ff0000', '#ff0000']
    for v, c in zip(percentiles_values, colors):
        p = plt.axhspan(prev_v, v, color=c, alpha=0.2)  # , facecolor='0.5'
        prev_v = v

    plt.plot(x, data, '-', alpha=0.8, label='100b')
    ds = 10
    plt.plot(np.arange(0, size, ds), SeqLoader.down_sample(data, ds) / ds, '-', alpha=0.8, label='1kb')
    plt.annotate('noise??', xy=(215, 2.6), arrowprops=dict(arrowstyle='->'), xytext=(210, 4), size='large')
    plt.annotate('smooth??', xy=(340, 4), arrowprops=dict(arrowstyle='->'), xytext=(420, 5), size='large')
    #plt.scatter(x, data, c='b')
    plt.axis((0, size, 0, np.max(data)))
    plt.legend(loc='upper right')
    ax.set_xticks(np.arange(0, size, 200))
    ax.set_xticklabels(['%ikb' % (x / 10) for x in np.arange(0, size, 200)])
    ax = plt.subplot(gs[1])  #2, 1, 2
    #plt.plot('')
    #ax.set_y_ticks()
    #plt.scatter(x, np.zeros(x.shape[0]), c=transform(data), cmap=matplotlib.cm.hsv)#, s=1
    colors = matplotlib.colors.ListedColormap(colors)
    plt.title('Naive (100bp)')
    plt.imshow(np.array([transform(data)]), cmap=colors, extent=[0, size, 0, 20])
    ax.set_yticks([])
    ax.set_xticks([])

    ax = plt.subplot(gs[2])
    from dnase import dnase_classifier

    model = dnase_classifier.load('Discrete500-a[0.000, 0.000000]')

    plt.title('Naive (1kb)')
    smoothed = np.array([transform(SeqLoader.down_sample(data, ds) / ds)] * ds).T.reshape(1, size)
    plt.imshow(smoothed, cmap=colors, extent=[0, size, 0, 20])  # , aspect='auto'  #matplotlib.cm.hsv
    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout()
    #plt.savefig()
    plt.show()


def hmm_approch():
    data = get_data()
    data = np.log(data + 1)
    size = 1000
    start = 12000
    data = data[start - size / 2:start + size / 2]
    percentiles_values = np.percentile(data, q=[70, 100])
    transform = DiscreteTransformer([70])

    x = np.arange(size)
    plt.figure(figsize=(16, 8))
    gs = matplotlib.gridspec.GridSpec(4, 1, height_ratios=[6, 1, 1, 1], hspace=0)
    ax = plt.subplot(gs[0])
    prev_v = 0
    colors = ['#ffff00', '#ff0000', '#ff0000', '#ff0000']
    for v, c in zip(percentiles_values, colors):
        p = plt.axhspan(prev_v, v, color=c, alpha=0.2)  # , facecolor='0.5'
        prev_v = v

    plt.plot(x, data, '-', alpha=0.8, label='100b')
    ds = 10

    plt.plot(np.arange(0, size, ds), SeqLoader.down_sample(data, ds) / ds, '-*', alpha=0.8, label='1kb')
    plt.annotate('noise??', xy=(215, 2.6), arrowprops=dict(arrowstyle='->'), xytext=(210, 4), size='large')
    plt.annotate('smooth??', xy=(340, 4), arrowprops=dict(arrowstyle='->'), xytext=(420, 5), size='large')
    #plt.scatter(x, data, c='b')
    plt.axis((0, size, 0, np.max(data)))
    plt.legend(loc='upper right')
    ax.set_xticks(np.arange(0, size, 200))
    ax.set_xticklabels(['%ikb' % (x / 10) for x in np.arange(0, size, 200)])
    ax = plt.subplot(gs[1])  #2, 1, 2
    #plt.plot('')
    #ax.set_y_ticks()
    #plt.scatter(x, np.zeros(x.shape[0]), c=transform(data), cmap=matplotlib.cm.hsv)#, s=1
    colors_n = matplotlib.colors.ListedColormap(colors)
    plt.title('Naive (100bp)')
    plt.imshow(np.array([transform(data)]), cmap=colors_n, extent=[0, size, 0, 20])
    ax.set_yticks([])
    ax.set_xticks([])

    ax = plt.subplot(gs[2])
    from dnase import dnase_classifier

    model = dnase_classifier.load('Discrete500-a[0.000, 0.000000]')

    plt.title('Naive (1kb)')
    smoothed = np.array([transform(SeqLoader.down_sample(data, ds) / ds)] * ds).T.reshape(1, size)
    colors_n = matplotlib.colors.ListedColormap(colors)
    plt.imshow(smoothed, cmap=colors_n, extent=[0, size, 0, 20])  # , aspect='auto'  #matplotlib.cm.hsv
    ax.set_yticks([])
    ax.set_xticks([])

    ax = plt.subplot(gs[3])
    plt.title('HMM')
    transform = DiscreteTransformer()
    classification = next(model.classify([{'x': transform(data)}]))['x']
    colors_n = matplotlib.colors.ListedColormap(colors)
    plt.imshow(np.array([classification]), cmap=colors_n, extent=[0, size, 0, 20])
    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout()
    #plt.savefig()
    plt.show()


hmm_approch()
naive_approch()