import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


# 超参数
parser = argparse.ArgumentParser(description='visualize for dlg or mia.')
parser.add_argument('--attack_type', type=str, default='dlg', help='mia or dlg')
args = parser.parse_args()

train_data = np.load('../data/TE/classify/train_data.npy')
for i in range(22):
    reverse_result = np.load('./results/'+args.attack_type+'/reverse_result/' + str(i) + '.npy')
    reverse_result_protected = np.load('./results/'+args.attack_type+'/reverse_result_protected/' + str(i) + '.npy')
    sns.set_style('dark')
    fig, ax = plt.subplots(3, 11, figsize=(20, 10))
    plt.title('Result of Class'+str(i))
    ax = ax.ravel()
    for j in range(33):
        sns.boxplot(train_data[0+i*600:600+i*600, j], ax=ax[j], whis=3).set_title('Variable'+str(j))
        ax[j].axvline(x=reverse_result[j], c='green')
        ax[j].axvline(x=reverse_result_protected[j], c='red')
        # ax[j].set(xlim=(-1, 1))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig('./results/'+args.attack_type+'/resultClass_' + str(i) + '.svg')
    plt.close()






