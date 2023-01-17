import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.DataFrame(columns=['base', 'dgc'], index=range(300))
df['base'] = np.load('results/dlg/baseline_test_acc.npy')
df['dgc'] = np.load('results/dlg/dgc0.005_test_acc.npy')
sns.set_style('dark')
sns.lineplot(data=df)
plt.legend(labels=['base', 'dgc'], loc='lower right')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('./results/dlg/test_acc_base_dgc.svg')
plt.show(block=True)