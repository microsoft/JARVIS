#!/usr/bin/env python2

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('bmh')

import argparse

print("""

Note: This example assumes that `name i` corresponds to `label i`
in `labels.csv`.

""")

parser = argparse.ArgumentParser()
parser.add_argument('workDir', type=str)
parser.add_argument('--names', type=str, nargs='+', required=True)
args = parser.parse_args()

y = pd.read_csv("{}/labels.csv".format(args.workDir)).as_matrix()[:, 0]
X = pd.read_csv("{}/reps.csv".format(args.workDir)).as_matrix()

target_names = np.array(args.names)
colors = cm.Dark2(np.linspace(0, 1, len(target_names)))

X_pca = PCA(n_components=50).fit_transform(X, X)
tsne = TSNE(n_components=2, init='random', random_state=0)
X_r = tsne.fit_transform(X_pca)

for c, i, target_name in zip(colors,
                             list(range(1, len(target_names) + 1)),
                             target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],
                c=c, label=target_name)
plt.legend()

out = "{}/tsne.pdf".format(args.workDir)
plt.savefig(out)
print("Saved to: {}".format(out))
