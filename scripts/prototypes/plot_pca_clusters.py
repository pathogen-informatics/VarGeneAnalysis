#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib import style
from sklearn.decomposition import PCA

def load_distance_matrix(matrix_file):
  matrix = pd.read_csv(matrix_file, delimiter='\t')
  matrix.set_index('id', inplace=True)
  return matrix

def train_pca(distance_matrix, n_components=20):
  pca = PCA(n_components=n_components)
  data = pca.fit_transform(distance_matrix)
  columns = ['PC_%s' % i for i in range(1,n_components+1)]
  matrix_components = pd.DataFrame(data=data, index=distance_matrix.index,
                                   columns=columns)
  return pca, matrix_components

def get_labels(classification_file, samples_order):
  data = pd.read_csv(classification_file, delimiter='\t')
  data.columns = ['name', 'subdomain']
  label_lookup = dict(zip(data['name'], data['subdomain']))
  return [label_lookup[sample] for sample in samples_order]

def pca_scatter_plot(data, axes, labels, title, variance, block=False):
  n_labels = len(set(data[labels]))
  assert n_labels < 40, "Assumes that there are fewer than 40 subdomains"
  markers = (['>', 'o', '*', 'x'] * 10)[:n_labels]
  g = sns.lmplot(axes[0], axes[1], hue=labels, data=data, fit_reg=False,
                 markers=markers)
  x_axis_label = "%s (variance: %.1f%%)" % (axes[0], variance[0]*100)
  y_axis_label = "%s (variance: %.1f%%)" % (axes[1], variance[1]*100)
  g.set_axis_labels(x_axis_label, y_axis_label)
  if title:
    plt.suptitle(title)
  plt.show(block=block)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--title', type=str, default=None)
  parser.add_argument('distance_matrix', type=argparse.FileType('r'))
  parser.add_argument('classification_table', type=argparse.FileType('r'))
  args = parser.parse_args()

  style.use('ggplot')

  distance_matrix = load_distance_matrix(args.distance_matrix)
  pca, matrix_components = train_pca(distance_matrix, n_components=20)
  samples_order = matrix_components.index.values
  samples_labels = get_labels(args.classification_table, samples_order)
  matrix_components['subdomain'] = samples_labels
  pca_scatter_plot(data=matrix_components,
                   axes=['PC_1', 'PC_2'],
                   labels='subdomain',
                   title=args.title,
                   variance=pca.explained_variance_ratio_[0:2],
                   block=False)
  pca_scatter_plot(data=matrix_components,
                   axes=['PC_3', 'PC_4'],
                   labels='subdomain',
                   title=args.title,
                   variance=pca.explained_variance_ratio_[2:4],
                   block=False)
  pca_scatter_plot(data=matrix_components,
                   axes=['PC_5', 'PC_6'],
                   labels='subdomain',
                   title=args.title,
                   variance=pca.explained_variance_ratio_[5:7],
                   block=False)
  pca_scatter_plot(data=matrix_components,
                   axes=['PC_7', 'PC_8'],
                   labels='subdomain',
                   title=args.title,
                   variance=pca.explained_variance_ratio_[7:9],
                   block=True)
