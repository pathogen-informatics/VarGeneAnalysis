#!/usr/bin/env python

import argparse
import random
import sys

import pandas as pd

from sklearn.decomposition import PCA

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--threshold', type=int, default=10)
  parser.add_argument('-b', '--bins', type=int, default=10)
  parser.add_argument('-c', '--components', type=int, default=6)
  parser.add_argument('input_matrix', type=argparse.FileType('r'),
                      default=sys.stdin)
  parser.add_argument('output_matrix', type=argparse.FileType('w'),
                      default=sys.stdout)
  args = parser.parse_args()

  full_matrix = pd.read_csv(args.input_matrix, delimiter='\t')
  full_matrix.set_index('id', inplace=True)

  pca = PCA(n_components=args.components)
  data = pca.fit_transform(full_matrix)

  pc_matrix = pd.DataFrame(columns=range(args.components),
                           index=full_matrix.index.values,
                           data=data)
  bins = pd.DataFrame({k: pd.cut(pc_matrix[k], args.bins, retbins=False, labels=False)
                       for k in pc_matrix}, index=pc_matrix.index.values)
  bins['bin'] = pd.DataFrame({k: bins[k]*args.bins**k for k in bins}).sum(axis=1)
  bins.drop(range(args.components), axis=1, inplace=True)
  bins.reset_index(inplace=True)
  bins.columns=['sample', 'bin']

  samples_to_keep = set()
  for bin_number, df in bins.groupby('bin'):
    samples = df['sample']
    if len(samples) > args.threshold:
      samples_to_keep.update(random.sample(samples, args.threshold))
    else:
      samples_to_keep.update(samples)

  output_matrix = full_matrix.loc[list(samples_to_keep), list(samples_to_keep)]
  output_matrix.to_csv(args.output_matrix, sep='\t')
