#!/usr/bin/env python

import argparse
import random
import sys

import pandas as pd

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--threshold', type=float, default=0.05,
                      help='Finds two (or more) samples which are < threshold in the distance matrix and throws all but one away (default: 0.05)')
  parser.add_argument('input_matrix', type=argparse.FileType('r'),
                      default=sys.stdin)
  parser.add_argument('output_matrix', type=argparse.FileType('w'),
                      default=sys.stdout)
  args = parser.parse_args()

  full_matrix = pd.read_csv(args.input_matrix, delimiter='\t')
  full_matrix.set_index('id', inplace=True)
  
  distance_pairs = full_matrix.unstack().reset_index()
  distance_pairs.columns = ['A', 'B', 'distance']
  distance_pairs = distance_pairs[distance_pairs['A'] < distance_pairs['B']]
  
  distance_groups = distance_pairs[distance_pairs['distance'] <= args.threshold].groupby('A')
  duplicates = []
  for A, Bs in distance_groups:
    duplicates.append(list(set(list(Bs['B']) + [A])))
  
  keepers = set()
  removed = set()
  for samples in duplicates:
    samples = list(set(samples).difference(keepers).difference(removed))
    if len(samples) >= 1:
      keepers.add(samples.pop(random.randint(0, len(samples)-1)))
    rows_to_remove = set(samples).intersection(full_matrix.index.values)  
    columns_to_remove = set(samples).intersection(full_matrix.columns.values)  
    full_matrix.drop(rows_to_remove, inplace=True)
    full_matrix.drop(columns_to_remove, axis=1, inplace=True)
    removed.update(samples)

  full_matrix.to_csv(args.output_matrix, sep='\t')
