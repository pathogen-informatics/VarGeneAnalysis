#!/usr/bin/env python

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import style
from StringIO import StringIO

def load_without_best(raw_file):
  """Remove comments, the 7th column and 'best' rows"""
  fake_file = StringIO()
  for line in raw_file:
    line = re.sub('\s*#.+$', '', line)
    if line.strip() == '':
      continue
    fake_file.write("\t".join(line.split("\t")[:7])+"\n")
  fake_file.seek(0)
  data = pd.read_csv(fake_file, delimiter='\t', names=['k', 'type', 'score', 'i', 'j', 'i','r'])
  data = data[data['type'] != 'best']
  return data

def plot_kmeans(data, block=False, title=None):
  data['k'] = map(int, data['k'])
  stats = data[['k', 'type', 'score']].groupby(['k', 'type']).apply(lambda df: pd.Series({'mean': df['score'].mean(), 'std': df['score'].std()}))
  means = stats['mean'].unstack()[['external']].sort_index()
  errors = stats['std'].unstack()[['external']].sort_index()
  chart = means.plot(yerr=errors, kind='bar')
  if title:
    chart.set_title(title)
  plt.show(block=block)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--title', type=str, default=None)
  parser.add_argument('input_files', nargs='+', type=argparse.FileType('r'))
  args = parser.parse_args()

  style.use('ggplot')
  data = load_without_best(args.input_files[0])
  for input_file in args.input_files[1:]:
    data = data.append(load_without_best(input_file))

  plot_kmeans(data, block=True, title=args.title)
