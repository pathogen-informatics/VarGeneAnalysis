#!/usr/bin/env python

import argparse

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import style
from StringIO import StringIO

def load_without_best(raw_file):
  """Remove the 7th column and 'best' rows"""
  fake_file = StringIO()
  for line in raw_file:
    fake_file.write("\t".join(line.split("\t")[:7])+"\n")
  fake_file.seek(0)
  data = pd.read_csv(fake_file, delimiter='\t', names=['k', 'type', 'score', 'i', 'j', 'i','r'])
  data = data[data['type'] != 'best']
  return data

def plot_kmeans(data, block=False, title=None):
  stats = data[['k', 'type', 'score']].groupby(['k', 'type']).apply(lambda df: pd.Series({'mean': df['score'].mean(), 'std': df['score'].std()}))
#  means = stats['mean'].unstack()[['internal', 'external']]
#  errors = stats['std'].unstack()[['internal', 'external']]
  means = stats['mean'].unstack()[['external']]
  errors = stats['std'].unstack()[['external']]
  chart = means.plot(yerr=errors, kind='bar')
  if title:
    chart.set_title(title)
  plt.show(block=block)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--title', type=str, default=None)
  parser.add_argument('input_file', type=argparse.FileType('r'))
  args = parser.parse_args()

  style.use('ggplot')
  data = load_without_best(args.input_file)
  plot_kmeans(data, block=True, title=args.title)
