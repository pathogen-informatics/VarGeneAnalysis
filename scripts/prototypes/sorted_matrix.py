#!/usr/bin/env python

import argparse
import logging
import sys

import pandas as pd

from compare_lots_of_clusters import plot_heatmap, normalise_along_axis

def load_classification_data(classification_summary):
  data = pd.read_csv(classification_summary, delimiter='\t')
  data.columns = ['name', 'subdomain']
  return data

def sort_by_subdomain(classification_data):
  logging.debug("Sorting by subdomain")
  return classification_data.sort('subdomain')

def get_matrix_columns(distance_matrix, sample_order):
  logging.debug("Removing irrelevant matrix columns")
  return distance_matrix[['id'] + list(sample_order)]

def get_relevant_data(relevant_columns, sample_order):
  logging.debug("Sorting matrix rows")
  data = relevant_columns.set_index('id')
  return data.loc[sample_order, :]

def get_cluster_sizes_and_order(classification_data):
  logging.debug("Counting cluster sizes")
  data = classification_data.groupby('subdomain').aggregate(len)
  data = data.reset_index()
  data.columns = ['subdomain', 'count']
  return data.sort('subdomain')

if __name__ == '__main__':
  logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument('classification_summary', type=argparse.FileType('r'))
  parser.add_argument('distance_matrix', type=argparse.FileType('r'))
  args = parser.parse_args()

  logging.debug("Loading data")
  classification_data = load_classification_data(args.classification_summary)
  distance_matrix = pd.read_csv(args.distance_matrix, delimiter='\t')

  classification_data = sort_by_subdomain(classification_data)
  sample_order = classification_data['name']
  cluster_counts = get_cluster_sizes_and_order(classification_data)

  relevant_columns = get_matrix_columns(distance_matrix, sample_order)
  relevant_data = get_relevant_data(relevant_columns, sample_order)
  cluster_counts.to_csv(sys.stdout, sep='\t', index=False)
  logging.debug("Presenting graph")
  plot_heatmap(relevant_data, block=True)
