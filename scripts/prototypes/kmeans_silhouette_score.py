#!/usr/bin/env python

import argparse
import logging
import re
import sys

import pandas as pd

from sklearn.metrics import silhouette_score

def get_matrix_columns(distance_matrix, sample_order):
  logging.debug("Removing irrelevant matrix columns")
  return distance_matrix[['id'] + list(sample_order)]

def get_relevant_data(relevant_columns, sample_order):
  logging.debug("Sorting matrix rows")
  data = relevant_columns.set_index('id')
  return data.loc[sample_order, :]

def get_sorted_matrix(distance_matrix, sample_order):
  data = get_matrix_columns(distance_matrix, sample_order)
  return get_relevant_data(data, sample_order)

def get_best_clusters(kmeans_result):
  results = []
  kmeans_result.seek(0)
  for line in kmeans_result:
    line = re.sub('\s*#.+$', '', line).strip()
    if line == '':
      continue
    row = [el.strip() for el in line.split('\t')]
    try:
      k, row_type = row[:2]
    except:
      print line
      raise
    if row_type == 'best':
      clusters_string = row[-1].strip()
      clusters = [cluster.split(',') for cluster in clusters_string.split(';')]
      results.append((k, clusters))
  return results

def get_cluster_details(clusters):
  sample_names = []
  sample_labels = []
  for cluster_index, cluster in enumerate(clusters):
    sample_names += cluster
    sample_labels += ["cluster_%s" % cluster_index for c in cluster]
  return (sample_names, sample_labels)

if __name__ == '__main__':
  logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument('distance_matrix', type=argparse.FileType('r'))
  parser.add_argument('kmeans_results', type=argparse.FileType('r'), nargs='*')
  parser.add_argument('output_file'), type=argparse.FileType('w'), default=sys.stdout)
  args = parser.parse_args()

  distance_matrix = pd.read_csv(args.distance_matrix, delimiter='\t')

  best_clusters = []
  for kmeans_result in args.kmeans_results:
    best_clusters += get_best_clusters(kmeans_result)

  first_k, clusters = best_clusters[0]
  sample_order, sample_labels = get_cluster_details(clusters)
  sorted_samples = sorted(sample_order)
  sorted_distance_matrix = get_sorted_matrix(distance_matrix, sorted_samples)

  results = []
  for k, clusters in best_clusters:
    sample_order, sample_labels = get_cluster_details(clusters)
    if sorted(sample_order) != sorted_samples:
      logging.error("Samples clustered for k=%s were different from k=%s" % (k, first_k))
      continue
    cluster_lookup = dict(zip(sample_order, sample_labels))
    sorted_sample_labels = pd.DataFrame({'label': [cluster_lookup[label] for label in sorted_samples]})
    score = silhouette_score(sorted_distance_matrix.values,
                             sorted_sample_labels['label'].values,
                             metric="precomputed")
    results.append((k, score))
    args.output_file.write("%s\t%s\n" % (k, score))
