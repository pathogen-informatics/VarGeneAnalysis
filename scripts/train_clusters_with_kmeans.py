#!/usr/bin/env python

import argparse
import csv
import logging
import numpy
import sys
import unittest

import pandas as pd

from collections import namedtuple, OrderedDict
from mock import patch
from sklearn import metrics
from sklearn.cluster import KMeans
from StringIO import StringIO

from create_subsets_for_other_domains import Split

logging.basicConfig(level=logging.DEBUG)

DEFAULT_NUMBER_OF_CLUSTERS=10

class TrainingData(object):
  @classmethod
  def from_file(cls, input_file):
    try:
      input_file.seek(0)
    except AttributeError:
      input_file = open(input_file, 'r')
    csv_file = pd.read_csv(input_file, delimiter='\t')
    csv_file.set_index('id', inplace=True)
    return cls(csv_file)

  def __init__(self, data):
    self.data = data
    logging.debug("Creating new data object: %s by %s" % self.data.shape)
    self.feature_names = data.columns.values
    self.sample_names = data.index.values

  def sort_sample_names(self, desired):
    return TrainingData(self.data.loc[desired])

  def sort_feature_names(self, desired):
    return TrainingData(self.data.loc[:,desired])

  def filter_sample_names(self, desired):
    """Filters data keeping rows from "original" which are in "desired"

    Preserves order of the retained rows in the "original" ordering"""
    matching_rows = self.data.index[self.data.index.isin(desired)]
    non_matching_data = self.data.loc[matching_rows]
    return TrainingData(non_matching_data)

  def filter_feature_names(self, desired):
    """Filters data keeping columns from "original" which are in "desired"

    Preserves order of the retained columns in the "original" ordering"""
    matching_columns = self.data.columns[self.data.columns.isin(desired)]
    non_matching_data = self.data.loc[:, matching_columns]
    return TrainingData(non_matching_data)

  def get_relevant_sample_names(self, desired_isolates):
    """Returns a list of the samples from the desired isolates"""
    def is_desired_sample(sample_name):
      try:
        isolate_name, gene, domain, position = sample_name.split('.')
      except ValueError:
        return False
      return isolate_name in desired_isolates

    desired_samples = filter(is_desired_sample, self.sample_names)
    return sorted(desired_samples)

  def remove(self, sample_names):
    matching_rows = self.data.index[~self.data.index.isin(sample_names)]
    matching_columns = self.data.columns[~self.data.columns.isin(sample_names)]
    non_matching_data = self.data.loc[matching_rows, matching_columns]
    return TrainingData(non_matching_data)

  def subsample(self, p=0.7, remove_features=True):
    some_data = TrainingData(self.data.sample(frac=p, replace=False))
    if remove_features:
      some_data = some_data.filter_feature_names(some_data.sample_names) 
    return some_data

def _train_classifier(training_data, k=DEFAULT_NUMBER_OF_CLUSTERS, n_init=10, max_iter=300):
  classifier = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, n_jobs=1)
  classifier.fit(training_data.data)
  logging.debug("Trained a classifier with %s clusters" % k)
  return classifier

def get_classifier_trainer(n_init=10, max_iter=300):
  def classifier(data, k):
    return _train_classifier(data, k=k, n_init=n_init, max_iter=max_iter)
  return classifier

def get_cluster_sizes(sample_labels, k):
  cluster_sizes = {cluster: 0 for cluster in xrange(k)}
  for cluster in sample_labels:
    cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1
  return sorted(cluster_sizes.values())

def compare_cluster_maps(cluster_map_a, cluster_map_b):
  def by_sample_name(t):
    sample_name, cluster_number = t
    return sample_name
  cluster_a_samples = set(cluster_map_a.keys())
  cluster_b_samples = set(cluster_map_b.keys())
  both_clusters_samples = cluster_a_samples.intersection(cluster_b_samples)
  logging.debug("Comparing clusters of length %s and %s: overlap is %s" % tuple(map(len, [cluster_a_samples, cluster_b_samples, both_clusters_samples])))
  cluster_list_a = [cluster_map_a[sample] for sample in both_clusters_samples]
  cluster_list_b = [cluster_map_b[sample] for sample in both_clusters_samples]
  if len(cluster_list_a) == 0:
    first_five_a = sorted(cluster_a_samples)[:5]
    first_five_b = sorted(cluster_b_samples)[:5]
    logging.warn("No overlap between clusters in a and b.  First five shown: %s... => %s..." % (first_five_a, first_five_b))
    return 0.0
  return metrics.adjusted_rand_score(cluster_list_a, cluster_list_b)

def predict_cluster(classifier, test_data):
  clusters = classifier.predict(test_data.data)
  predictions = dict(zip(test_data.sample_names, clusters))
  logging.debug("Produced predictions")
  return predictions

def clusters_to_string(cluster):
  cluster_map = {}
  for sample_name,cluster_number in cluster.items():
    cluster_map.setdefault(cluster_number, []).append(sample_name)
  for cluster_number in cluster_map:
    cluster_map[cluster_number].sort()
  def by_cluster_size(t):
    cluster_number,sample_names = t
    return len(sample_names)
  cluster_strings = (",".join(sample_names) for cluster_number,sample_names in sorted(cluster_map.items(), key=by_cluster_size))
  return ";".join(cluster_strings)

def write_row(output_file, row):
  output_line = "\t".join(map(str, row))
  output_file.write(output_line + "\n")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('distance_matrix', type=argparse.FileType('r'))
  parser.add_argument('output_file', type=argparse.FileType('a'), default=sys.stdout)
  parser.add_argument('-f', '--min-k', type=int, default=10)
  parser.add_argument('-t', '--max-k', type=int, default=50)
  parser.add_argument('-i', '--max-iters', type=int, default=300)
  parser.add_argument('-s', '--cluster-seeds', type=int, default=10)
  parser.add_argument('-r', '--repetitions', type=int, default=5)

  args = parser.parse_args()

  all_data = TrainingData.from_file(args.distance_matrix)
  logging.debug("Shape of all_data is %s by %s" % all_data.data.shape)

  train_classifier = get_classifier_trainer(n_init=args.cluster_seeds, max_iter=args.max_iters)

  for k in xrange(args.min_k, args.max_k):
    logging.debug("Running analysis for k=%s" % k)
    scores = []
    for i in xrange(args.repetitions):
      logging.debug("Running iteration %s for k=%s" % (i, k))
      test_data = all_data.subsample(p=0.2, remove_features=False)
      training_data = all_data.remove(test_data.sample_names)

      logging.debug("Training on set a for i=%s, k=%s" % (i, k))
      training_subsample_a = training_data.subsample(p=0.7, remove_features=True)
      classifier_a = train_classifier(training_subsample_a, k=k)
      testing_subsample_a = test_data.sort_feature_names(training_subsample_a.feature_names)
      test_predictions_from_a = predict_cluster(classifier_a, testing_subsample_a)
      del classifier_a

      logging.debug("Training on set b for i=%s, k=%s" % (i, k))
      training_subsample_b = training_data.subsample(p=0.7, remove_features=True)
      classifier_b = train_classifier(training_subsample_b, k=k)
      testing_subsample_b = test_data.sort_feature_names(training_subsample_b.feature_names)
      test_predictions_from_b = predict_cluster(classifier_b, testing_subsample_b)

      logging.debug("Creating predictions for i=%s, k=%s" % (i, k))
      all_predictions = predict_cluster(classifier_b, all_data.sort_feature_names(training_subsample_b.feature_names))
      del classifier_b

      logging.debug("Calculating scores for i=%s, k=%s" % (i, k))
      score = compare_cluster_maps(test_predictions_from_a, test_predictions_from_b)
      row = [k, "external", score, i, i, args.max_iters, args.cluster_seeds]
      write_row(args.output_file, row)
      scores.append((score, i, all_predictions))

    logging.debug("Analysing the best clusters for k=%s" % k)
    # take the predictions in the 80th percentile
    score, i, predictions = sorted(scores)[int(args.repetitions*0.8)]
    cluster_string = clusters_to_string(predictions)
    row = [k, "best", score, i, i, args.max_iters, args.cluster_seeds, cluster_string]
    write_row(args.output_file, row)

    args.output_file.flush()

class TestAll(unittest.TestCase):
  def test_sort_sample_names(self):
    expected = numpy.array([[1,2,3]]).astype('float').transpose()
    data = numpy.array([[3,2,1,4,5]]).astype('float').transpose()
    original = numpy.array(['third','second','first','forth','fifth']).astype('str')
    desired = numpy.array(['first','second','third']).astype('str')
    features = numpy.array(['sample_1'])
    training_data = TrainingData(original, features, data)
    actual = training_data.sort_sample_names(desired)
    numpy.testing.assert_array_almost_equal(actual.data, expected)

  def test_filter_sample_names(self):
    expected = numpy.array([[3,2,1]]).astype('float').transpose()
    data = numpy.array([[3,2,1,4,5]]).astype('float').transpose()
    original = numpy.array(['third','second','first','forth','fifth']).astype('str')
    desired = numpy.array(['first','second','third']).astype('str')
    features = numpy.array(['sample_1'])
    training_data = TrainingData(original, features, data)
    actual = training_data.filter_sample_names(desired)
    numpy.testing.assert_array_almost_equal(actual.data, expected)

  def test_sort_feature_names(self):
    expected = numpy.array([[1,2,3]]).astype('float')
    data = numpy.array([[3,2,1,4,5]]).astype('float')
    original = numpy.array(['third','second','first','forth','fifth']).astype('str')
    desired = numpy.array(['first','second','third']).astype('str')
    samples = numpy.array(['sample_1'])
    training_data = TrainingData(samples, original, data)
    actual = training_data.sort_feature_names(desired)
    numpy.testing.assert_array_almost_equal(actual.data, expected)

  def test_from_file(self):
    input_file = StringIO("""\
id	sample_1	sample_2	sample_3
sample_2	0.2	0.0	0.5
sample_1	0.0	0.2	0.3
sample_3	0.3	0.5	0.0""")
    expected_column_labels = numpy.array(['sample_1', 'sample_2', 'sample_3']).astype('str')
    expected_row_labels = numpy.array(['sample_2', 'sample_1', 'sample_3']).astype('str')
    expected_data = numpy.array([[0.2, 0.0, 0.5], [0.0, 0.2, 0.3], [0.3, 0.5, 0.0]])
    actual_data = TrainingData.from_file(input_file)
    numpy.testing.assert_array_equal(actual_data.feature_names, expected_column_labels)
    numpy.testing.assert_array_equal(actual_data.sample_names, expected_row_labels)
    numpy.testing.assert_array_almost_equal(actual_data.data, expected_data)

  @patch("%s.metrics.adjusted_rand_score" % __name__)
  def test_compare_cluster_maps(self, score_mock):
    cluster_a = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4
    }
    cluster_b = {
            'a': 1,
            'd': 5,
            'c': 4,
            'e': 3,
            'f': 7
    }
    score_mock.return_value = 0.0
    self.assertEqual(compare_cluster_maps(cluster_a, cluster_b), 0.0)
    score_mock.assert_called_with([1,3,4], [1,4,5])

  def test_get_most_internally_consistent_cluster(self):
    clusterings = [
      numpy.array([0,1,2,2,2]),
      numpy.array([2,1,0,0,0]),
      numpy.array([0,1,2,2,2]),
      numpy.array([2,1,2,2,2]),
      numpy.array([1,1,2,2,2]),
      numpy.array([0,1,2,2,0]),
      numpy.array([0,1,0,1,2]),
      numpy.array([0,1,1,0,2])
    ]
    scores = {
      0: {         1:  1.0,  2:  1.0,  3:  0.4,  4:  0.8,  5:  0.2,  6: -0.3,  7: -0.3 },
      1: {0:  1.0,           2:  1.0,  3:  0.4,  4:  0.8,  5:  0.2,  6: -0.3,  7: -0.3 },
      2: {0:  1.0, 1:  1.0,            3:  0.4,  4:  0.8,  5:  0.2,  6: -0.3,  7: -0.3 },
      3: {0:  0.4, 1:  0.4,  2:  0.4,            4:  0.2,  5:  0.3,  6: -0.1,  7: -0.1 },
      4: {0:  0.8, 1:  0.8,  2:  0.8,  3:  0.2,            5:  0.1,  6: -0.4,  7: -0.4 },
      5: {0:  0.2, 1:  0.2,  2:  0.2,  3:  0.3,  4:  0.1,            6: -0.25, 7: -0.25},
      6: {0: -0.3, 1: -0.3,  2: -0.3,  3: -0.1,  4: -0.4,  5: -0.25,           7: -0.25},
      7: {0: -0.3, 1: -0.3,  2: -0.3,  3: -0.1,  4: -0.4,  5: -0.25, 6: -0.25          }
    }
    actual = get_most_internally_consistent_cluster(scores)
    numpy.testing.assert_array_almost_equal(actual, 0)

  def test_get_most_externally_consistent_cluster(self):
    scores = {
      0: {         1:  1.0,  2:  1.0,  3:  0.4,  4:  0.8,  5:  0.2,  6: -0.3,  7: -0.3 },
      1: {0:  1.0,           2:  1.0,  3:  0.4,  4:  0.8,  5:  0.2,  6: -0.3,  7: -0.3 },
      2: {0:  1.0, 1:  1.0,            3:  0.4,  4:  0.8,  5:  0.2,  6: -0.3,  7: -0.3 },
      3: {0:  0.4, 1:  0.4,  2:  0.4,            4:  0.2,  5:  0.3,  6: -0.1,  7: -0.1 },
      4: {0:  0.8, 1:  0.8,  2:  0.8,  3:  0.2,            5:  0.1,  6: -0.4,  7: -0.4 },
      5: {0:  0.2, 1:  0.2,  2:  0.2,  3:  0.3,  4:  0.1,            6: -0.25, 7: -0.25},
      6: {0: -0.3, 1: -0.3,  2: -0.3,  3: -0.1,  4: -0.4,  5: -0.25,           7: -0.25},
      7: {0: -0.3, 1: -0.3,  2: -0.3,  3: -0.1,  4: -0.4,  5: -0.25, 6: -0.25          }
    }
    actual = get_most_externally_consistent_cluster(scores, 0)
    numpy.testing.assert_array_almost_equal(actual, 1)

  def test_clusters_to_string(self):
    # clusters is normally just a dict not an OrderedDict
    clusters = OrderedDict([
      ('d', 1),
      ('b', 0),
      ('a', 0),
      ('c', 1),
      ('e', 1),
      ('f', 2)
    ])
    expected = "f;a,b;c,d,e"
    actual = clusters_to_string(clusters)
    self.assertEqual(actual, expected)
