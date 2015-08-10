#!/usr/bin/env python

import argparse
import csv
import logging
import numpy
import sys
import unittest

from sklearn import metrics
from sklearn.cluster import KMeans
from StringIO import StringIO

from split_other_samples import Split
from sample_stats import parse_sample

logging.basicConfig(level=logging.DEBUG)

DEFAULT_NUMBER_OF_CLUSTERS=10

def get_data(input_file):
  input_file.seek(0)
  csv_file = csv.reader(input_file, delimiter='\t')
  column_labels = numpy.array(csv_file.next()[1:]).astype('str')
  row_labels = numpy.array([row[0] for row in csv_file]).astype('str')
  input_file.seek(0)
  csv_file.next() # skip the header this time
  data = numpy.array([row[1:] for row in csv_file]).astype('float')
  return column_labels, row_labels, data

def sort_rows(data, original, desired):
  row_lookup_table = {value: index for index,value in enumerate(original)}
  lookup_row_indices = numpy.vectorize(row_lookup_table.get)
  new_row_ordering = lookup_row_indices(desired)
  return data[new_row_ordering,:]

def sort_columns(data, original, desired):
  column_lookup_table = {value: index for index,value in enumerate(original)}
  lookup_column_indices = numpy.vectorize(column_lookup_table.get)
  new_column_ordering = lookup_column_indices(desired)
  return data[:,new_column_ordering]

def filter_rows(data, original, desired):
  """Filters data keeping rows from "original" which are in "desired"

  Preserves order of the retained rows in the "original" ordering"""
  check_in_desired = numpy.vectorize(lambda el: el in desired)
  rows_to_keep = check_in_desired(original)
  return data[rows_to_keep,:]

def filter_columns(data, original, desired):
  """Filters data keeping columns from "original" which are in "desired"

  Preserves order of the retained columns in the "original" ordering"""
  check_in_desired = numpy.vectorize(lambda el: el in desired)
  columns_to_keep = check_in_desired(original)
  return data[:, columns_to_keep]

def filter_row_labels(row_labels, desired):
  check_in_desired = numpy.vectorize(lambda el: el in desired)
  labels_to_keep = check_in_desired(row_labels)
  return row_labels[labels_to_keep]

def filter_column_labels(column_labels, desired):
  check_in_desired = numpy.vectorize(lambda el: el in desired)
  labels_to_keep = check_in_desired(column_labels)
  return column_labels[labels_to_keep]

def get_relevant_sample_names(sample_names, desired_isolates):
  """Returns a list of the samples from the desired isolates"""
  def isolate_name(sample_name):
    isolate, gene, domain = parse_sample(sample_name)
    return isolate

  def is_desired_sample(sample_name):
    return isolate_name(sample_name) in desired_isolates

  desired_samples = filter(is_desired_sample, sample_names)
  return sorted(desired_samples)

def get_training_data(input_file, isolate_names):
  """Returns data with a row for each training sample and a column for each
  feature.

  Columns are sorted alphabetically by feature name; rows remain unsorted.  The
  names of the training samples are taken to be the first N_SEQUENCES samples
  listed in the header row of input_file; the rest are assumed to be test
  data"""
  logging.info("Getting training data")
  sample_names, row_labels, data = get_data(input_file)
  feature_names = get_relevant_sample_names(sample_names, isolate_names)
  training_data = sort_columns(data, sample_names, feature_names)
  training_data = filter_rows(training_data, row_labels, feature_names)
  training_sample_names = filter_row_labels(row_labels, feature_names)

  logging.debug("Found %s samples and %s features" %
                (len(training_sample_names), len(feature_names)))
  return training_sample_names, feature_names, training_data

def get_testing_data(input_file, feature_names, isolate_names):
  logging.info("Getting testing data")
  sample_names, row_labels, data = get_data(input_file)
  testing_names = get_relevant_sample_names(sample_names, isolate_names)
  testing_data = sort_columns(data, sample_names, feature_names)
  testing_data = filter_rows(testing_data, row_labels, testing_names)
  testing_sample_names = filter_row_labels(row_labels, testing_names)

  logging.debug("Found %s samples using %s features" %
                (len(testing_sample_names), len(feature_names)))
  return testing_sample_names, testing_data

def _train_classifier(training_data, k=DEFAULT_NUMBER_OF_CLUSTERS, n_init=10, max_iter=300):
  classifier = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, n_jobs=3)
  classifier.fit(training_data)
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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('split_file', type=argparse.FileType('r'))
  parser.add_argument('matrix_file_a', type=argparse.FileType('r'))
  parser.add_argument('matrix_file_b', type=argparse.FileType('r'))
  parser.add_argument('test_subset', type=str)
  parser.add_argument('training_subset_a', type=str)
  parser.add_argument('training_subset_b', type=str)
  parser.add_argument('output_file', type=argparse.FileType('wa'), default=sys.stdout)
  parser.add_argument('-f', '--min-k', type=int, default=10)
  parser.add_argument('-t', '--max-k', type=int, default=50)
  parser.add_argument('-i', '--max-iters', type=int, default=300)
  parser.add_argument('-s', '--cluster-seeds', type=int, default=10)
  parser.add_argument('-r', '--repetitions', type=int, default=10)
  
  args = parser.parse_args()

  splits = {split.name: split for split in Split.from_file(args.split_file)}
  
  test_isolates = splits[args.test_subset].isolates
  training_isolates_a = splits[args.training_subset_a].isolates
  training_isolates_b = splits[args.training_subset_b].isolates

  training_sample_names_a, feature_names_a, training_data_a = get_training_data(args.matrix_file_a, training_isolates_a)
  logging.debug("Shape of training_data_a is %s by %s" % training_data_a.shape)
  test_sample_names_a, testing_data_a = get_testing_data(args.matrix_file_a, feature_names_a, test_isolates)
  logging.debug("Shape of testing_data_a is %s by %s" % testing_data_a.shape)
  training_sample_names_b, feature_names_b, training_data_b = get_training_data(args.matrix_file_b, training_isolates_b)
  logging.debug("Shape of training_data_b is %s by %s" % training_data_b.shape)
  test_sample_names_b, testing_data_b = get_testing_data(args.matrix_file_b, feature_names_b, test_isolates)
  logging.debug("Shape of testing_data_b is %s by %s" % testing_data_b.shape)

  train_classifier = get_classifier_trainer(n_init=args.cluster_seeds, max_iter=args.max_iters)

  for k in xrange(args.min_k, args.max_k):
    classifier_a_original = train_classifier(training_data_a, k=k)
    training_clusters_a_original = classifier_a_original.predict(training_data_a)
    for i in xrange(args.repetitions):
      classifier_a = train_classifier(training_data_a, k=k)
      training_clusters_a = classifier_a.predict(training_data_a)
      # Check if input_file_a is clustered consistently
      internal_consistency_score = metrics.adjusted_rand_score(training_clusters_a_original,
                                                               training_clusters_a)
      testing_clusters_a = classifier_a.predict(testing_data_a)

      classifier_b = train_classifier(training_data_b, k=k)
      testing_clusters_b = classifier_b.predict(testing_data_b)
      # Check if clusters from input_a and input_b are consistent
      test_data_consistency_score = metrics.adjusted_rand_score(testing_clusters_a,
                                                                testing_clusters_b)

      # What is the distribution of cluster sizes like?
      cluster_sizes = ",".join(map(str,get_cluster_sizes(testing_clusters_a, k)))

      output_line = "\t".join(map(str,[k,internal_consistency_score,test_data_consistency_score,i,cluster_sizes]))
      args.output_file.write(output_line + "\n")
      args.output_file.flush()

class TestAll(unittest.TestCase):
  def test_sort_rows(self):
    expected = numpy.array([[1,2,3]]).astype('float').transpose()
    data = numpy.array([[3,2,1,4,5]]).astype('float').transpose()
    original = numpy.array(['third','second','first','forth','fifth']).astype('str')
    desired = numpy.array(['first','second','third']).astype('str')
    actual = sort_rows(data, original, desired)
    numpy.testing.assert_array_almost_equal(actual, expected)

  def test_filter_rows(self):
    expected = numpy.array([[3,2,1]]).astype('float').transpose()
    data = numpy.array([[3,2,1,4,5]]).astype('float').transpose()
    original = numpy.array(['third','second','first','forth','fifth']).astype('str')
    desired = numpy.array(['first','second','third']).astype('str')
    actual = filter_rows(data, original, desired)
    numpy.testing.assert_array_almost_equal(actual, expected)

  def test_sort_columns(self):
    expected = numpy.array([[1,2,3]]).astype('float')
    data = numpy.array([[3,2,1,4,5]]).astype('float')
    original = numpy.array(['third','second','first','forth','fifth']).astype('str')
    desired = numpy.array(['first','second','third']).astype('str')
    actual = sort_columns(data, original, desired)
    numpy.testing.assert_array_almost_equal(actual, expected)

  def test_get_data(self):
    input_file = StringIO("""\
id	sample_1	sample_2	sample_3
sample_2	0.2	0.0	0.5
sample_1	0.0	0.2	0.3
sample_3	0.3	0.5	0.0""")
    expected_column_labels = numpy.array(['sample_1', 'sample_2', 'sample_3']).astype('str')
    expected_row_labels = numpy.array(['sample_2', 'sample_1', 'sample_3']).astype('str')
    expected_data = numpy.array([[0.2, 0.0, 0.5], [0.0, 0.2, 0.3], [0.3, 0.5, 0.0]])
    actual_column_labels, actual_row_labels, actual_data = get_data(input_file)
    numpy.testing.assert_array_equal(actual_column_labels, expected_column_labels)
    numpy.testing.assert_array_equal(actual_row_labels, expected_row_labels)
    numpy.testing.assert_array_almost_equal(actual_data, expected_data)

  def test_get_training_data(self):
    input_file = StringIO("""\
id	train_1.g1.DBLa.1	train_1.g2.DBLa.1	test_3.g3.DBLa.3	test_4.g4.DBLa.4
test_3.g3.DBLa.3	0.3	0.5	0.0	0.9
train_1.g2.DBLa.1	0.2	0.0	0.5	0.8
train_1.g1.DBLa.1	0.0	0.2	0.3	0.7
test_4.g4.DBLa.4	0.7	0.8	0.9	0.0""")
    expected_sample_names = numpy.array(['train_1.g2.DBLa.1', 'train_1.g1.DBLa.1']).astype('str')
    expected_feature_names = numpy.array(['train_1.g1.DBLa.1', 'train_1.g2.DBLa.1']).astype('str')
    expected_training_data = numpy.array([[0.2, 0.0], [0.0, 0.2]]).astype('float')
    training_sample_names, feature_names, training_data = get_training_data(input_file, ['train_1'])
    numpy.testing.assert_array_equal(training_sample_names, expected_sample_names)
    numpy.testing.assert_array_equal(feature_names, expected_feature_names)
    numpy.testing.assert_array_almost_equal(training_data, expected_training_data)

  def test_get_training_data_subset(self):
    input_file = StringIO("""\
id	train_1.g1.DBLa.1	train_2.g2.DBLa.1	test_3.g3.DBLa.3	test_4.g4.DBLa.4
test_3.g3.DBLa.3	0.3	0.5	0.0	0.9
train_2.g2.DBLa.1	0.2	0.0	0.5	0.8
train_1.g1.DBLa.1	0.0	0.2	0.3	0.7
test_4.g4.DBLa.4	0.7	0.8	0.9	0.0""")
    expected_sample_names = numpy.array(['train_1.g1.DBLa.1']).astype('str')
    expected_feature_names = numpy.array(['train_1.g1.DBLa.1']).astype('str')
    expected_training_data = numpy.array([[0.0]]).astype('float')
    training_sample_names, feature_names, training_data = get_training_data(input_file, ['train_1'])
    numpy.testing.assert_array_equal(training_sample_names, expected_sample_names)
    numpy.testing.assert_array_equal(feature_names, expected_feature_names)
    numpy.testing.assert_array_almost_equal(training_data, expected_training_data)

  def test_get_testing_data(self):
    input_file = StringIO("""\
id	train_1.g1.DBLa.1	train_2.g2.DBLa.2	test_3.g3.DBLa.3	test_4.g4.DBLa.4
test_3.g3.DBLa.3	0.3	0.5	0.0	0.9
train_2.g2.DBLa.2	0.2	0.0	0.5	0.8
train_1.g1.DBLa.1	0.0	0.2	0.3	0.7
test_4.g4.DBLa.4	0.7	0.8	0.9	0.0""")
    feature_names = numpy.array(['train_1.g1.DBLa.1', 'train_2.g2.DBLa.2']).astype('str')

    expected_sample_names = numpy.array(['test_3.g3.DBLa.3', 'test_4.g4.DBLa.4']).astype('str')
    expected_testing_data = numpy.array([[0.3, 0.5], [0.7, 0.8]]).astype('float')
    testing_sample_names, testing_data = get_testing_data(input_file,
                                                          feature_names,
                                                          ['test_3', 'test_4'])
    numpy.testing.assert_array_equal(testing_sample_names, expected_sample_names)
    numpy.testing.assert_array_almost_equal(testing_data, expected_testing_data)

  def test_get_testing_data_subset(self):
    input_file = StringIO("""\
id	train_1.g1.DBLa.1	train_2.g2.DBLa.2	test_3.g3.DBLa.3	test_4.g4.DBLa.4
test_3.g3.DBLa.3	0.3	0.5	0.0	0.9
train_2.g2.DBLa.2	0.2	0.0	0.5	0.8
train_1.g1.DBLa.1	0.0	0.2	0.3	0.7
test_4.g4.DBLa.4	0.7	0.8	0.9	0.0""")
    feature_names = numpy.array(['train_1.g1.DBLa.1', 'train_2.g2.DBLa.2']).astype('str')

    expected_sample_names = numpy.array(['test_3.g3.DBLa.3']).astype('str')
    expected_testing_data = numpy.array([[0.3, 0.5]]).astype('float')
    testing_sample_names, testing_data = get_testing_data(input_file,
                                                          feature_names,
                                                          ['test_3'])
    numpy.testing.assert_array_equal(testing_sample_names, expected_sample_names)
    numpy.testing.assert_array_almost_equal(testing_data, expected_testing_data)
