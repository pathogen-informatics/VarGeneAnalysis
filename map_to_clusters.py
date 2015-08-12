#!/usr/bin/env python

import argparse
import csv
import logging
import numpy
import sys
import unittest

from collections import namedtuple, OrderedDict
from mock import patch
from sklearn import metrics
from sklearn.cluster import KMeans
from StringIO import StringIO

from split_other_samples import Split
from sample_stats import parse_sample

logging.basicConfig(level=logging.DEBUG)

DEFAULT_NUMBER_OF_CLUSTERS=10

class TrainingData(namedtuple('TestData', 'sample_names feature_names data')):
  @classmethod
  def from_file(cls, input_file):
    input_file.seek(0)
    csv_file = csv.reader(input_file, delimiter='\t')
    column_labels = numpy.array(csv_file.next()[1:]).astype('str')
    row_labels = numpy.array([row[0] for row in csv_file]).astype('str')
    input_file.seek(0)
    csv_file.next() # skip the header this time
    data = numpy.array([row[1:] for row in csv_file]).astype('float')
    return cls(row_labels, column_labels, data)
  
  def sort_sample_names(self, desired):
    row_lookup_table = {value: index for index,value in enumerate(self.sample_names)}
    lookup_row_indices = numpy.vectorize(row_lookup_table.get)
    new_row_ordering = lookup_row_indices(desired)
    return TrainingData(
             self.sample_names[new_row_ordering],
             self.feature_names,
             self.data[new_row_ordering,:]
           )
  
  def sort_feature_names(self, desired):
    column_lookup_table = {value: index for index,value in enumerate(self.feature_names)}
    lookup_column_indices = numpy.vectorize(column_lookup_table.get)
    new_column_ordering = lookup_column_indices(desired)
    return TrainingData(
             self.sample_names,
             self.feature_names[new_column_ordering],
             self.data[:,new_column_ordering]
           )
  
  def filter_sample_names(self, desired):
    """Filters data keeping rows from "original" which are in "desired"
  
    Preserves order of the retained rows in the "original" ordering"""
    check_in_desired = numpy.vectorize(lambda el: el in desired)
    rows_to_keep = check_in_desired(self.sample_names)
    return TrainingData(
             self.sample_names[rows_to_keep],
             self.feature_names,
             self.data[rows_to_keep,:]
           )
  
  def filter_feature_names(self, desired):
    """Filters data keeping columns from "original" which are in "desired"
  
    Preserves order of the retained columns in the "original" ordering"""
    check_in_desired = numpy.vectorize(lambda el: el in desired)
    columns_to_keep = check_in_desired(self.feature_names)
    return TrainingData(
             self.sample_names,
             self.feature_names[columns_to_keep],
             self.data[:,columns_to_keep]
           )
  
  def get_relevant_sample_names(self, desired_isolates):
    """Returns a list of the samples from the desired isolates"""
    def isolate_name(sample_name):
      isolate, gene, domain = parse_sample(sample_name)
      return isolate
  
    def is_desired_sample(sample_name):
      try:
        return isolate_name(sample_name) in desired_isolates
      except ValueError:
        return False
  
    desired_samples = filter(is_desired_sample, self.sample_names)
    return sorted(desired_samples)
  
  @classmethod
  def get_training_data(cls, input_file, isolate_names):
    """Returns data with a row for each training sample and a column for each
    feature.
  
    Columns are sorted alphabetically by feature name; rows remain unsorted.  The
    names of the training samples are taken to be the first N_SEQUENCES samples
    listed in the header row of input_file; the rest are assumed to be test
    data"""
    logging.info("Getting training data")
    training_data = cls.from_file(input_file)
    feature_names = training_data.get_relevant_sample_names(isolate_names)
    training_data = training_data.sort_feature_names(feature_names)
    training_data = training_data.filter_sample_names(feature_names)
  
    logging.debug("Found %s samples and %s features" %
                  (len(training_data.sample_names), len(training_data.feature_names)))
    logging.debug("First five sample names: %s..." % sorted(training_data.sample_names)[:5])
    return training_data
  
  def get_testing_data(self, input_file, isolate_names):
    logging.info("Getting testing data")
    test_data = TrainingData.from_file(input_file)
    test_names = test_data.get_relevant_sample_names(isolate_names)
    test_data = test_data.sort_feature_names(self.feature_names)
    test_data = test_data.filter_sample_names(test_names)
  
    logging.debug("Found %s samples and %s features" %
                  (len(test_data.sample_names), len(test_data.feature_names)))
    logging.debug("First five sample names: %s..." % sorted(test_data.sample_names)[:5])
    return test_data

  def subsample(self, p=0.7):
    number_to_keep = int(p*len(self.feature_names))
    features_to_keep = numpy.random.choice(self.feature_names, number_to_keep, replace=False)
    training_data = self.sort_feature_names(features_to_keep)
    training_data = training_data.filter_sample_names(features_to_keep)
    return training_data

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
  return dict(zip(test_data.sample_names, clusters))

def get_best_cluster_score_tuple(t1, t2):
  cluster_number_1, score_1 = t1
  cluster_number_2, score_2 = t2
  if score_1 > score_2:
    return t1
  elif score_1 < score_2:
    return t2
  elif cluster_number_2 < cluster_number_1:
    return t2
  else:
    return t1

def get_most_internally_consistent_cluster(scores):
  # NB scores is a 2D dictionary showing the similarity between two clusters (e.g. score[A][B])
  # it shouldn't normally contain score[A][A] but it shouldn't matter too much if it does
  top_5_scores = {cluster: sorted(cluster_scores.values(),reverse=True)[:5] for cluster, cluster_scores in scores.items()}
  mean_top_scores = {cluster: numpy.mean(cluster_scores) for cluster,cluster_scores in top_5_scores.items()}
  best_cluster_number, best_score = reduce(get_best_cluster_score_tuple, mean_top_scores.items())
  return best_cluster_number

def get_most_externally_consistent_cluster(all_scores, internally_consistent_cluster):
  # NB scores is a 1D dictionary mapping a single cluster to each of the other clusters in clusterings
  scores = all_scores[internally_consistent_cluster] 
  best_cluster_number, best_score = reduce(get_best_cluster_score_tuple, scores.items())
  return best_cluster_number

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
  parser.add_argument('split_file', type=argparse.FileType('r'))
  parser.add_argument('matrix_file_a', type=argparse.FileType('r'))
  parser.add_argument('matrix_file_b', type=argparse.FileType('r'))
  parser.add_argument('test_subset', type=str)
  parser.add_argument('training_subset_a', type=str)
  parser.add_argument('training_subset_b', type=str)
  parser.add_argument('output_file', type=argparse.FileType('a'), default=sys.stdout)
  parser.add_argument('-f', '--min-k', type=int, default=10)
  parser.add_argument('-t', '--max-k', type=int, default=50)
  parser.add_argument('-i', '--max-iters', type=int, default=300)
  parser.add_argument('-s', '--cluster-seeds', type=int, default=10)
  parser.add_argument('-r', '--repetitions', type=int, default=5)
  
  args = parser.parse_args()

  splits = {split.name: split for split in Split.from_file(args.split_file)}
  
  test_isolates = splits[args.test_subset].isolates
  training_isolates_a = splits[args.training_subset_a].isolates
  training_isolates_b = splits[args.training_subset_b].isolates

  training_data_a = TrainingData.get_training_data(args.matrix_file_a, training_isolates_a)
  logging.debug("Shape of training_data_a is %s by %s" % training_data_a.data.shape)
  testing_data_a = training_data_a.get_testing_data(args.matrix_file_a, test_isolates)
  logging.debug("Shape of testing_data_a is %s by %s" % testing_data_a.data.shape)
  training_data_b = TrainingData.get_training_data(args.matrix_file_b, training_isolates_b)
  logging.debug("Shape of training_data_b is %s by %s" % training_data_b.data.shape)
  testing_data_b = training_data_b.get_testing_data(args.matrix_file_b, test_isolates)
  logging.debug("Shape of testing_data_b is %s by %s" % testing_data_b.data.shape)

  train_classifier = get_classifier_trainer(n_init=args.cluster_seeds, max_iter=args.max_iters)

  for k in xrange(args.min_k, args.max_k):
    logging.debug("Running analysis for k=%s" % k)
    all_training_predictions_from_a = []
    all_test_predictions_from_a = []
    all_test_predictions_from_b = []
    for i in xrange(args.repetitions):
      logging.debug("Running iteration %s for k=%s" % (i, k))
      training_subsample_a = training_data_a.subsample()
      classifier_a = train_classifier(training_subsample_a, k=k)
      training_predictions_from_a = predict_cluster(classifier_a, training_subsample_a)
      all_training_predictions_from_a.append(training_predictions_from_a)

      testing_subsample_a = testing_data_a.sort_feature_names(training_subsample_a.feature_names)
      test_predictions_from_a = predict_cluster(classifier_a, testing_subsample_a)
      all_test_predictions_from_a.append(test_predictions_from_a)

      training_subsample_b = training_data_b.subsample()
      classifier_b = train_classifier(training_subsample_b, k=k)
      testing_subsample_b = testing_data_b.sort_feature_names(training_subsample_b.feature_names)
      test_predictions_from_b = predict_cluster(classifier_b, testing_subsample_b)
      all_test_predictions_from_b.append(test_predictions_from_b)

    logging.debug("Writing internal consistency scores for k=%s" % k)
    internal_score_matrix = {}
    for i, prediction_1 in enumerate(all_training_predictions_from_a[:-1]):
      for j, prediction_2 in enumerate(all_training_predictions_from_a[i+1:]):
        score = compare_cluster_maps(prediction_1, prediction_2)
        row = [k, "internal", score, i, i+j+1, args.max_iters, args.cluster_seeds]
        write_row(args.output_file, row)
        internal_score_matrix.setdefault(i, {})[j] = score
        internal_score_matrix.setdefault(j, {})[i] = score

    logging.debug("Writing external consistency scores for k=%s" % k)
    external_score_matrix = {}
    for i, prediction_1 in enumerate(all_test_predictions_from_a):
      for j, prediction_2 in enumerate(all_test_predictions_from_b):
        score = compare_cluster_maps(prediction_1, prediction_2)
        row = [k, "external", score, i, j, args.max_iters, args.cluster_seeds]
        write_row(args.output_file, row)
        external_score_matrix.setdefault(i, {})[j] = score

    logging.debug("Analysing the best clusters for k=%s" % k)
    best_cluster_a = get_most_internally_consistent_cluster(internal_score_matrix)
    best_cluster_b = get_most_externally_consistent_cluster(external_score_matrix, best_cluster_a)
    best_cluster = all_test_predictions_from_b[best_cluster_b]
    cluster_string = clusters_to_string(best_cluster)
    row = [k, "best", external_score_matrix[best_cluster_a][best_cluster_b], best_cluster_a, best_cluster_b, args.max_iters, args.cluster_seeds, cluster_string]
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
    training_data = TrainingData.get_training_data(input_file, ['train_1'])
    numpy.testing.assert_array_equal(training_data.sample_names, expected_sample_names)
    numpy.testing.assert_array_equal(training_data.feature_names, expected_feature_names)
    numpy.testing.assert_array_almost_equal(training_data.data, expected_training_data)

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
    training_data = TrainingData.get_training_data(input_file, ['train_1'])
    numpy.testing.assert_array_equal(training_data.sample_names, expected_sample_names)
    numpy.testing.assert_array_equal(training_data.feature_names, expected_feature_names)
    numpy.testing.assert_array_almost_equal(training_data.data, expected_training_data)

  def test_get_testing_data(self):
    input_file = StringIO("""\
id	train_1.g1.DBLa.1	train_2.g2.DBLa.2	test_3.g3.DBLa.3	test_4.g4.DBLa.4
test_3.g3.DBLa.3	0.3	0.5	0.0	0.9
train_2.g2.DBLa.2	0.2	0.0	0.5	0.8
train_1.g1.DBLa.1	0.0	0.2	0.3	0.7
test_4.g4.DBLa.4	0.7	0.8	0.9	0.0""")
    feature_names = numpy.array(['train_1.g1.DBLa.1', 'train_2.g2.DBLa.2']).astype('str')

    expected_sample_names = numpy.array(['test_3.g3.DBLa.3', 'test_4.g4.DBLa.4']).astype('str')
    expected_feature_names = numpy.array(['train_1.g1.DBLa.1', 'train_2.g2.DBLa.2']).astype('str')
    expected_testing_data = numpy.array([[0.3, 0.5], [0.7, 0.8]]).astype('float')
    training_data = TrainingData.get_training_data(input_file, ['train_1', 'train_2'])
    testing_data = training_data.get_testing_data(input_file, ['test_3', 'test_4'])
    numpy.testing.assert_array_equal(testing_data.sample_names, expected_sample_names)
    numpy.testing.assert_array_equal(testing_data.feature_names, expected_feature_names)
    numpy.testing.assert_array_almost_equal(testing_data.data, expected_testing_data)

  def test_get_testing_data_subset(self):
    input_file = StringIO("""\
id	train_1.g1.DBLa.1	train_2.g2.DBLa.2	test_3.g3.DBLa.3	test_4.g4.DBLa.4
test_3.g3.DBLa.3	0.3	0.5	0.0	0.9
train_2.g2.DBLa.2	0.2	0.0	0.5	0.8
train_1.g1.DBLa.1	0.0	0.2	0.3	0.7
test_4.g4.DBLa.4	0.7	0.8	0.9	0.0""")
    feature_names = numpy.array(['train_1.g1.DBLa.1', 'train_2.g2.DBLa.2']).astype('str')

    expected_sample_names = numpy.array(['test_3.g3.DBLa.3']).astype('str')
    expected_feature_names = numpy.array(['train_1.g1.DBLa.1', 'train_2.g2.DBLa.2']).astype('str')
    expected_testing_data = numpy.array([[0.3, 0.5]]).astype('float')
    training_data = TrainingData.get_training_data(input_file, ['train_1', 'train_2'])
    testing_data = training_data.get_testing_data(input_file, ['test_3'])
    numpy.testing.assert_array_equal(testing_data.sample_names, expected_sample_names)
    numpy.testing.assert_array_equal(testing_data.feature_names, expected_feature_names)
    numpy.testing.assert_array_almost_equal(testing_data.data, expected_testing_data)

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
