import csv
import numpy
import unittest

from StringIO import StringIO

NUMBER_OF_CLUSTERS = 27
N_SAMPLES = 1000

cluster_matrices = ["CIDRa.Matrix.1000ac.txt", "CIDRa.Matrix.1000bc.txt"]

def get_sample_names():
  sample_sets = {}
  for filename in cluster_matrices:
    sample_file  = open(filename, 'r')
    header_row = csv.reader(sample_file, delimiter='\t').next()
    sample_names = header_row[1:]
    sample_sets[filename] = set(sample_names)
    sample_file.close()
  return sample_sets

def print_sample_similarity_matrix(sample_sets):
  print "\t".join([""] + sample_sets.keys())
  for i,(filename_a, set_a) in enumerate(sample_sets.items()):
    row = [filename_a] + [""]*i
    for filename_b,set_b in sample_sets.items()[i:]:
      row.append(len(set_a.intersection(set_b)))
    print "\t".join(map(str, row))

def get_training_sample_names(sample_names):
  return sample_names[:N_SAMPLES]

def get_test_sample_names(sample_names):
  return sample_names[N_SAMPLES:]

def get_data(input_file):
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

def filter_row_labels(row_labels, desired):
  check_in_desired = numpy.vectorize(lambda el: el in desired)
  labels_to_keep = check_in_desired(row_labels)
  return row_labels[labels_to_keep]

def get_training_data(input_file):
  """Returns data with a row for each training sample and a column for each
  feature.

  Columns are sorted alphabetically by feature name; rows remain unsorted.  The
  names of the training samples are taken to be the first N_SEQUENCES samples
  listed in the header row of input_file; the rest are assumed to be test
  data"""
  sample_names, row_labels, data = get_data(input_file)
  feature_names = numpy.sort(sample_names[:N_SAMPLES])
  training_data = sort_columns(data, sample_names, feature_names)
  training_data = filter_rows(training_data, row_labels, feature_names)
  training_sample_names = filter_row_labels(row_labels, feature_names)

  return training_sample_names, feature_names, training_data

def get_testing_data(input_file, feature_names):
  sample_names, row_labels, data = get_data(input_file)
  testing_names = numpy.sort(sample_names[N_SAMPLES:])
  testing_data = sort_columns(data, sample_names, feature_names)
  testing_data = filter_rows(testing_data, row_labels, testing_names)
  testing_sample_names = filter_row_labels(row_labels, testing_names)

  return testing_sample_names, testing_data

if __name__ == '__main__':
  sample_sets = get_sample_names()
  print_sample_similarity_matrix(sample_sets)

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
    global N_SAMPLES
    prev_N_SAMPLES = N_SAMPLES
    N_SAMPLES = 2
    input_file = StringIO("""\
id	train_1	train_2	test_3	test_4
test_3	0.3	0.5	0.0	0.9
train_2	0.2	0.0	0.5	0.8
train_1	0.0	0.2	0.3	0.7
test_4	0.7	0.8	0.9	0.0""")
    expected_sample_names = numpy.array(['train_2', 'train_1']).astype('str')
    expected_feature_names = numpy.array(['train_1', 'train_2']).astype('str')
    expected_training_data = numpy.array([[0.2, 0.0], [0.0, 0.2]]).astype('float')
    training_sample_names, feature_names, training_data = get_training_data(input_file)
    N_SAMPLES = prev_N_SAMPLES
    numpy.testing.assert_array_equal(training_sample_names, expected_sample_names)
    numpy.testing.assert_array_equal(feature_names, expected_feature_names)
    numpy.testing.assert_array_almost_equal(training_data, expected_training_data)

  def test_get_testing_data(self):
    global N_SAMPLES
    prev_N_SAMPLES = N_SAMPLES
    N_SAMPLES = 2
    input_file = StringIO("""\
id	train_1	train_2	test_3	test_4
test_3	0.3	0.5	0.0	0.9
train_2	0.2	0.0	0.5	0.8
train_1	0.0	0.2	0.3	0.7
test_4	0.7	0.8	0.9	0.0""")
    feature_names = numpy.array(['train_1', 'train_2']).astype('str')

    expected_sample_names = numpy.array(['test_3', 'test_4']).astype('str')
    expected_testing_data = numpy.array([[0.3, 0.5], [0.7, 0.8]]).astype('float')
    testing_sample_names, testing_data = get_testing_data(input_file,
                                                          feature_names)
    N_SAMPLES = prev_N_SAMPLES
    numpy.testing.assert_array_equal(testing_sample_names, expected_sample_names)
    numpy.testing.assert_array_almost_equal(testing_data, expected_testing_data)
