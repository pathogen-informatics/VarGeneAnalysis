#!/usr/bin/env python

import argparse
import csv
import sys
import logging
import numpy

from map_to_clusters import get_data, filter_rows, filter_columns, \
                            filter_row_labels, filter_column_labels

def build_filter_function(desired):
  def parse_sample_name(sample_name):
    try:
      isolate_name, gene, domain, _ = sample_name.split('.')
      return isolate_name
    except:
      return None

  def filter_sample_names(sample_name):
    isolate_name = parse_sample_name(sample_name)
    return isolate_name in desired

  return numpy.vectorize(filter_sample_names)
    

def filter_rows(data, original, desired):
  """Filters data keeping rows from "original" which are in "desired"

  Preserves order of the retained rows in the "original" ordering"""
  row_filter = build_filter_function(desired)
  rows_to_keep = row_filter(original)
  return data[rows_to_keep,:]

def filter_columns(data, original, desired):
  """Filters data keeping columns from "original" which are in "desired"

  Preserves order of the retained columns in the "original" ordering"""
  column_filter = build_filter_function(desired)
  columns_to_keep = column_filter(original)
  return data[:, columns_to_keep]

def filter_row_labels(row_labels, desired):
  row_filter = build_filter_function(desired)
  rows_to_keep = row_filter(row_labels)
  return row_labels[rows_to_keep]

def filter_column_labels(column_labels, desired):
  column_filter = build_filter_function(desired)
  columns_to_keep = column_filter(column_labels)
  return column_labels[columns_to_keep]

def get_sample_names(subsets, split_file):
  sample_names = []
  for row in csv.reader(split_file, delimiter='\t'):
    subset_name, samples_string = row[0], row[3]
    if subset_name in subsets:
      sample_names += samples_string.strip().split(',')
  return list(set(sample_names))

def header_row(column_labels):
  return numpy.array([[""] + list(column_labels)])

def print_data(column_labels, row_labels, data):
  row_labels = numpy.array([row_labels]).T
  data_rows = numpy.append(row_labels, data, axis=1)
  all_rows = numpy.append(header_row(column_labels), data_rows, axis=0)
  for row in all_rows:
    print "\t".join(map(str,row))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('split_file', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('full_matrix', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('subsets', type=str, nargs="*")
  
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARN)

  sample_names = get_sample_names(args.subsets, args.split_file)
  logging.debug("Looking for the following isolates: %s" % ','.join(sample_names))

  column_labels, row_labels, data = get_data(args.full_matrix)
  logging.debug("Found the following sample names in '%s': %s" % (args.full_matrix.name, ','.join(column_labels)))
  relevant_rows = filter_rows(data, row_labels, sample_names)
  relevant_data = filter_columns(relevant_rows, column_labels, sample_names)
  relevant_row_labels = filter_row_labels(row_labels, sample_names)
  relevant_column_labels = filter_column_labels(column_labels, sample_names)

  print_data(relevant_column_labels, relevant_row_labels, relevant_data)
