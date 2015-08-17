#!/usr/bin/env python

import argparse
import csv
import sys
import logging
import numpy

def get_sample_names(subsets, split_file):
  sample_names = []
  split_file.seek(0)
  for row in csv.reader(split_file, delimiter='\t'):
    subset_name, samples_string = row[0], row[3]
    if subset_name in subsets:
      sample_names += samples_string.strip().split(',')
  return sample_names

def get_column_labels(input_file):
  input_file.seek(0)
  header_row = csv.reader(input_file, delimiter='\t').next()
  return header_row[1:]

def build_row_filter(column_labels, desired_columns):
  def isolate_name(sample_name):
    try:
      isolate_name, gene, domain, _ = sample_name.split('.')
      return isolate_name
    except:
      return None

  desired_column_indexes = [index+1 for index,column_label in enumerate(column_labels) if isolate_name(column_label) in desired_columns]

  def row_filter(row, force=False):
    sample_name = row[0]
    if force or isolate_name(sample_name) in desired_columns:
      return [sample_name] + [row[c] for c in desired_column_indexes]
    return None

  return row_filter

def reformat_rows(input_file, row_filter):
  input_file.seek(0)
  rows = csv.reader(input_file, delimiter='\t')
  header_row = rows.next()
  yield row_filter(header_row, force=True)
  for row in rows:
    reformatted_row = row_filter(row)
    if reformatted_row is None:
      continue
    else:
      yield reformatted_row

def print_row(row):
  print "\t".join(map(str,row))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('split_file', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('full_matrix', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('subsets', type=str, nargs="*")
  
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG)

  sample_names = get_sample_names(args.subsets, args.split_file)
  logging.debug("Looking for the following isolates: %s" % ','.join(sample_names))

  column_labels = get_column_labels(args.full_matrix)
  logging.debug("Found the following sample names in '%s': %s" % (args.full_matrix.name, ','.join(column_labels)))

  row_filter = build_row_filter(column_labels, sample_names)
  for row in reformat_rows(args.full_matrix, row_filter):
    print_row(row)
