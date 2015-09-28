#!/usr/bin/env python

import argparse
import csv
import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.DEBUG)

def get_sample_name_order_lookup(matrix_file):
  matrix_file.seek(0)
  # Ignore the header
  args.matrix_file.next()
  csv_matrix = csv.reader(args.matrix_file, delimiter='\t')
  sample_name_order = enumerate((row[0] for row in csv_matrix))
  return {sample_name: i for i, sample_name in sample_name_order}

def sort_summary_file(csv_summary_file, sample_name_order_lookup):
  csv_summary_file['order'] = map(sample_name_order_lookup.get,
                                  csv_summary_file['name'])

  missing_samples = csv_summary_file[pd.isnull(csv_summary_file).any(axis=1)]['name']
  if not missing_samples.empty:
    logging.error("Could not find the following in the matrix:\n%s" %
                  "\n".join((sorted(map(str, missing_samples)))))
    
  csv_summary_file = csv_summary_file[~pd.isnull(csv_summary_file).any(axis=1)]
  csv_summary_file.sort(columns='order', inplace=True)
  del csv_summary_file['order']
  return csv_summary_file

def add_colours(csv_summary_file):
  colours = "blue,brown,cadetblue,chartreuse4,chocolate4,cyan,darkblue,darkgrey,darkgreen,darkmagenta,darkorange,darkolivegreen1".split(',')
  all_subdomains=sorted(list(set(csv_summary_file['subdomain'])))
  colour_map = dict(zip(all_subdomains,colours))
  csv_summary_file['colour'] = map(colour_map.get,
                                   csv_summary_file['subdomain'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('matrix_file', type=argparse.FileType('r'))
  parser.add_argument('summary_file', type=argparse.FileType('r'))
  parser.add_argument('sorted_summary_file', type=argparse.FileType('w'),
                      default=sys.stdout)

  args = parser.parse_args()

  sample_name_order_lookup = get_sample_name_order_lookup(args.matrix_file)

  csv_summary_file = pd.read_csv(args.summary_file, delimiter='\t')
  csv_summary_file.columns=['name', 'subdomain']
  csv_summary_file = sort_summary_file(csv_summary_file, sample_name_order_lookup)
  add_colours(csv_summary_file)

  csv_summary_file.to_csv(args.sorted_summary_file, sep='\t', index=False)
