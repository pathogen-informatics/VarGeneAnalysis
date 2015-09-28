#!/usr/bin/env python

import argparse
import datetime
import logging
import re
import sys
import csv

import pandas as pd

from sklearn.metrics import adjusted_rand_score

def write_row(f, row):
  row_string = "\t".join(map(str, row))
  f.write(row_string + "\n")

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument("input_a", type=argparse.FileType('r'))
  parser.add_argument("input_b", type=argparse.FileType('r'))
  parser.add_argument("output_file", type=argparse.FileType('wb'), default=sys.stdout)

  args = parser.parse_args()

  csv_a = pd.read_csv(args.input_a, delimiter="\t")
  csv_b = pd.read_csv(args.input_b, delimiter="\t")

  # Get union of domains in a and b; report A AND !B / B AND !A
  domains_a = set(csv_a.columns.values[1:])
  domains_b = set(csv_b.columns.values[1:])
  domains_both = domains_a.intersection(domains_b)
  logging.info("A and B have the following domains in common: %s" %
               ', '.join(sorted(map(str, domains_both))))
  only_a = domains_a.difference(domains_b)
  only_b = domains_b.difference(domains_a)
  if only_a:
    logging.info("Only A has: %s" %
                 ', '.join(sorted(map(str, only_a))))
  if only_b:
    logging.info("Only B has: %s" %
                 ', '.join(sorted(map(str, only_b))))

  for domain in sorted(domains_both):
    joint_data = pd.merge(csv_a[['name', domain]],
                          csv_b[['name', domain]],
                          on='name', how='outer')
    joint_data.columns = ['name', 'A', 'B']
    unknown_sample_names = joint_data[pd.isnull(joint_data).any(axis=1)]['name']
    if not unknown_sample_names.empty:
      logging.debug("The following samples could not be classified:\n%s" %
                    "\n".join(map(str, unknown_sample_names)))
    joint_data[pd.isnull(joint_data)] = 'unknown'
    confusion_matrix = joint_data.groupby(['A', 'B']).aggregate(len).unstack()
    confusion_matrix[pd.isnull(confusion_matrix)] = 0
    args.output_file.write("Confusion matrix for %s:\n" % domain)
    confusion_matrix.to_csv(args.output_file, sep='\t')
    joint_data_only_known = joint_data[(joint_data != 'unknown').any(axis=1)]
    score = adjusted_rand_score(joint_data_only_known['A'], joint_data_only_known['B'])
    args.output_file.write("Score for %s: %s\n\n" % (domain, score))
