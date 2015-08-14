#!/usr/bin/env python

import argparse
import datetime
import logging
import re
import sys
import csv

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

  csv_a = csv.reader(args.input_a, delimiter="\t")
  csv_b = csv.reader(args.input_b, delimiter="\t")

  columns_a = csv_a.next()[1:]
  columns_b = csv_b.next()[1:]

  sample_cluster_map = {}
  row_cluster_names = {}
  logging.debug("Making our way through A")
  for row in csv_a:
    sample_name = row[0]
    sample_clusters = row[1:]
    for column_label,cluster_name_a in zip(columns_a, sample_clusters):
      sample_cluster_map.setdefault(column_label, {})[sample_name] = cluster_name_a
      row_cluster_names.setdefault(column_label, set()).add(cluster_name_a)
  for column_label, cluster_names in row_cluster_names.items():
    logging.debug("Found %s cluster names for %s in A" % (len(cluster_names), column_label))

  cluster_cluster_map = {}
  column_cluster_names = {}
  logging.debug("Making our way through B")
  for row in csv_b:
    sample_name = row[0]
    sample_clusters = row[1:]
    for column_label,cluster_name_b in zip(columns_b, sample_clusters):
      try:
        cluster_name_a = sample_cluster_map[column_label][sample_name]
      except:
        continue
      previous_count = cluster_cluster_map.setdefault(column_label, {}).setdefault(cluster_name_a, {}).setdefault(cluster_name_b, 0)
      cluster_cluster_map[column_label][cluster_name_a][cluster_name_b] = previous_count + 1
      column_cluster_names.setdefault(column_label, set()).add(cluster_name_b)

  for column_label, cluster_names in column_cluster_names.items():
    logging.debug("Found %s cluster names for %s in B" % (len(cluster_names), column_label))

  for column_name,cluster_matrix in cluster_cluster_map.items():
    logging.debug("Outputing matrix for %s" % column_name)
    logging.debug("Expecting a %s by %s matrix" % (len(column_cluster_names.get(column_name, [])), len(row_cluster_names.get(column_name, []))))
    header_row = [column_name] + sorted(column_cluster_names.get(column_name, []))
    header_row = [el if el != '' else 'unknown' for el in header_row]
    write_row(args.output_file, header_row)

    for row_cluster in sorted(row_cluster_names.get(column_name, [])):
      row = [row_cluster]
      for column_cluster in sorted(column_cluster_names.get(column_name, [])):
        row.append(cluster_matrix.get(row_cluster, {}).get(column_cluster, 0))
      row = [el if el != '' else 'unknown' for el in row]
      write_row(args.output_file, row)
    args.output_file.write("\n\n")
