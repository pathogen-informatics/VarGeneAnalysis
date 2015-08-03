#!/usr/bin/env python

import csv
import random

N_SAMPLES = 1000
output_matrices = ["CIDRa.Matrix.1000a.txt", "CIDRa.Matrix.1000b.txt", "CIDRa.Matrix.1000c.txt"]

distance_matrix_file = open('../CIDRa.Matrix.15517.txt', 'r')
csv_file = csv.reader(distance_matrix_file, delimiter='\t')
header = csv_file.next()

sample_indices = list(enumerate(header))[1:]

all_samples = random.sample(sample_indices, N_SAMPLES*len(output_matrices))

def column_index(t):
  return t[0]

def keep_columns(row, columns):
  return [row[i] for i in [0] + columns]

def row_to_text(row):
  return "\t".join(row) + '\n'

output_files = {filename: open(filename, 'w') for filename in output_matrices}
columns_to_keep = {filename: [column for column,name in all_samples[i::len(output_matrices)]] for i,filename in enumerate(output_matrices)}
sample_file_map = {name: output_matrices[i % len(output_files)] for i,(column,name) in enumerate(all_samples)}

for output_filename in output_matrices:
  header_row = keep_columns(header, columns_to_keep[output_filename])
  output_files[output_filename].write(row_to_text(header_row))
for row in csv_file:
  sample_name = row[0]
  try:
    output_filename = sample_file_map[sample_name]
    output_file = output_files[output_filename]
    output_row = keep_columns(row, columns_to_keep[output_filename])
    output_file.write(row_to_text(output_row))
  except KeyError:
    pass

for output_file in output_files.values():
  output_file.close()
