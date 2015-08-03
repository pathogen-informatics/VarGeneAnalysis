#!/usr/bin/env python

import csv
import logging
import random

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

N_SAMPLES = 1000
output_matrices = ["CIDRa.Matrix.1000ac.txt", "CIDRa.Matrix.1000bc.txt"]
SOURCE_MATRIX = '../CIDRa.Matrix.15517.txt'

logger.info("Picking %s training and one test set of %s samples from %s" % (len(output_matrices), N_SAMPLES, SOURCE_MATRIX))

distance_matrix_file = open(SOURCE_MATRIX, 'r')
csv_file = csv.reader(distance_matrix_file, delimiter='\t')
header = csv_file.next()
logger.debug("Read header from distance_matrix")

sample_indices = list(enumerate(header))[1:]

def just_column_indices(samples):
  return map(column_index, samples)

def column_index(t):
  return t[0]

logger.debug("Picking some random samples")
all_samples = random.sample(sample_indices, N_SAMPLES*(len(output_matrices) + 1))

logger.debug("Mapping training samples to output files and vice versa")
training_samples_per_file = {filename: all_samples[i*N_SAMPLES:(i+1)*N_SAMPLES] for i,filename in enumerate(output_matrices)}
file_for_training_sample = {sample: filename for filename,samples in training_samples_per_file.items() for i,sample in samples}

logger.debug("Calculating test column names and indices")
test_sample_names = [sample for index,sample in all_samples[len(output_matrices)*N_SAMPLES:]]
test_sample_columns = [index for index,sample in all_samples[len(output_matrices)*N_SAMPLES:]]

logger.debug("Calculating which columns to include in each file")
columns_per_file = {filename: just_column_indices(samples) + test_sample_columns for filename, samples in training_samples_per_file.items()}

logger.debug("Creating the output files")
output_files = {filename: open(filename, 'w') for filename in output_matrices}

def keep_columns(row, columns):
  return [row[i] for i in [0] + columns]

def row_to_text(row):
  return "\t".join(row) + '\n'

def write_row_to_file(row, filename):
  logger.debug("Writing sample '%s' to '%s'" % (row[0], filename))
  output_file = output_files[filename]
  output_row = keep_columns(row, columns_per_file[filename])
  output_file.write(row_to_text(output_row))

def write_header_rows(header):
  for output_filename in output_matrices:
    write_row_to_file(header, output_filename)

def write_test_sample_to_all_files(row):
  for output_filename in output_files:
    write_row_to_file(row, output_filename)

logger.debug("Starting to write to files")
write_header_rows(header)
for row in csv_file:
  sample_name = row[0]
  if sample_name in test_sample_names:
    write_test_sample_to_all_files(row)
  else:
    try:
      output_filename = file_for_training_sample[sample_name]
      write_row_to_file(row, output_filename)
    except KeyError:
      pass

logger.debug("Closing all of the files")
for output_file in output_files.values():
  output_file.close()
