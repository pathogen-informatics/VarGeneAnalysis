#!/usr/bin/env python

import argparse
import Bio.SeqIO
import csv
import logging
import os
import sys
import unittest

from StringIO import StringIO

logging.basicConfig(level=logging.DEBUG)

def parse_sequence_id(seq_id):
  isolate = seq_id.strip().split('.')[0]
  return isolate

def parse_subsets(input_file):
  subsets = {}
  for row in csv.reader(input_file, delimiter="\t"):
    subset, isolate_count, sample_count, cluster_string = row
    isolates = cluster_string.strip().split(',')
    subsets[subset] = isolates
  return subsets

def build_sample_file_map(output_directory, subsets, filename_template="{subset}.fa"):
  isolate_file_map = {}
  all_files = []
  for subset,isolates in subsets.items():
    subset_filename = os.path.join(output_directory, filename_template.format(subset=subset))
    subset_file = open(subset_filename, 'a')
    all_files.append(subset_file)
    for isolate in isolates:
      isolate_file_map[isolate] = subset_file
  return isolate_file_map, all_files

def write_sequences(input_fasta, sample_output_fasta_map):
  sequences = Bio.SeqIO.parse(input_fasta, 'fasta')
  for sequence in sequences:
    isolate = parse_sequence_id(sequence.id)
    output_file = sample_output_fasta_map.get(isolate)
    if output_file != None:
      output_file.write(sequence.format('fasta'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('subset_file', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('-o', '--output_dir', type=str, default=os.getcwd())
  parser.add_argument('input_fasta', type=argparse.FileType('r'))
  parser.add_argument('subsets', type=str, nargs="*")

  args = parser.parse_args()

  _subsets = parse_subsets(args.subset_file)
  if args.subsets != []:
    subsets = {}
    try:
      for subset in args.subsets:
        subsets[subset] = _subsets[subset]
    except ValueError:
      raise ValueError("Problem finding %s in %s" % (subset, args.subset_file.name))
  else:
    subsets = _subsets

  isolate_file_map, all_files = build_sample_file_map(args.output_dir, subsets)
  write_sequences(args.input_fasta, isolate_file_map)

  for output_file in all_files:
    output_file.close()
