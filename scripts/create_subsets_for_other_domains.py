#!/usr/bin/env python

import argparse
import csv
import json
import logging
import random
import sys

from create_subsets_of_isolate_names import get_isolate_counts

class Split(object):
  @classmethod
  def from_line(cls, line):
    split_name, n_isolates, n_domain_examples, isolates_string = line.strip().split('\t')
    isolates = isolates_string.split(',')
    return Split(split_name, n_domain_examples, isolates)

  @classmethod
  def from_file(cls, splits_file):
    splits = []
    for line in splits_file:
      splits.append(cls.from_line(line))
    return splits

  def __init__(self, name, count=0, isolates=None):
    self.name = name
    self.isolates = isolates or []
    self.count = count

  def __repr__(self):
    return self.name

def create_split_map_dict(splits):
  split_map_dict = {}
  for split in splits:
    for isolate in split.isolates:
      split_map_dict[isolate] = split.name
  return split_map_dict

def create_split_mapper(splits, split_map_dict):
  def map_to_split(isolate_name, count):
    split_name = split_map_dict.get(isolate_name, 'unknown')
    split = splits.setdefault(split_name, Split(split_name))
    split.isolates.append(isolate_name)
    split.count += count
  return map_to_split

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('stats_file', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('splits_file', type=argparse.FileType('r'))
  parser.add_argument('domain', type=str)
  
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARN)
  
  stats = json.load(args.stats_file)
  isolate_counts = get_isolate_counts(stats, args.domain)

  existing_splits = Split.from_file(args.splits_file)
  new_splits = {}
  split_map = create_split_map_dict(existing_splits)
  split_mapper_function = create_split_mapper(new_splits, split_map)

  for isolate,count in isolate_counts.items():
    split_mapper_function(isolate, count)
    
  for split in sorted(new_splits.values(), key=lambda split: split.name):
    isolates_string = ','.join(split.isolates) 
    isolates_count = len(split.isolates)
    print "\t".join(map(str,[split.name, isolates_count, split.count, isolates_string]))
