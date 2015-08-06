#!/usr/bin/env python

import argparse
import csv
import json
import random
import sys

def get_isolate_counts(stats):
  domain = stats['samples'][args.domain]
  
  isolate_counts = {}
  for isolate,isolate_data in domain.items():
    count = isolate_data['count']
    isolate_counts[isolate] = count
  return isolate_counts

def check_splits(splits, target_size):
  split_sizes = [sum((count for isolate,count in split)) for split in splits]
  return max(split_sizes) <= target_size

def split_data(isolate_counts, n_splits):
  def by_value(t):
    return t[1]

  data = sorted(isolate_counts.items(), key=by_value)
  return [[data[i] for i in xrange(n,len(data),n_splits)] for n in xrange(n_splits)]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('stats_file', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('target_size', type=int, default=1000)
  parser.add_argument('domain', type=str)
  
  args = parser.parse_args()
  
  stats = json.load(args.stats_file)
  isolate_counts = get_isolate_counts(stats)

  for n_splits in range(1,len(isolate_counts)+1):
    splits = split_data(isolate_counts, n_splits)
    if check_splits(splits, args.target_size):
      break
  
  random.shuffle(splits)

  for n,split in enumerate(splits):
    name = "subset_%s" % n
    n_domains = sum((count for isolate,count in split))
    isolates = [isolate for isolate,count in split]
    isolates_string = ','.join(map(str, isolates))
    n_isolates = len(isolates)
    print "\t".join(map(str, [name, n_isolates, n_domains, isolates_string]))

import unittest

class TestSplit(unittest.TestCase):
  def test_split(self):
    data = {
      'a': 1,
      'b': 2,
      'c': 3
    }
    expected = [[('a', 1), ('c', 3)], [('b', 2)]]
    self.assertEqual(split_data(data, 2), expected)

  def test_split_2(self):
    data = {
      'a': 1,
      'c': 2,
      'b': 3
    }
    expected = [[('a', 1), ('b', 3)], [('c', 2)]]
    self.assertEqual(split_data(data, 2), expected)

  def test_split_3(self):
    data = {
      'd': 4,
      'a': 1,
      'c': 2,
      'b': 3
    }
    expected = [[('a', 1), ('b', 3)], [('c', 2), ('d', 4)]]
    self.assertEqual(split_data(data, 2), expected)
