#!/usr/bin/env python

import argparse
import csv
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('samples_list', type=argparse.FileType('r'), default=sys.stdin)

args = parser.parse_args()

stats = {'samples': {}}
skipped_samples = 0
parsed_samples = 0

def parse_sample(sample):
  try:
    isolate, gene, domain, position = sample.split('.')
  except ValueError, e:
    raise ValueError("Could not split %s" % sample)
  return isolate, gene, domain

for sample in args.samples_list:
  sample = sample.strip()
  try:
    isolate, gene, domain = parse_sample(sample)
  except ValueError:
    skipped_samples += 1
    continue # Skip this sample
  else:
    parsed_samples += 1
  stats['samples'].setdefault(domain, {}).setdefault(isolate, {'count': 0, 'samples': []})
  stats['samples'][domain][isolate]['count'] += 1
  stats['samples'][domain][isolate]['samples'].append(sample)

stats.update({
  'skipped_samples': skipped_samples,
  'parsed_samples': parsed_samples
})

print json.dumps(stats, sort_keys=True, indent=2, separators=(',', ': '))
