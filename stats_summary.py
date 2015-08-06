#!/usr/bin/env python

import argparse
import csv
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('stats_file', type=argparse.FileType('r'), default=sys.stdin)

args = parser.parse_args()

stats = json.load(args.stats_file)

for domain,domain_stats in stats['samples'].items():
  for isolate,isolate_data in domain_stats.items():
    count = isolate_data['count']
    print "\t".join(map(str, [count, domain, isolate]))
