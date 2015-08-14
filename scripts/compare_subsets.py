#!/usr/bin/env python

import argparse
import csv
import logging
import sys

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('stats_files', type=argparse.FileType('r'), nargs="*", default=sys.stdin)
  
  args = parser.parse_args()
  stats = {}

  logging.basicConfig(level=logging.DEBUG)

  for stats_file in args.stats_files:
    domain = stats_file.name[:-8]
    logging.debug("Domain: %s" % domain)
    for row in csv.reader(stats_file, delimiter='\t'):
      split_name, n_domain_examples = row[0], row[2]
      stats.setdefault(split_name, {})[domain] = n_domain_examples

  def by_split_name(t):
    return t[0]

  def by_domain_name(t):
    return t[0]

  domains  = sorted(stats.values()[1].keys())

  header_row = ["subset"] + domains
  print "\t".join(map(str, header_row))
  for split_name,domain_stats in sorted(stats.items(), key=by_split_name):
    row  = [split_name] + [domain_stats.get(domain, 0) for domain in domains]
    print "\t".join(map(str, row))


