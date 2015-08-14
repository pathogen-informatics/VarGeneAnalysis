#!/usr/bin/env python

import argparse
import Bio.SearchIO
import datetime
import json
import logging
import re
import sys

def write_row(f, row):
  row_string = "\t".join(map(str, row))
  f.write(row_string + "\n")

def domain_from_subdomain(subdomain):
  (domain,) = re.findall("^cluster\.\d+\.([^_]+)_k\d+$", subdomain)
  return domain

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--threshold", type=float, default=1)
  parser.add_argument("input_hmm", type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument("output_json", type=argparse.FileType('wb'), default=sys.stdout)

  args = parser.parse_args()

  hmmer_queries = Bio.SearchIO.parse(args.input_hmm, 'hmmer3-tab')
  data = []
  for query in hmmer_queries:
    best_for_domain = {}
    sample_name = query.id
    for hit in query:
      subdomain = hit.id
      domain = domain_from_subdomain(subdomain)
      current_best = best_for_domain.get(domain)
      best_for_domain[domain] = min(hit, current_best) if current_best != None else hit
    domains = []
    for domain,hit in best_for_domain.items():
      if hit.evalue < args.threshold:
        domains.append({
          'domain': domain,
          'sub_domain': hit.id,
          'score': hit.evalue
        })
    data.append({
      'sample_name': sample_name,
      'domains': domains
    })

  json.dump({
    'data': data,
    'created_at': str(datetime.datetime.now())
  }, args.output_json)
