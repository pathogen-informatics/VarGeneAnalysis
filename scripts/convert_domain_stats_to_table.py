#!/usr/bin/env python

import argparse
import datetime
import json
import logging
import re
import sys

def write_row(f, row):
  row_string = "\t".join(map(str, row))
  f.write(row_string + "\n")

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument("input_json", type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument("output_table", type=argparse.FileType('wb'), default=sys.stdout)

  args = parser.parse_args()

  data = json.load(args.input_json)['data']
  all_domains = set()
  rows = {}

  for sample in data:
    sample_name = sample['sample_name']
    domains = sample['domains']

    all_domains.update([domain['domain'] for domain in domains])
    rows[sample_name] = {domain['domain']: domain['sub_domain'] for domain in domains}

  write_row(args.output_table, ['name'] + list(all_domains))
  for sample_name, sample_data in rows.items():
    row = [sample_name] + [sample_data.get(domain, '') for domain in all_domains]
    write_row(args.output_table, row)
