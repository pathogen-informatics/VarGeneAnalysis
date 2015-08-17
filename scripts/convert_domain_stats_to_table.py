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
  
  write_row(args.output_table, ['name', 'CIDRa_subdomain', 'DBLa_subdomain'])

  for sample in data:
    sample_name = sample['sample_name']
    all_domains = sample['domains']

    CIDRa_subdomains = [domain['sub_domain'] for domain in all_domains if domain['domain'] == "CIDRa"]
    if len(CIDRa_subdomains) == 0:
      CIDRa_subdomain = ''
    else:
      (CIDRa_subdomain,) = CIDRa_subdomains

    DBLa_subdomains = [domain['sub_domain'] for domain in all_domains if domain['domain'] == "DBLa"]
    if len(DBLa_subdomains) == 0:
      DBLa_subdomain = ''
    else:
      (DBLa_subdomain,) = DBLa_subdomains

    row = [sample_name, CIDRa_subdomain, DBLa_subdomain]
    write_row(args.output_table, row)
