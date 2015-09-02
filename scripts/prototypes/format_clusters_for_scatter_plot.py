#!/usr/bin/env python

import argparse
import pandas as pd
import sys

def extract_domain_classification(data, domain):
  rows = map(lambda i: domain in i, data['name'])
  domain_data = data[rows][['name', '%s_subdomain'%domain]]
  domain_data.columns = ['name', 'cluster']
  domain_data['cluster'] = map(lambda c: c.split('.')[1], domain_data['cluster'])
  colours = "blue,brown,cadetblue,chartreuse4,chocolate4,cyan,darkblue,darkgrey,darkgreen,darkmagenta,darkorange,darkolivegreen1"
  colours = colours.split(',')
  domain_data['colour'] = map(lambda c: colours.__getitem__(int(c)), domain_data['cluster'])
  return domain_data


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("input_table", type=argparse.FileType('r'))
  parser.add_argument("domain", type=str)
  parser.add_argument("output_file", type=argparse.FileType('w'),
                      default=sys.stdout)
  args = parser.parse_args()

  data = pd.read_csv(args.input_table, delimiter='\t')
  domain_data = extract_domain_classification(data, args.domain)
  print domain_data.to_csv(sep='\t', index=False)
