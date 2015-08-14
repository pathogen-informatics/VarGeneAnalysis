#!/usr/bin/env python

import argparse
import datetime
import json
import logging
import re
import sys

from bs4 import BeautifulSoup

def has_nested_table(table):
  return table.table != None

def get_domain_rows(table):
  rows = table.table.find_all('tr')
  return [row for row in rows if row.td.text.strip() == 'Domains']

def unnest_areas(row):
  # area tags weren't being closed properly
  for area in row.map.find_all('area')[-1::-1]:
    row.map.append(area)

def parse_domain_alt_text(alt_text):
  ((sub_domain, start, end),) = re.findall("(\S+)\s+-\s+start-pos:\s+(\d+)\s+-\s+end-pos:\s+(\d+)", alt_text)
  (domain,) = re.findall("^(\D+)", sub_domain)
  return {
    'domain': domain,
    'sub_domain': sub_domain,
    'start': start,
    'end': end
  }

def parse_domain_row(row):
  unnest_areas(row)
  alt_texts = [area['alt'] for area in row.find_all('area')]
  domains = [parse_domain_alt_text(alt_text) for alt_text in alt_texts]
  return domains

def parse_table(table):
  sample_name = table.tr.td.span.text
  links = table.tr.td.find_all('a')
  (protein_url,) = [link['href'] for link in links if link.text == 'Protein']
  domain_rows = get_domain_rows(table)
  domains = [domain for domain_row in domain_rows for domain in parse_domain_row(domain_row)]
  return {
    'sample_name': sample_name,
    'protein_url': protein_url,
    'domains': domains
  }

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)

  parser = argparse.ArgumentParser()
  parser.add_argument("input_html", type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument("output_json", type=argparse.FileType('wb'), default=sys.stdout)

  args = parser.parse_args()

  html_doc = args.input_html.read()
  # div tags aren't closed which leads to recursion issues, this is a hack
  html_doc = html_doc.replace('<div style="text-align: left;">', '</div><div style="text-align: left;">')

  soup = BeautifulSoup(html_doc, 'html.parser')
  tables = soup.find_all('table')
  tables_we_want = filter(has_nested_table, tables)
  domain_data = []
  number_of_tables = len(tables_we_want)
  parse_failures = 0
  logging.info("About to parse %s tables" % number_of_tables)
  for i,table in enumerate(tables_we_want):
    logging.debug("Parsing table %s of %s" % (i, number_of_tables))
    try:
      domain_data.append(parse_table(table))
    except AttributeError:
      logging.warn("Couldn't parse table %s of %s" % (i, number_of_tables))
      parse_failures += 1
      # Couldn't parse table
      pass

  logging.info("Failed to parse %s of %s tables" % (parse_failures,
                                                    number_of_tables))

  json.dump({
    'created_at': str(datetime.datetime.now()),
    'data': domain_data
  }, args.output_json)
