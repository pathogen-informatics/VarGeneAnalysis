#!/usr/bin/env python

import argparse
import Bio.SearchIO
import datetime
import json
import logging
import re
import sys
import unittest

from StringIO import StringIO

def write_row(f, row):
  row_string = "\t".join(map(str, row))
  f.write(row_string + "\n")

def domain_from_subdomain(subdomain):
  # gets domain from something like cluster.5.DBLa
  (domain,) = re.findall("^cluster\.\d+\.([^_]+)$", subdomain)
  return domain

def get_best_hits_for_query(query, threshold=None):
  best_for_domain = {}
  sample_name = query.id
  by_bitscore = lambda hit: hit.bitscore
  for hit in query:
    subdomain = hit.id
    domain = domain_from_subdomain(subdomain)
    current_best = best_for_domain.get(domain)
    best_for_domain[domain] = max(hit, current_best, key=by_bitscore) if current_best != None else hit
  domains = []
  for domain,hit in best_for_domain.items():
    if threshold == None or hit.bitscore > threshold:
      domains.append({
        'domain': domain,
        'sub_domain': hit.id,
        'score': hit.bitscore,
        'evalue': hit.evalue
      })
  return {
    'sample_name': sample_name,
    'domains': domains
  }

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
    data.append(get_best_hits_for_query(query, args.threshold))

  json.dump({
    'data': data,
    'created_at': str(datetime.datetime.now())
  }, args.output_json)

class TestBestHits(unittest.TestCase):
  logging.basicConfig(level=logging.DEBUG)

  def test_best_hits(self):
    input_file = StringIO("""\
cluster.11.CIDRa_k12 -          DD2var30             -           2.3e-107  349.1  30.6  2.3e-107  349.1  30.6   7.7   8   2   1   9   9   1   2 -
cluster.8.CIDRa_k12  -          DD2var30             -           5.2e-107  348.1  71.2  8.6e-106  344.1  30.8   8.0   8   1   0   9   9   2   2 -
cluster.5.CIDRa_k12  -          DD2var30             -            5.9e-98  317.7  64.4   1.3e-82  267.5  28.9   8.3   8   1   0   9   9   2   2 -
cluster.0.CIDRa_k12  -          DD2var30             -            1.4e-89  290.7  68.7   1.9e-86  280.4  33.0   7.7   8   0   0   8   8   2   2 -
cluster.7.CIDRa_k12  -          DD2var30             -            2.7e-70  227.5  65.8   5.7e-61  196.9  30.5   8.2   8   1   1   9   9   2   3 -
cluster.3.CIDRa_k12  -          DD2var30             -            5.1e-69  223.0  62.4   1.8e-53  172.0  27.6   8.3   8   0   0   8   8   2   3 -
cluster.6.CIDRa_k12  -          DD2var30             -            2.9e-68  220.4  67.1     7e-53  170.0  34.1   8.5   8   1   0   8   8   2   3 -
cluster.4.CIDRa_k12  -          DD2var30             -            9.7e-60  192.5  64.6   9.6e-46  146.7  29.3   7.0   8   0   0   8   8   2   3 -
cluster.2.CIDRa_k12  -          DD2var30             -              8e-57  183.0  63.1   7.6e-48  153.6  28.6   7.7   8   0   0   8   8   2   4 -
cluster.9.CIDRa_k12  -          DD2var30             -            2.7e-52  168.6  71.3   1.6e-51  166.0  32.8   8.8   7   3   2   9   9   2   2 -
cluster.10.CIDRa_k12 -          DD2var30             -            9.8e-51  162.8  68.4   2.2e-39  125.6  33.1   8.6   8   2   0   8   8   2   3 -
cluster.1.CIDRa_k12  -          DD2var30             -            9.9e-31   97.3  60.8   8.3e-21   64.8  19.2   8.0   9   0   0   9   9   3   4 -
""")
    query = Bio.SearchIO.parse(input_file, 'hmmer3-tab').next()
    expected = {
      'sample_name': 'DD2var30',
      'domains': [
        {
          'domain': 'CIDRa',
          'sub_domain': 'cluster.11.CIDRa_k12',
          'score': 349.1,
          'evalue': 2.3e-107
        }
      ]
    }
    actual = get_best_hits_for_query(query)
    self.assertEqual(actual, expected)

    actual = get_best_hits_for_query(query, 300)
    self.assertEqual(actual, expected)

    expected = {
      'sample_name': 'DD2var30',
      'domains': []
    }
    actual = get_best_hits_for_query(query, 1000)
    self.assertEqual(actual, expected)

  def test_best_hits_for_multiple_domains(self):
    input_file = StringIO("""\
cluster.0.DBLa_k6    -          DD2var30             -           5.1e-251  823.5 112.0  5.1e-189  619.2  34.2   8.4   6   3   0   6   6   3   3 -
cluster.1.DBLa_k6    -          DD2var30             -           9.8e-236  772.7 156.9    2e-165  541.2  30.9   8.6   8   1   0   8   8   3   4 -
cluster.3.DBLa_k6    -          DD2var30             -           2.9e-235  771.6 107.2  1.2e-180  591.6  35.5   9.2   7   2   1   8   8   4   4 -
cluster.5.DBLa_k6    -          DD2var30             -           7.8e-233  763.3 106.6  5.4e-169  553.0  32.2   8.0   6   3   0   6   6   3   3 -
cluster.4.DBLa_k6    -          DD2var30             -           2.6e-223  732.1 162.1  4.5e-163  533.6  33.9   8.2   6   2   0   6   6   3   3 -
cluster.2.DBLa_k6    -          DD2var30             -             6e-204  667.7 162.7  2.2e-125  409.0  30.1   8.8   7   2   1   8   8   3   4 -
cluster.11.CIDRa_k12 -          DD2var30             -           2.3e-107  349.1  30.6  2.3e-107  349.1  30.6   7.7   8   2   1   9   9   1   2 -
cluster.8.CIDRa_k12  -          DD2var30             -           5.2e-107  348.1  71.2  8.6e-106  344.1  30.8   8.0   8   1   0   9   9   2   2 -
cluster.5.CIDRa_k12  -          DD2var30             -            5.9e-98  317.7  64.4   1.3e-82  267.5  28.9   8.3   8   1   0   9   9   2   2 -
cluster.0.CIDRa_k12  -          DD2var30             -            1.4e-89  290.7  68.7   1.9e-86  280.4  33.0   7.7   8   0   0   8   8   2   2 -
cluster.7.CIDRa_k12  -          DD2var30             -            2.7e-70  227.5  65.8   5.7e-61  196.9  30.5   8.2   8   1   1   9   9   2   3 -
cluster.3.CIDRa_k12  -          DD2var30             -            5.1e-69  223.0  62.4   1.8e-53  172.0  27.6   8.3   8   0   0   8   8   2   3 -
cluster.6.CIDRa_k12  -          DD2var30             -            2.9e-68  220.4  67.1     7e-53  170.0  34.1   8.5   8   1   0   8   8   2   3 -
cluster.4.CIDRa_k12  -          DD2var30             -            9.7e-60  192.5  64.6   9.6e-46  146.7  29.3   7.0   8   0   0   8   8   2   3 -
cluster.2.CIDRa_k12  -          DD2var30             -              8e-57  183.0  63.1   7.6e-48  153.6  28.6   7.7   8   0   0   8   8   2   4 -
cluster.9.CIDRa_k12  -          DD2var30             -            2.7e-52  168.6  71.3   1.6e-51  166.0  32.8   8.8   7   3   2   9   9   2   2 -
cluster.10.CIDRa_k12 -          DD2var30             -            9.8e-51  162.8  68.4   2.2e-39  125.6  33.1   8.6   8   2   0   8   8   2   3 -
cluster.1.CIDRa_k12  -          DD2var30             -            9.9e-31   97.3  60.8   8.3e-21   64.8  19.2   8.0   9   0   0   9   9   3   4 -
""")
    #self.maxDiff = None
    query = Bio.SearchIO.parse(input_file, 'hmmer3-tab').next()
    expected = {
      'sample_name': 'DD2var30',
      'domains': [
        {
          'domain': 'CIDRa',
          'sub_domain': 'cluster.11.CIDRa_k12',
          'score': 349.1,
          'evalue': 2.3e-107
        },
        {
          'domain': 'DBLa',
          'sub_domain': 'cluster.0.DBLa_k6',
          'score': 823.5,
          'evalue': 5.1e-251
        }
      ]
    }
    actual = get_best_hits_for_query(query)
    self.assertEqual(actual, expected)

    expected = {
      'sample_name': 'DD2var30',
      'domains': [
        {
          'domain': 'DBLa',
          'sub_domain': 'cluster.0.DBLa_k6',
          'score': 823.5,
          'evalue': 5.1e-251
        }
      ]
    }
    actual = get_best_hits_for_query(query, 700)
    self.assertEqual(actual, expected)

    expected = {
      'sample_name': 'DD2var30',
      'domains': []
    }
    actual = get_best_hits_for_query(query, 1000)
    self.assertEqual(actual, expected)
