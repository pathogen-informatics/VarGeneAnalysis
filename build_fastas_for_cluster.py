#!/usr/bin/env python

import argparse
import Bio.SeqIO
import logging
import os
import sys
import unittest

from StringIO import StringIO

logging.basicConfig(level=logging.DEBUG)

def parse_clusters(input_file):
  clusters = []
  def remove_empty_string(clusters):
    return [cluster for cluster in clusters if cluster != '']
  def remove_empty_clusters(el):
    return el != []
  for line in input_file:
    new_cluster_strings = line.strip().split(';')
    new_clusters = [[sample.strip() for sample in cluster_string.strip().split(',')] for cluster_string in new_cluster_strings]
    clusters += filter(remove_empty_clusters, map(remove_empty_string, new_clusters))
  return clusters

def build_sample_file_map(domain, output_directory, clusters, filename_template="cluster.{i}.{domain}.fa"):
  sample_file_map = {}
  for cluster_number,cluster in enumerate(clusters):
    cluster_filename = os.path.join(output_directory, filename_template.format(i=cluster_number, domain=domain))
    cluster_file = open(cluster_filename, 'w')
    for sample in cluster:
      sample_file_map[sample] = cluster_file
  return sample_file_map

def write_sequences(input_fastas, sample_output_fasta_map):
  sequences = (seq for input_fasta in input_fastas for seq in Bio.SeqIO.parse(input_fasta, 'fasta'))
  for sequence in sequences:
    output_file = sample_output_fasta_map.get(sequence.id)
    if output_file != None:
      output_file.write(sequence.format('fasta'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('domain', type=str)
  parser.add_argument('cluster_file', type=argparse.FileType('r'), default=sys.stdin)
  parser.add_argument('-o', '--output_dir', type=str, default=os.getcwd())
  parser.add_argument('input_fastas', type=argparse.FileType('r'), nargs="*")

  args = parser.parse_args()

  clusters = parse_clusters(args.cluster_file)
  sample_file_map = build_sample_file_map(args.domain, args.output_dir, clusters)
  write_sequences(args.input_fastas, sample_file_map)

  for output_file in sample_file_map.values():
    output_file.close()

class FakeFile(StringIO):
 def __init__(self, name):
    self.name = name
    StringIO.__init__(self)
        
class test_main(unittest.TestCase):
  def test_parse_clusters(self):
    input_file = StringIO("""1,2,3;4,5
6;7,8;
9,""")
    expected = [['1', '2', '3'], ['4', '5'], ['6'], ['7', '8'], ['9']]
    self.assertEqual(parse_clusters(input_file), expected)

  def test_write_sequences(self):
    input_files = [
      StringIO("""\
>foo1
ABC
>foo2
DEF
>foo3
GHI
"""),
      StringIO("""\
>bar1
JKL
>bar2
MNO
""")]
    cluster_1 = FakeFile('cluster_1')
    cluster_2 = FakeFile('cluster_2')
    sample_cluster_map = {
      'foo1': cluster_1,
      'bar1': cluster_1,
      'foo3': cluster_2,
      'baz4': cluster_2
    }
    expected_1 = """\
>foo1
ABC
>bar1
JKL
"""
    expected_2 = """\
>foo3
GHI
"""
    write_sequences(input_files, sample_cluster_map)
    cluster_1.seek(0)
    cluster_2.seek(0)
    self.assertEqual(cluster_1.read(), expected_1)
    self.assertEqual(cluster_2.read(), expected_2)
