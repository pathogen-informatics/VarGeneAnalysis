#!/usr/bin/env python

import argparse
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument('matrix_file', type=argparse.FileType('r'), default=sys.stdin)

args = parser.parse_args()

header_row = csv.reader(args.matrix_file, delimiter='\t').next()

print "\n".join(header_row[1:])
