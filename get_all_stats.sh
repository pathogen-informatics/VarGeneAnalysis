#!/bin/bash

set -eux

for domain in CIDRb CIDRg DBLb DBLd DBLe DBLg DBLz; do
#for domain in DBLa; do
  matrix="/nfs/pathogen003/tdo/Pfalciparum/VAR/Assembly.Version2/Normalized.60/Analysis/Domain_Aug_Reblast_may2014/Matrix.${domain}.txt"
  ./list_samples.py $matrix | ./sample_stats.py - > ${domain}.stats
  ./split_other_samples.py ${domain}.stats CIDRa.subsets ${domain} > ${domain}.subsets
done
