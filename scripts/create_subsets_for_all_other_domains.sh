#!/bin/bash

set -eux

for domain in CIDRb CIDRg DBLb DBLd DBLe DBLg DBLz; do
#for domain in DBLa; do
  matrix="/nfs/pathogen003/tdo/Pfalciparum/VAR/Assembly.Version2/Normalized.60/Analysis/Domain_Aug_Reblast_may2014/Matrix.${domain}.txt"
  ./list_samples_from_distance_matrix.py $matrix | ./domain_frequency_stats_from_sample_names.py - > ${domain}.stats
  ./create_subsets_for_other_domains ${domain}.stats CIDRa.subsets ${domain} > ${domain}.subsets
done
