#!/bin/bash

set -eu

test_set=${1-"subset_2"}
training_set_a=${2-"subset_0"}
training_set_b=${3-"subset_1"}
full_matrix_folder=${4-"/nfs/pathogen003/tdo/Pfalciparum/VAR/Assembly.Version2/Normalized.60/Analysis/Domain_Aug_Reblast_may2014"}
output_folder=${5-$(pwd)}

for domain in CIDRa DBLa; do
  full_matrix="${full_matrix_folder}/Matrix.${domain}.txt"
  ./extract_distance_matrix_for_subset.py ../isolates_per_subset/${domain}.subsets $full_matrix $test_set $training_set_a > ${output_folder}/matrix.${domain}.${test_set}_${training_set_a}.txt
  ./extract_distance_matrix_for_subset.py ../isolates_per_subset/${domain}.subsets $full_matrix $test_set $training_set_b > ${output_folder}/matrix.${domain}.${test_set}_${training_set_b}.txt
done
