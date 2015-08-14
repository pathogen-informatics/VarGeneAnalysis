#!/bin/bash

test_set=subset_2
training_set_a=subset_0
training_set_b=subset_1

for domain in CIDRa DBLa; do
  full_matrix="/nfs/pathogen003/tdo/Pfalciparum/VAR/Assembly.Version2/Normalized.60/Analysis/Domain_Aug_Reblast_may2014/Matrix.${domain}.txt"
  ./subsample_matrix.py subsets/${domain}.subsets $full_matrix $test_set $training_set_a > ../input_data_subsets/matrix.${domain}.${test_set}_${training_set_a}.txt
  ./subsample_matrix.py subsets/${domain}.subsets $full_matrix $test_set $training_set_b > ../input_data_subsets/matrix.${domain}.${test_set}_${training_set_b}.txt
done
