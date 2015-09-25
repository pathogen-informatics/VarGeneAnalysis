#!/bin/bash

# Iterate over all domains and compare their kmeans graphs

RESULTS_A=${1-"../../results/subsets_0,2_v_1,2/kmeans_results"}
RESULTS_B=${2-"../../results/subsets_3,5_v_4,5/kmeans_results"}
TITLE_SUFFIX_A=${3-"012"}
TITLE_SUFFIX_A=${4-"345"}

for DOMAIN in $(ls $RESULTS_A); do 
  ./prototypes/graph_kmeans.py -t ${DOMAIN}_${TITLE_SUFFIX_A} ${RESULTS_A}/${DOMAIN}/* &
  ./prototypes/graph_kmeans.py -t ${DOMAIN}_${TITLE_SUFFIX_A} ${RESULTS_B}/${DOMAIN}/*
done
