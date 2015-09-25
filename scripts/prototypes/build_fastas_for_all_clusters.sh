#!/bin/bash

hmmer_folder=${1-"../../results/subsets_3,5_v_4,5/hmmer_models"}
for domain_k in $(ls ${hmmer_folder}); do 
  IFS='_'; set $domain_k; domain=$1; k=$2; unset IFS; 
  ./build_fastas_for_cluster.py \
    -o ${hmmer_folder}/${domain}_${k}/fastas \
    $domain \
    ${hmmer_folder}/${domain}_${k}/cluster_members.log \
    ../../amino_acid_sequences/Domain.${domain}.fasta; 
done
