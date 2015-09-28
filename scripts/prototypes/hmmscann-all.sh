#!/bin/bash

hmmer_models="../../../results/subsets_*/hmmer_models"

for d in $(find $hmmer_models -mindepth 1 -maxdepth 1); do
  domain_k=$(echo $d | sed 's/.\+\///');
  IFS='_'; set $domain_k; domain=$1; k=$2; unset IFS;
  bsub.py --threads 4 1 j%J-hmmscan-${domain_k} hmmscan \
    --cpu 4 -E 1e-6 --domE 1e-6 \
    --tblout ${d}/hmmer.${domain_k}.subset_012345.tblout.txt \
    --domtblout ${d}/hmmer.${domain_k}.subset_012345.domtblout.txt \
    -o ${d}/hmmer.${domain_k}.subset_012345.output \
    ${d}/hmms/hmm.${domain_k}.hmm \
    ../../subset_sequences/${domain}/subset_012345.fa
done
