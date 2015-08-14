#!/bin/bash

set -eux

FASTA_FILENAME_TO_ALN_FILENAME='s/\(.\+\)\.fa$/\1.aln/'
FASTA_FILENAME_TO_HMM_FILENAME='s/cluster\.\([^.]\+\)\.\([^.]\+\)\.fa$/hmm.\1.\2.hmm/'

for fasta_name in "$@"; do 
  aln_name=$(echo $fasta_name | sed $FASTA_FILENAME_TO_ALN_FILENAME)
  hmm_name=$(echo $fasta_name | sed $FASTA_FILENAME_TO_HMM_FILENAME)
  echo "Aligning $fasta_name"
  clustalw $fasta_name | grep -v "^Sequence" | grep -v "^Group"
  echo "Building hmmer model from $aln_name"
  hmmbuild --cpu 1 $hmm_name $aln_name
done
