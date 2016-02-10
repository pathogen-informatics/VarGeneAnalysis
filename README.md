# VarGene Analysis

This repo includes a series of scripts used to cluster VarGene domains into subdomains.

This work was somewhat successful but further work would be required to make these scripts more usable by others.

This code is published here for reference, you shouldn't run it without reading it to understand what it's doing.

## High level overview

The idea is that, for each VarGene domain, there are probably clusters of examples of domains which share something in common and that we could identify some subdomains.  By analysing a lot of examples of each domain, we can group examples into each subdomain and use them to train a HMMer model to categorise future samples.

### Basic steps

We start with N sequences for each domain and an N by N distance matrix showing the similarity between each example.

We then split the data into training and test sets whcih we will later use to evaluate the clustering algorithms.

Using different training sets we build models to claisfy the training set.  We repeat the process using a different training set and compare the consistency of the classification of the test set.

### Structure

This repo is made up of lots of simple scripts which each do one simple job, commonly on the output of the previous script.  In many cases data can be piped from one script into the next.

The contents of the `scripts` directory are pretty good, the contents of the `prototype` subdirectory are a lot hackier and may require manual tweaking to get around poor command line options.

## Instructions

If you are going to try and run this, here are some notes on how you might do it.

### Dependencies

Install dependencies:

```
sudo apt-get install libopenblas-base \
                     libopenblas-dev \
                     liblapack-dev \
                     libatlas-dev \
                     python-dev \
                     gfortran \
                     build-essential
pip install -r requirements.txt
```

## Create subsets of data

Scripts used to split the data into training and test sets.

### `list_samples_from_distance_matrix.py`

Takes a distance matrix and outputs the samples listed in the first header row as a list, one samples per line.

```
id	sample_1.g1.DBLa.100	sample_1.g2.DBLa.200
sample_1.g1.DBLa.100	1	0.7
sample_1.g2.DBLa.200	0.7	1
```

goes to:

```
sample_1.g1.DBLa.100
sample_1.g2.DBLa.200
```

This is used as the input for `domain_frequency_stats_from_sample_names.py`

## `domain_frequency_stats_from_sample_names.py`

This script parses the list of samples to create statistics on the number of examples of each domain for each sample.  It assumes that samples use the following format:

```
SAMPLE_NAME.GENE_NAME.DOMAIN.OTHER_IDENTIFIER
```

For example, from a list of sample_names like this:

```
sample_1.g1.DBLa.100
sample_1.g2.DBLa.200
```

It creates some json like so:

```
{
  'samples': {
    'DBLa': {
      'sample_1': {
        'count': 2,
        'samples': [
          'sample_1.g1.DBLa.100',
          'sample_1.g2.DBLa.200'
        ]
      }
    }
  }
}
```

Examples from our dataset can be found in `domain_frequency_stats/`

### `domain_frequency_stats_to_table.py`

Converts the json based statistics from `domain_frequency_stats_from_sample_names.py` into a table like:

```
2	DBLa	sample_1
```

This is just a more humanly readable output to check for wierdness.

### `create_subsets_of_isolate_names.py`

Used to subsample a dataset into subsets with approximately equal numbers of 
examples.  You give it json from `domain_frequency_stats_from_sample_names.py`,
a domain and the maximum number of gene examples per subset and it gives groups
of isolates which give you roughly the right number of examples.

Format is: `subset_name`, `number_of_isolates`, `number_of_gene_examples`, `names_of_isolates`

For example:

```
subset_0	4	100	MAL6P1,PH0344-C,PF0235-C,PT0002-CW
subset_1	5 99	PH0246-C,PD0123-C,PF0214-C,PH0259-C,PD0091-C
```

Play with the `target_size` parameter until you get your prefered number of subsets.  I picked 7 subsets: I used 3 as one training set, 3 as another and kept one as a test set.

### `create_subsets_for_other_domains.py`

Takes the output from `create_subsets_of_isolate_names.py` and the names of
isolates in each subset of the data (from `list_samples_from_distance_matrix.py`) and outputs a similar table for another
domain.  This is useful to check that splitting isolates according to the
frequency of examples of one domain doesn't result in horribly skewed
frequencies for the other domains.  In this case it didn't.

An example from out dataset can be found in `isolates_per_subset/`

### `create_subsets_for_all_other_domains.sh`

A convienience shell script which runs `create_subsets_for_other_domains.py` for a group of domains.

### `compare_domain_counts_across_subsets.py`

Creates a table from files created by `create_subsets_for_other_domains.py`
with the number of samples of each domain in each subset of the data.

Use to double check that the data segmentations isn't wierd across domains.

### `extract_distance_matrix_for_subset.py`

Creates a distance matrix file for one or more subsets of the data.

Uses a subset file created by `create_subsets_for_other_domains.py`, a matrix
file and a list of subsets and pulls out the relevant bits.

Pass it the names of multiple subsets to create one file with a combined matrix.

### `extract_distance_matrix_for_tests_set.sh`

Wrapper script around `extract_distance_matrix_for_subset.py` to create the
training and test sets we need.

## Model training and selection

Scripts used to train the kmeans models and pick which value of k is best.

We create models from a training set using different values of K and evaluate their consistency.  We then do the same with the other training set and use this to reasure ourselves that we've picked a good value for K (if that value is consistenly good regardless of the data used to evaluate it).

### `train_clusters_with_kmeans.py`

For a number of values of k, repeatedly train clusters and output consistency
scores from cross validation.

It does this by removing 20% of the training data as a temporary test set. It then picks 70% of the remaining data and, for a given value of k, clusters the samples.  It then extracts a different (but overlapping 70%) of the test_data and clusters it.  It then uses the results of clustering the two overlapping training sets to claisfy the temporary test set and evaluates how consistently it is clustered.

This is repeated a number of times; at each step outputing a row in the following format:

```
<k>	external	<consistency_score_for_clusters_with_different_data>	<clustering_attempt_index>	<other_clustering_attempt_index> <number_of_iterations>	<number_of_initial_conditions_tried>
```

Having does this a number of times (often 15); it picks one of the better correlation scores (top of the 80% percentile) and outputs it in a row as follows:

```
<k>	best <consistency_score_for_clusters_with_different_data>	<clustering_attempt_index>	<other_clustering_attempt_index> <number_of_iterations>	<number_of_initial_conditions_tried> <list_of_samples_in_each_cluster>
```

### `prototypes/graph_kmeans.py`

This is a hacky script which you can run locally to graph the results of `train_clusters_with_kmeans.py`.  For a given file, it finds all of the 'external' lines and produces a graph of mean and standard deviation of consistency scores for each value of K.

You can also provide it with a list of files and it will combine and sort them before producing the graph.

You then use these graphs to pick a value of K.  Exactly how to do this is outside the scope of this README but generally higher mean is better, lower standard deviation is better and a higher value of K is better.  I tended to rank by top 3 or 4 preferences for K as deduced from one training set; try and forget those values; pick 3 or 4 from another training set and then go with whichever was consistently highest between the two sets.  If I was feeling super scientific I might ask a couple of different people to do the same and compare our conclusions.

It is important to hold back some data while selecting K for a final validation set which will help you to detect if your model has overfitted to your data.

### `build_fastas_for_cluster.py`

Having picked a value for K, this is used to build a fasta file for each cluster.  You do this by finding the line in the output created by `train_clusters_with_kmeans.py` which corresponds to the 'best' clustering for the value or K.  At the end of this line is a list of samples separated by commas and semi-colons.  Semi-colons
distinguish the boundaries of a cluster, commas separe the names of samples.

### `make_hmmer_models.sh`

Takes a list of fasta files as arguments.  For each it aligns the contents of the
fasta with `clustalw` and then builds a hmmer profile with `hmmbuild`

### Run HMMer

`hmmer` is run by first `cat`-ing profiles together, then `hmmpress`, then running the `hmmscan`

For example:
```
cat cluster_files/*.hmm > cluster_files/hmm.all.CIDRa_k12_DBLa_k6.hmm
hmmpress cluster_files/hmm.all.CIDRa_k12_DBLa_k6.hmm
hmmscan --cpu 4 -E 1e-6 --domE 1e-6 --tblout res.SevenGenomes.CIDRa_k12_DBLa_k6.tblout.txt \
        --domtblout res.SevenGenomes.CIDRa_k12_DBLa_k6.domtblout.txt \
        -o res.SevenGenomes.CIDRa_k12_DBLa_k6.output \
        cluster_files/hmm.all.CIDRa_k12_DBLa_k6.hmm ../SevenGenomes.aa.fasta
```

### `parse_hmmer_tblout.py`

Parses the output from `hmmscan` into json for all of the best hits above an optional threshold

```
./parse_hmmer_tblout.py res.SevenGenomes.CIDRa_k12_DBLa_k6.tblout.txt - | python -mjson.tool | less -S
```

### `parse_seven_genomes_subdomain_info.py`

Script which parses the output from the [VarDom service](http://www.cbs.dtu.dk/services/VarDom/)
to output details of which subdomains are in which of the genes in the original Seven Genomes.

You can find the results of this script in `original_seven_genomes_domains.json`

### `convert_domain_stats_to_table.py`

Takes json from `parse_seven_genomes_subdomain_info.py` or `parse_hmmer_tblout.py`
and outputs a table.  Each line is a sample and the subdomain for CIDRa
and DBLa.

The original seven genomes subdomains can be found here:

```
original_seven_genomes_domains.just_ATS.table
original_seven_genomes_domains.just_NTS.table
original_seven_genomes_domains.table
```

### `compare_domain_stats_tables.py`

Takes a couple of tables from `convert_domain_stats_to_table.py` and creates a
matrix for each column in the input table.

Each table has a column for each cluster label in one input and a row for each
label in the other input.  The contents of the matrix show the number of times
a sample classified as cluster_a in one table is classified as cluster_b in
the other.

This is useful to compare two different clustering algorithms.

### `build_fastas_for_subset.py`

Build a fasta of amino acids for one or more subsets for use with `parse_hmmer_tblout.py`
