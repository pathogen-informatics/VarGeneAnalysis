# General

## `install.sh`

Installs the things you need to get scikit-learn working

# Create subsets of data

Scripts used to create the test data subsets

## `list_samples_from_distance_matrix.py`

Returns a list of samples listed in the header row of a many to many distance matrix

```
id	sample_1.g1.DBLa.100	sample_1.g2.DBLa.200
sample_1.g1.DBLa.100	1	0.7
sample_1.g2.DBLa.200	0.7	1
```

turns into the input for `domain_frequency_stats_from_sample_names.py`

## `domain_frequency_stats_from_sample_names.py`

From a list of sample_names like this:

```
sample_1.g1.DBLa.100
sample_1.g2.DBLa.200
```

Create some json like so:

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

## `domain_frequency_stats_to_table.py`

Converts the json based statistics from `domain_frequency_stats_from_sample_names.py` into a table like:

```
2	DBLa	sample_1
```

## `create_subsets_of_isolate_names.py`

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

## `create_subsets_for_other_domains.py`

Takes the output from `create_subsets_of_isolate_names.py` and the names of
isolates in each subset of the data and outputs a similar table for another
domain.  This is useful to check that splitting isolates according to the
frequency of examples of one domain doesn't result in horribly skewed
frequencies for the other domains.  In this case it didn't.

## `create_subsets_for_all_other_domains.sh`

Runs `create_subsets_for_other_domains.py` for a group of domains.

## `compare_domain_counts_across_subsets.py`

Creates a table from files created by `create_subsets_for_other_domains.py`
with the number of samples of each domain in each subset of the data.

## `extract_distance_matrix_for_subset.py`

Creates a distance matrix file for one or more subsets of the data.

Uses a subset file created by `create_subsets_for_other_domains.py`, a matrix
file and a list of subsets and pulls out the relevant bits.

## `extract_distance_matrix_for_tests_set.sh`

Wrapper script around `extract_distance_matrix_for_subset.py` to create the
training and test sets we need.

# Model training and selection

## `train_clusters_with_kmeans.py`

For a number of values of k, repeatedly train clusters and output consistency
scores from cross validation.

Scores are output in a table with three different types of row:

```
<k>	internal	<consistency_score_for_clusters_with_similar_data>	<clustering_attempt_index>	<other_clustering_attempt_index> <number_of_iterations>	<number_of_initial_conditions_tried>
<k>	external	<consistency_score_for_clusters_with_different_data>	<clustering_attempt_index>	<other_clustering_attempt_index> <number_of_iterations>	<number_of_initial_conditions_tried>
<k>	best <consistency_score_for_clusters_with_different_data>	<clustering_attempt_index>	<other_clustering_attempt_index> <number_of_iterations>	<number_of_initial_conditions_tried> <list_of_samples_in_each_cluster>
```

## `build_fastas_for_cluster.py`

Used to build a fasta file for each cluster identified by `train_clusters_with_kmeans.py`

Takes a list of samples separated by commas and semi-colons.  Semi-colons
distinguish the boundaries of a cluster, commas separe the names of samples.
This input is the same as the last column of the 'best' rows from `train_clusters_with_kmeans.py`

## `make_hmmer_models.sh`

Takes a list of fasta files as arguments.  For each it aligns the contents of the
fasta with `clustalw` and then builds a hmmer profile with `hmmbuild`

## `parse_hmmer_tblout.py`

Parses the output from `hmmscan` into json for all of the best hits above an optional threshold

`hmmer` is run by first `cat`-ing profiles together, then `hmmpress`, then running the `hmmscan`

For example:
```
cat cluster_files/*.hmm > cluster_files/hmm.all.CIDRa_k12_DBLa_k6.hmm
hmmpress cluster_files/hmm.all.CIDRa_k12_DBLa_k6.hmm
hmmscan --cpu 4 -E 1e-6 --domE 1e-6 --tblout res.SevenGenomes.CIDRa_k12_DBLa_k6.tblout.txt \
        --domtblout res.SevenGenomes.CIDRa_k12_DBLa_k6.domtblout.txt \
        -o res.SevenGenomes.CIDRa_k12_DBLa_k6.output \
        cluster_files/hmm.all.CIDRa_k12_DBLa_k6.hmm ../SevenGenomes.aa.fasta
./parse_hmmer_tblout.py res.SevenGenomes.CIDRa_k12_DBLa_k6.tblout.txt - | python -mjson.tool | less -S
```

## `parse_seven_genomes_subdomain_info.py`

Script which parses the output from the [VarDom service](http://www.cbs.dtu.dk/services/VarDom/)
to output details of which subdomains are in which of the genes in the original Seven Genomes.

## `convert_domain_stats_to_table.py`

Takes json from `parse_seven_genomes_subdomain_info.py` or `parse_hmmer_tblout.py`
and outputs a table.  Each line is a sample and the subdomain for CIDRa
and DBLa.

## `compare_domain_stats_tables.py`

Takes a couple of tables from `convert_domain_stats_to_table.py` and creates a
matrix for each column in the input table.

Each table has a column for each cluster label in one input and a row for each
label in the other input.  The contents of the matrix show the number of times
a sample classified as cluster_a in one table is classified as cluster_b in
the other.

This is useful to compare two different clustering algorithms.
