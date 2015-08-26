#!/usr/bin/env python

import pandas as pd

subsets_a = pd.read_csv("subset_6_012.CIDRa_k12_DBLa_k6.table.log", delimiter="\t")
subsets_b = pd.read_csv("subset_6_345.CIDRa_k12_DBLa_k6.table.log", delimiter="\t")

def split_sample_name(sample_name):
  try:
    isolate, gene, domain, _ = sample_name.split('.', 3)
    return "%s.%s" % (isolate, gene), domain
  except ValueError:
    return sample_name, "unknown"

def get_subset_domains(subsets):
  new_columns_data = zip(*subsets.name.map(split_sample_name))
  subsets['gene'], subsets['domain'] = new_columns_data
  
  subsets.loc[subsets['domain'] == 'CIDRa', 'DBLa_subdomain'] = np.nan
  subsets.loc[subsets['domain'] == 'DBLa', 'CIDRa_subdomain'] = np.nan
  
  counts = subsets.groupby(['gene']).apply(lambda df: pd.Series({'CIDRa_count':
                                                                    len(df.CIDRa_subdomain.dropna()),
                                                                    'DBLa_count':
                                                                    len(df.DBLa_subdomain.dropna())}))
  genes_with_multiple_CIDRa = counts[counts.CIDRa_count > 1].index.values
  genes_with_multiple_DBLa = counts[counts.DBLa_count > 1].index.values
  
  subsets = subsets[subsets['gene'].isin(genes_with_multiple_DBLa) == False]
  subsets = subsets[subsets['gene'].isin(genes_with_multiple_CIDRa) == False]
  
  subsets_list = pd.melt(subsets, id_vars=['name', 'gene', 'domain'])
  subsets_list = subsets_list[pd.notnull(subsets_list['value'])]
  
  subsets_domains = subsets_list[['gene', 'variable', 'value']].set_index(['gene', 'variable']).unstack('variable')['value']
  return subsets_domains


subsets_a_domains = get_subset_domains(subsets_a)
subsets_a_domains['source'] = 'A'
subsets_a_domains.set_index('source', append=True, inplace=True)

subsets_b_domains = get_subset_domains(subsets_b)
subsets_b_domains['source'] = 'B'
subsets_b_domains.set_index('source', append=True, inplace=True)

subset_domains = pd.concat([subsets_a_domains, subsets_b_domains]).unstack()

metrics.adjusted_rand_score(*(zip(*subset_domains.stack(0).values)))

# 93.9%
