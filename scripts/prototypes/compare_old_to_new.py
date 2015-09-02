#!/usr/bin/env python

import argparse
import pandas as pd

from compare_lots_of_clusters import plot_heatmap, normalise_along_axis

def stack_with_source(data, source_name):
  data = data.set_index('gene').stack().reset_index()
  data['source'] = source_name
  data.columns = ['gene', 'domain', 'cluster', 'source']
  data.head()
  data.set_index(['gene', 'source', 'domain'])
  data.set_index(['gene', 'source', 'domain']).unstack()
  data.set_index(['gene', 'source', 'domain']).unstack().unstack()
  data.set_index(['gene', 'source', 'domain']).unstack(1).unstack()
  data = data.set_index(['gene', 'source', 'domain']).unstack(1).unstack()
  return data

def join(data_a, data_b):
  data_a = data_a.reset_index()
  both = pd.merge(data_a, data_b, left_on='gene', right_index=True).set_index('gene')
  both = both.stack().stack().unstack().reset_index()
  both.columns = ['gene', 'domain', 'new', 'original']
  matrices = both.groupby(['domain', 'new', 'original']).apply(len).unstack()
  return matrices
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("original_data_table", type=argparse.FileType('r'))
  parser.add_argument("new_data_table", type=argparse.FileType('r'))
  parser.add_argument("domains", nargs="+", type=str)
  args = parser.parse_args()

  original_data = pd.read_csv(args.original_data_table,
                              delimiter='\t')
  original_data.columns = ['gene'] + args.domains
  original_data = stack_with_source(original_data, 'original')
  
  new_data = pd.read_csv(args.new_data_table,
                         delimiter='\t')
  new_data.columns = ['gene'] + args.domains
  new_data = stack_with_source(new_data, 'new')
  
  matrices = join(new_data, original_data)
  for domain in args.domains:
    domain_matrix = matrices.loc[domain].dropna(how='all', axis=1)
    normalised_matrix = normalise_along_axis(domain_matrix, 0)
    plot_heatmap(normalised_matrix)
    print normalised_matrix.T.fillna(0).to_csv(sep='\t')

  try:
    input("Press ENTER to close the windows")
  except:
    pass
