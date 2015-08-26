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
  original_data = pd.read_csv("original_seven_genomes_domains.table.log", delimiter='\t')
  original_data.columns = ['gene', 'CIDRa', 'DBLa']
  original_data = stack_with_source(original_data, 'original')
  
  new_data = pd.read_csv("seven_genomes_data.CIDRa_k12_DBLa_k6.table.log",
                         delimiter='\t')
  new_data.columns = ['gene', 'CIDRa', 'DBLa']
  new_data = stack_with_source(new_data, 'new')
  
  matrices = join(new_data, original_data)
  plot_heatmap(matrices.loc['CIDRa'].dropna(how='all', axis=1))
  plot_heatmapmatrices.loc['DBLa'].dropna(how='all', axis=1))
