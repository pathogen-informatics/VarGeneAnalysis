import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def split_sample_names(data):
  data.set_index('name', inplace=True)
  data.columns = ['CIDRa', 'DBLa']
  data = data.stack()
  data.index.names = ['sample_name', 'cluster_domain']
  data = data.reset_index()
  data.columns = ['sample_name', 'cluster_domain', 'cluster']
  data['isolate'], data['gene'], data['domain'] = zip(*data.sample_name.map(lambda name: name.split('.',3)[:3]))
  del data['sample_name']
  data = data[['isolate', 'gene', 'domain', 'cluster_domain', 'cluster']]
  return data

def remove_silly_classifications(data):
  data = data[data.domain == data.cluster_domain]
  del data['cluster_domain']
  return data

def remove_duplicates(data):
  """Remove genes which have more than one of each domain"""
  counts = data.groupby(['isolate', 'gene', 'domain']).apply(len)
  duplicate_counts = counts[counts > 1].reset_index()
  del duplicate_counts[0]
  data = data.set_index(['isolate', 'gene', 'domain'])
  data_to_remove = data.loc[map(tuple, duplicate_counts.values)]
  data.drop(data_to_remove.index, inplace=True)
  return data

def get_domain_classifications(data, domain_a, domain_b):
  domain_data = data.unstack(2)
  domain_data = domain_data['cluster']
  domain_data.reset_index(inplace=True)
  domain_data = domain_data.groupby([domain_a, domain_b]).apply(len)
  domain_data = domain_data.unstack()
  return domain_data

def plot_heatmap(matrix, title=None, block=False):
  plt.figure()
  sns.heatmap(matrix, mask=np.isnan(matrix))
  plt.yticks(rotation=0)
  plt.xticks(rotation=90)
  if title:
    plt.suptitle(title)
  plt.show(block=block)

def normalise_along_axis(matrix, axis=0):
  assert axis in [0,1], "axis should be 0 or 1"
  if axis == 0:
    return matrix / np.sum(matrix, axis=0)
  elif axis == 1:
    return matrix.T / np.sum(matrix, axis=1)

def cross_compare(data):
  clusters = list(set(data['cluster']))
  data_dict = {a: [1 if a != b else 0 for b in clusters] for a in clusters}
  index = clusters
  return pd.DataFrame(data_dict, index=index)

def create_cross_matrix(data, domain):
  data = data.reset_index()
  data = data[data['domain'] == domain]
  data = data[['isolate', 'cluster']]
  data = data.groupby('isolate').apply(cross_compare)
  data = data.stack().reset_index()
  data.columns = ['isolate', 'cluster_a', 'cluster_b', 'count']
  counts = data.groupby(['cluster_a', 'cluster_b']).apply(lambda df: np.sum(df['count']))
  counts = counts.reset_index()
  counts.columns = ['cluster_a', 'cluster_b', 'count']
  counts.loc[counts['cluster_a'] == counts['cluster_b'],'count'] = np.nan
  counts = counts.set_index(['cluster_a', 'cluster_b'])
  counts = counts.unstack()
  return counts

if __name__ == '__main__':
  data = pd.read_csv("subsets_3456.CIDRa_k12_DBLa_k6.table.log", delimiter="\t")
  data = split_sample_names(data)
  data = remove_silly_classifications(data)
  data = remove_duplicates(data)
  matrix = get_domain_classifications(data, 'CIDRa', 'DBLa')
  plot_heatmap(matrix)
  plot_heatmap(normalise_along_axis(matrix, 0), "Normalise by DBLa count")
  plot_heatmap(normalise_along_axis(matrix, 1), "Normalise by CIDRa count")
  
  # Some obvious correlations
  
  counts = create_cross_matrix(data, 'CIDRa')
  plot_heatmap(counts, 'CIDRa / CIDRa matrix')
  counts = create_cross_matrix(data, 'DBLa')
  plot_heatmap(counts, 'DBLa / DBLa matrix')
  
  # no disernably pattern
