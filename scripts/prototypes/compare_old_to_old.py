from compare_lots_of_clusters import plot_heatmap, normalise_along_axis

if __name__ == '__main__':
  original_data = pd.read_csv("original_seven_genomes_domains.table.log", delimiter='\t')
  original_data.columns = ['gene', 'CIDRa', 'DBLa']
  
  plot_heatmap(original_data.groupby(['CIDRa', 'DBLa']).apply(len).unstack())
