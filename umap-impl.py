import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import umap.plot

metrics = [
        'euclidean',
        'manhattan',
        'chebyshev',
        'minkowski',
        'canberra',
        'braycurtis',
        'mahalanobis',
        'wminkowski',
        'seuclidean',
        'cosine',
        'correlation',
        'haversine',
        'hamming',
        'jaccard',
        'dice',
        'russelrao',
        'kulsinski',
        'll_dirichlet',
        'hellinger',
        'rogerstanimoto',
        'sokalmichener',
        'sokalsneath', 
        'yule']

emb = pd.read_csv('n2v-out.emb', skiprows = 1, sep = ' ', header = None, index_col=0)
emb_data = emb.to_numpy()
emb_labels = emb.index.to_numpy()

for m in metrics:
    for n in metrics:
        filen = 'plots/' + 'M-' + m + '_' + 'OM-' + n + '.png'
        try:
            embmapper = umap.UMAP(metric = m, output_metric = n, n_epochs= 400).fit(emb_data)
            umap.plot.points(embmapper, labels = emb_labels, show_legend= False, theme = 'fire')
            plt.savefig(filen)
        except:
            print(filen + ' failed')

'''
hover_data = pd.DataFrame({'index': np.arange(len(emb_labels)), 'label': emb_labels})
umap.plot.output_notebook()
p = umap.plot.interactive(embmapper, labels = emb_labels, hover_data = hover_data, point_size = 2)
umap.plot.show(p)
'''