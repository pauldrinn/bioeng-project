import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import umap.plot

emb = pd.read_csv('fi.emb', skiprows = 1, sep = ' ', header = None, index_col=0)
emb_data = emb.to_numpy()
emb_labels = emb.index.to_numpy()

embmapper = umap.UMAP(n_epochs= 400).fit(emb_data)


#p = umap.plot.points(embmapper, labels = emb_labels, show_legend= False, theme = 'fire')

hover_data = pd.DataFrame({'index': np.arange(len(emb_labels)), 'label': emb_labels})
umap.plot.output_notebook()
p = umap.plot.interactive(embmapper, labels = emb_labels, hover_data = hover_data, point_size = 2)

umap.plot.show(p)