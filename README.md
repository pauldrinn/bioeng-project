## Bioengineering project/thesis

### **Gene expression**
Data containing gene expression for breast cancer patients is called **brca_counts.rds** and it needs to be converted to *.csv* as python cannot use *.rds* files.
```R
brca <- readRDS('brca_counts.rds')
write.csv(brca, 'brca.csv')
```

### **Node2Vec**
The functional network used is from
> Wu, G., Feng, X. & Stein, L. A human functional protein interaction network and its application to cancer data analysis. Genome Biol 11, R53 (2010). https://doi.org/10.1186/gb-2010-11-5-r53

Node2Vec only allows integers as nodes and the nodes of the network file above are UniProt accession numbers so it has to be converted:
```R
library(tidygraph)
library(dplyr)

network_table <- read.table('FIs_043009.txt', header = FALSE)
networkasgraph <- as_tbl_graph(network_table)

networkasgraph %>%
  activate(edges) %>%
  write.table('n2v-in.edgelist', row.names = FALSE, col.names = FALSE)
```
Then, Node2Vec is run with the default parameters:
```
./node2vec -i:n2v-in.edgelist -o:n2v-out.emb -v
```
And the Node2Vec output is visualized using UMAP:
```py
import numpy as np
import pandas as pd
import umap
import umap.plot

emb = pd.read_csv('n2v-out.emb', skiprows = 1, sep = ' ', header = None, index_col=0)
emb_data = emb.to_numpy()
emb_labels = emb.index.to_numpy()

embmapper = umap.UMAP(n_epochs= 400).fit(emb_data)

p = umap.plot.points(embmapper, labels = emb_labels, show_legend= False, theme = 'fire')

umap.plot.show(p)
```
![UMAP figure](UMAP.png)