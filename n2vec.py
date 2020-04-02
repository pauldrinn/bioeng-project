import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec

G = nx.read_edgelist('numerical_edge_list.csv')

n2v = Node2Vec(G, workers = 4)

model.n2v.fit()
model.wv.save_word2vec_format('embeddings.emb')