import numpy as np
import pandas as pd

groundt = pd.read_csv('samples_fileid_tumor.tsv', sep = '\t', index_col = 0)
brcaex = pd.read_csv('brca.csv').rename(columns = {'Unnamed: 0': 'Gene stable ID'})
template = pd.read_csv('template-data.csv', index_col = 0)
genes = brcaex.iloc[:, 0]

allimg = np.zeros((brcaex.shape[1] - 1, 300, 300, 3)).astype('uint8')

for a in range(brcaex.shape[1]):
    if a == brcaex.shape[1] - 1:
        continue

    patientex = pd.concat([genes, brcaex.iloc[:, a + 1]], axis = 1)
    coorex = pd.merge(template, patientex).sort_values(by = 'Gene stable ID', ignore_index = True).drop(columns = ['node_id']).groupby(['x', 'y']).mean().round().astype(int)
    ((k, l), m) = zip(*coorex.index), coorex.iloc[:, -1].values
    
    nimg = np.zeros((300, 300, 3)).astype('uint8')
    
    for i in range(len(k)):
        b = m[i] & 255
        g = (m[i] >> 8) & 255
        r = (m[i] >> 16) & 255
    
        if k[i] == 300 or l[i] == 300:
            continue
        
        nimg[k[i]][l[i]] = [r, g, b]
    
    allimg[a] = nimg


intermediatedf = pd.merge(brcaex.iloc[0:1,:], groundt.transpose(), how = 'outer').dropna(axis = 1)
labelarray = intermediatedf.loc[1].to_numpy().astype('uint8')

np.savez_compressed('train-data', data = allimg, labels = labelarray)