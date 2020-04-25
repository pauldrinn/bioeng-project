import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extractPatientInfo(file, col = 1):
    patientdf = pd.read_csv(file)
    patientdf.rename(columns={patientdf.columns[0]: "Gene stable ID"}, inplace = True)
    return patientdf.iloc[:, :col + 1]

def createPatientArray(file):
    """ Creates an image-ready array filled with patient gene expression data as colored pixels.

    file: File name to extract the patient information from. File should have at least 2 columns with the first one containing ENSEMBL gene names and the second one containing expression levels.
    """
    patientex = extractPatientInfo(file)
    template = pd.read_csv('template-data.csv', index_col = 0)
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
    
    return nimg