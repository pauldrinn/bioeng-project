import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'txt', 'csv', 'tsv'}

predictor = load_model('static/brca/2')

def extractPatientInfo(file, col = 1):
    patientdf = pd.read_csv(file)
    patientdf.rename(columns={patientdf.columns[0]: "Gene stable ID"}, inplace = True)
    return patientdf.iloc[:, :col + 1]

def createPatientArray(file):
    """ 
    Creates an image-ready array filled with patient gene expression data as colored pixels.

    file: Name of the file from which the patient gene expression data will be extracted. File should have at least 2 columns with the first one containing ENSEMBL gene names and the second one containing expression levels.
    """
    patientex = extractPatientInfo(file)
    template = pd.read_csv('static/template-data.csv', index_col = 0)
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

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html', pred = '')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['userdata']
        f.save(secure_filename(f.filename))
        parray = createPatientArray(secure_filename(f.filename))
        parray = np.expand_dims(parray, axis = 0)
        prediction = np.argmax(predictor.predict(parray), axis=-1)
        if prediction == [1]:
            prediction = "sorry :/ it's a tumor"
        elif prediction == [0]:
            prediction = "yay :D it's not a tumor"
        print('INFO Predictions: {}'.format(prediction))
        os.remove(secure_filename(f.filename))

    return render_template('index.html', pred=prediction)


def main():
    app.run()


if __name__ == '__main__':
    main()
