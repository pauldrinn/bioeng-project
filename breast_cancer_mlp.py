import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import pandas as pd

data = pd.read_csv('data.csv')

# Number of rows to be trained (percentage of data used for training)
ndata_train = round(len(data) * 0.85)

# Extracting and slicing data
train_data = data.iloc[:ndata_train, 2:]
test_data = data.iloc[ndata_train:, 2:]
train_labels = data.iloc[:ndata_train, 1].to_numpy()
test_labels = data.iloc[ndata_train:, 1].to_numpy()

normalized_train = (train_data / np.max(train_data)).to_numpy()
normalized_test = (test_data / np.max(test_data)).to_numpy()


# Turn labels into binary format and reshape
test_labels[test_labels == 'B'] = 0; test_labels[test_labels == 'M'] = 1
train_labels[train_labels == 'B'] = 0; train_labels[train_labels == 'M'] = 1

train_labels = train_labels.reshape(train_labels.shape[0], 1)
test_labels = test_labels.reshape(test_labels.shape[0], 1)

# Create the model
model = Sequential()
model.add(Dense(20, input_dim=30, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=500, batch_size=16, verbose=1, validation_split=0.15)

test_results = model.evaluate(test_data, test_labels, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=1.0)
plt.plot(history.history['val_loss'], 'b', linewidth=1.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

plt.figure(figsize=[8, 6])
plt.plot(history.history['accuracy'], 'r', linewidth=1.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=1.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

plt.show()