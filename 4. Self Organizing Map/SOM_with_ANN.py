# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:12:45 2019

@author: Sefa3
"""

# ----------------| Mega Case Study - Make a Hyprid Deep Learning Model |----------------

# -------------| Part 1 : Identify the frauds with the Self-Organizing Map |-------------
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#----------------------------------------------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
df      = dataset.copy()
X       = df.iloc[:, :-1].values
y       = df.iloc[:, -1].values
#----------------------------------------------------------------------------------------
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
#----------------------------------------------------------------------------------------
# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X) # to initialize the weights randomly.
som.train_random(data = X, num_iteration = 100) # to train SOM.
#----------------------------------------------------------------------------------------
# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()                        # this is the window that will contain the map.
pcolor(som.distance_map().T)  # all the different colors corresponding to the MID's.
colorbar()                    # white colors are the outliers (frauds).
markers = ['o', 's']          # red circles(r, o) : the customers who didn't get approval.
colors  = ['r', 'g']          # green squares(g, s) : the customers who got approval.
for i, j in enumerate(X):     # i : indexes, j : all the vectors of customers at i.
    w = som.winner(j)         # winning node.
    plot(w[0] + 0.5,          # we want to put the marker at the center of the square.
         w[1] + 0.5,          # we want to put the marker at the center of the square.
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
#----------------------------------------------------------------------------------------
# Finding the frauds
mappings = som.win_map(X)
frauds   = np.concatenate((mappings[(1, 6)], mappings[(3, 7)], mappings[(2, 8)]), axis = 0)
# frauds   = np.array(mappings[(7, 2)])
frauds   = sc.inverse_transform(frauds)

# --------------| Part 2 : Going from Unsupervised to Supervised Learning |--------------

# Creating the Matrix of Features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc        = StandardScaler()
customers = sc.fit_transform(customers)

# Importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()
classifier.add(Dense(6, activation = 'relu', input_shape = (15, )))

# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 20)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]