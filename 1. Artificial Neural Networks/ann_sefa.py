# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:53:20 2019

@author: Sefa3
"""

#-----------------------Part 1 : Data Preprocessing----------------------------
""" 
The problem that we are about to deal with is a classification problem.
We have several independent variables, like credit score, the balance, 
the number of products...
And based on these independent variables,
we are trying to predict which customers are leaving the bank.
ANN can do a terrific job at doing this,
and making that kind of predictions...
"""
#------------------------------------------------------------------------------
"""
* Theano Libray
Theano is an open source numerical computations library,
very efficient for fast numerical computations.
And that is based on numpy syntax.
"""
#------------------------------------------------------------------------------
"""
* Tensorflow Library
Tensorflow is another numerical computations library
that runs very fast computations.
And that can run our CPU or GPU
CPU : Central Processing Unit
GPU : Graphical Processing Unit
"""
#------------------------------------------------------------------------------
"""
* Keras Library
The Keras library is an amazing library to build deep learning models,
in a few lines of code.
Keras is a library based on Theano and Tensorflow,
and exactly as we use scikit-learn to build very efficiently 
machine learning models.
"""
#------------------------------------------------------------------------------
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#------------------------------------------------------------------------------
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#------------------------------------------------------------------------------
# the key thing to understand here is that
# all these variables here are independent variables.
# but the last columns is our dependent variable.
# 1 : exited, 0 : stayed.
dataset.head()
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13:14].values
X
y
#------------------------------------------------------------------------------
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X1 = LabelEncoder()
X[:, 1]         = labelencoder_X1.fit_transform(X[:, 1])
X

labelencoder_X2 = LabelEncoder()
X[:, 2]         = labelencoder_X2.fit_transform(X[:, 2])
X
#------------------------------------------------------------------------------
"""
to create dummy variables
our categorical variables are not ordinal
that means that there is no relational order
between our categorical variables.
France is not higher than Germany, ...
to avoid dummy variable trap;
"""
onehotencoder   = OneHotEncoder(categorical_features = [1])
X               = onehotencoder.fit_transform(X).toarray()
#------------------------------------------------------------------------------
# first 3 variables : dummy variables
X = X.astype('float64')
# to avoid dummy variable trap,
# we need to drop the first column.
X = X[:, 1:]
#------------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)
#------------------------------------------------------------------------------
"""
Feature Scaling
we need to apply feature scaling
to ease all these calculations.
because, we don't want to have one independent variable
that dominating another one.
"""
from sklearn.preprocessing import StandardScaler
sc      = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
#------------------------------------------------------------------------------
#______________________________________________________________________________

#--------------------Part 2 : Now let's make the ANN!--------------------------
# Importing the Keras libraries and packages
# import tensorflow as tf

# to initialize our neural network.
# from tensorflow.keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense
# this is the model will use to create layers in our ANN.
# from tensorflow.keras.layers import Dense
#------------------------------------------------------------------------------
# Initializing the ANN
classifier = Sequential()
#------------------------------------------------------------------------------
"""
Adding the input layer and the first hidden layer
we'll choose the 'rectifier activation function' for the hidden layers
and we'll choose the 'sigmoid activation function' for the output layer.
"""
# output_dim = (11 + 1) / 2 = 6
classifier.add(Dense(6, activation = 'relu', input_shape = (11, )))
#------------------------------------------------------------------------------
# Adding the second hidden layer
classifier.add(Dense(6, activation = 'relu'))
#------------------------------------------------------------------------------
# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))
#------------------------------------------------------------------------------
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
#------------------------------------------------------------------------------
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#------------------------------------------------------------------------------
#______________________________________________________________________________

#----------Part 3 : Making the predictions and evaluating the model------------
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm       = confusion_matrix(y_test, y_pred)
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
#------------------------------------------------------------------------------
#______________________________________________________________________________

#---------------Part 4 : Predicting a single new observation-------------------
"""
Predict if the customer with the following information will leave the bank:
    Geography           : France = [0, 0] <-- corresponds to
    Credit Score        : 600
    Gender              : Male   = [1]    <-- corresponds to
    Age                 : 40
    Tenure              : 3
    Balance             : 60000
    Number of Products  : 2
    Has Credit Card     : Yes    = [1]    <-- corresponds to
    Is Active Member    : Yes    = [1]    <-- corresponds to
    Estimated Salary    : 50000
"""

my_array       = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
normal_array   = sc.transform(my_array)  # to normalize  
new_prediction = classifier.predict(my_array)
new_y_pred     = (new_prediction > 0.5)
#______________________________________________________________________________

#-------------Part 5 : Evaluating, Improving and Tuning the ANN----------------
# Evaluating the ANN
'''
to fix this variance problem;
k-Fold Cross Validation fix it by splitting the training set
into 10 folds when K = 10, and most of the time K = 10
and we train our model on 9-folds and we test it on the
last remaining fold.
there we take 10 different combination of 9-folds to train
a model and 1-fold to test it.
that means we can train the model and test the model
on 10 combinations of training and test sets.
And that will give us a much better idea of the model
performance because, we take an average of different
accuracies of the 10 evaluations and also compute
the standart deviation to have a look at the variance.
So eventually, our analysis will be much more relevant.
'''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, activation = 'relu', input_shape = (11, )))
    classifier.add(Dense(6, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
    return classifier

classifier_cv   = KerasClassifier(build_fn = build_classifier,
                                batch_size = 10, epochs = 100, verbose = 0)

accuracies      = cross_val_score(estimator = classifier_cv, X = X_train,
                             y = y_train, cv = 10, n_jobs = 1)

mean            = accuracies.mean()
variance        = accuracies.std()
"""
we are in 'Low Bias Low Variance'.
means, best accuracy low varince
accuracy : % 85.9
variance : % 1.22
"""
#------------------------------------------------------------------------------
# Improving the ANN
"""
Dropout Regularization:
it is the solution for overfitting in deep learning.
Overfitting is when your model was trained too much
on the training set, too much that it becomes much less
performance on the test set and we can observe this
when we have large difference of accuracies between
training set and the test set.
Generally, when overfitting happens, you have a much
higher accuracy on the training set than the test set.
And another way to detect overfitting is when you
observe high variance when applying k-fold cv
because indeed,  when it's overfitted on the training
set, that is when your model learn too much and
this may cause your model won't succeed on 'other' test
sets because the correlations learned too much.
"""
# Dropout Regularization to reduce overfitting if needed.
from keras.layers import Dropout

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, activation = 'relu', input_shape = (11, )))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(6, activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.add(Dropout(p = 0.1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#------------------------------------------------------------------------------
# Tuning the ANN
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    
    classifier.add(Dense(6, activation = 'relu', input_shape = (11, )))
    classifier.add(Dense(6, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    
    classifier.compile(optimizer = optimizer, 
                       loss      = 'binary_crossentropy',
                       metrics   = ['accuracy']
                       )
    return classifier

classifier_cv   = KerasClassifier(build_fn = build_classifier)

parameters      = {'batch_size' : [25, 32],
                   'epochs'     : [100, 500],
                   'optimizer'  : ['adam', 'rmsprop']
                   }

grid_search     = GridSearchCV(estimator  = classifier_cv,
                               param_grid = parameters,
                               scoring    = 'accuracy',
                               cv = 10
                               )

grid_seach_cv   = grid_search.fit(X = X_train, y = y_train)
best_parameters = grid_seach_cv.best_params_
best_accuracy   = grid_seach_cv.best_score_

# best_parameters = {'batch_size': 32, 'epochs': 500, 'optimizer': 'rmsprop'}
# best_accuracy = 0.860125
#------------------------------------------------------------------------------
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, activation = 'relu', input_shape = (11, )))

# Adding the second hidden layer
classifier.add(Dense(6, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)

y_pred_tuned   = classifier.predict(X_test)
y_pred_tuned   = (y_pred_tuned > 0.5)

cm_tuned       = confusion_matrix(y_test, y_pred_tuned)
accuracy_tuned = (cm_tuned[0, 0] + cm_tuned[1, 1]) / np.sum(cm_tuned)

# accuracy_tuned : % 86.4
#______________________________________________________________________________