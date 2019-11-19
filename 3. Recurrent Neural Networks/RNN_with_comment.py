# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 08:51:33 2019

@author: Sefa3
"""

"""
We are going to predict the stock price of Google.

We will make an LSTM that will try to capture the downward and upward trend
of the Google stock price.

We are going to train our LSTM model on five years of the Google stock price.

The begging of 2012 to the end of 2016.

We will try to predict the first month of 2017. (January 2017)
"""

# ----------------------- Part 1 : Data Preprocessing -------------------------
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
df            = dataset_train.copy()
training_set  = df.iloc[:, 1:2].values
#------------------------------------------------------------------------------
"""
Whenever you build an RNN and especially if there is a sigmoid function as
the activation function in the output layer of your RNN, I recommend to
apply the normalization.

x_norm = (x - min(x)) / (max(x) - min(x))

"""
# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set) # normalization
#------------------------------------------------------------------------------
"""
At each time T, the RNN is going to look at 60 stock prices before time T,
and based on the trends it is capturing during these 60 previous timesteps,
it will try to predict the next output.

So 60 timespteps of the past information from which our RNN is going to try
to learn and understand some correlations, or some trends. And based on it's
understanding, it's going to try to predict the next output.
"""
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
#------------------------------------------------------------------------------
"""
Predictors are indicators. We have one indicator.

Adding more dimensionality to this data structure, we will be able to add more 
indicators if you think could help to predict better.
"""
# Reshaping
# new dimension of X_train corresponding to the indicator: (1198, 60, 1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#______________________________________________________________________________

# ----------------------- Part 2 : Building the RNN ---------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#------------------------------------------------------------------------------
# Initialising the RNN
regressor = Sequential()
#------------------------------------------------------------------------------
"""
To avoid 'overfitting' we need to use Dropout regularisation.
"""
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#------------------------------------------------------------------------------
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))
#------------------------------------------------------------------------------
# Adding the output layer
regressor.add(Dense(units = 1))
#------------------------------------------------------------------------------
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#------------------------------------------------------------------------------
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#______________________________________________________________________________

# ------- Part 3 : Making the predictions and visualising the results ---------
# Getting the real stock price of 2017
dataset_test     = pd.read_csv('Google_Stock_Price_Test.csv')
df_test          = dataset_test.copy()
real_stock_price = df_test.iloc[:, 1:2].values
#------------------------------------------------------------------------------
"""
We trained our model to be able to predict the stock price at time T+1 based
on the 60 previous stock prices, therefore; to predict each stock price of
financial day of January 2017, we will need the 60 previous stock prices of the
60 previous financial days, before the actual day.

If we scaled real_stock_price, we will change the actual test values.

We have to keep the actual test values as they are, therefore; we will handle
this problem by making another concatenation which will be to concatenate
the original DataFrames that we wtill we still have because these are
dataset_train.

From this concatenation we will get the input of each prediction, that is the
60 produced stock prices at each time T and this is then what we will scale.
"""
# Getting the predicted stock price of 2017
dataset_total = pd.concat((df['Open'], df_test['Open']), axis = 0)
df_total      = dataset_total.copy()
inputs        = df_total[len(df_total) - len(df_test) - 60 :].values
inputs        = np.reshape(inputs, (-1, 1))
inputs        = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#------------------------------------------------------------------------------
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Price')
plt.legend()
plt.show()

#______________________________________________________________________________