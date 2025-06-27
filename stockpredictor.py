#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:14:36 2025

@author: jaisaxena
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = yf.download('AAPL', start = '2019-01-01', end = '2024-01-01')
df.head()

df['Close'].plot(figsize=(12,6), title='AAPL Close Price')
plt.show()



# Lag-based prediction (predict tomorrow's price using today's features)
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Target'] = df['Close'].shift(-1)  # predict next day's closing price

# Data Clean any NaNs
df.dropna(inplace=True)

# Define features (X) and target (y)
X = df[['Close', 'MA_20']]
y = df['Target']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print("Mean Squared Error:", mse)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, preds, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show() 
