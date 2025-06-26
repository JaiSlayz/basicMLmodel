#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:14:36 2025

@author: jaisaxena
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


df = yf.download('AAPL', start = '2019-01-01', end = '2024-01-01')
df.head()

df['Close'].plot(figsize=(12,6), title='AAPL Close Price')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Lag-based prediction (e.g., predict tomorrow's price using today's features)
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Target'] = df['Close'].shift(-1)  # predict next day's closing price

# 3. Drop rows with NaN (from rolling and shift)
df.dropna(inplace=True)

# 4. Define features (X) and target (y)
X = df[['Close', 'MA_20']]
y = df['Target']

# 5. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print("Mean Squared Error:", mse)

# 8. Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, preds, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show() 