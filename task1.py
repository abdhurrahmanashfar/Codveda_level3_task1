

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
df.columns = ['Passengers']

df.plot(title='Monthly Passengers', figsize=(8,4))
plt.grid()
plt.show()

decomp = seasonal_decompose(df['Passengers'], model='multiplicative')
decomp.plot()
plt.suptitle("Decomposition")
plt.show()

df['MA'] = df['Passengers'].rolling(12).mean()
df[['Passengers', 'MA']].plot(title='Moving Average', figsize=(8,4))
plt.grid()
plt.show()

model = SimpleExpSmoothing(df['Passengers']).fit(smoothing_level=0.2)
df['Smoothed'] = model.fittedvalues
df[['Passengers', 'Smoothed']].plot(title='Exponential Smoothing', figsize=(8,4))
plt.grid()
plt.show()

train = df['Passengers'][:-12]
test = df['Passengers'][-12:]

arima = ARIMA(train, order=(2,1,2)).fit()
forecast = arima.forecast(12)

plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt.legend()
plt.title("ARIMA Forecast")
plt.grid()
plt.show()

rmse = np.sqrt(mean_squared_error(test, forecast))
print("RMSE:", rmse)


