import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import yfinance as yf
from scipy.optimize import minimize

# df = pd.read_csv('dados.csv',sep=';')
# df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y')
# df.set_index('Date',inplace=True)

# Define parameters
start_date = '2019-01-01'
end_date = '2022-01-01'


def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']]
    return df

def Modelo(data, short_window, long_window):
    indicador = pd.DataFrame(index=data.index)
    indicador['signal'] = 0.0
    indicador['profit']=0.0

    # Short e long AVGs
    indicador['short_mavg'] = data.rolling(window=short_window, min_periods=1, center= False).mean()
    indicador['long_mavg'] = data.rolling(window=long_window, min_periods=1, center=False).mean()

    # Sinal de compra/venda
    indicador['signal'][short_window:] = np.where(
        indicador['short_mavg'][short_window:] > indicador['long_mavg'][short_window:],1.0,0.0
    )
    indicador['posicao'] = indicador['signal'].diff()

    # Calculate profit based on operations
    buy_mask = indicador['posicao'] > 0.0
    sell_mask = indicador['posicao'] < 0.0

    for buy_index in indicador.index[buy_mask]:
        next_sell_index = indicador.index[sell_mask & (indicador.index > buy_index)].min()

        if pd.notna(next_sell_index):
            # Calculate profit for buy operations and store it in 'profit' column
            indicador.loc[buy_index, 'profit'] = -(data['Close'][buy_index] + 5)
        else: indicador.loc[buy_index, 'posicao'] = 0.0

    # Calculate profit for sell operations and store it in 'profit' column
    indicador.loc[sell_mask, 'profit'] = data['Close'][sell_mask] - 5

    # Cumulative profit over time
    indicador['cumulative_profit'] = indicador['profit'].cumsum()

    return indicador

def plot_strategy(data, signals, title='grafico'):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot closing price
    data['Close'].plot(ax=ax, label='Close Price')

    # Plot short and long moving averages
    signals[['short_mavg', 'long_mavg']].plot(ax=ax)

    # Plot buy signals
    ax.plot(
        signals.loc[signals.posicao == 1.0].index,
        signals.short_mavg[signals.posicao == 1.0],
        '^',
        markersize=10,
        color='g',
    )

    # Plot sell signals
    ax.plot(
        signals.loc[signals.posicao == -1.0].index,
        signals.short_mavg[signals.posicao == -1.0],
        'v',
        markersize=10,
        color='r',
    )

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()


# Fetch historical data for soybean and corn
soybean_data = fetch_data('ZS=F', start_date, end_date)  # Soybean futures
corn_data = fetch_data('ZC=F', start_date, end_date)  # Corn futures

# # Apply moving average strategy
# soybean_signals = Modelo(soybean_data, short_window, long_window)
# corn_signals= Modelo(corn_data, short_window, long_window)

def find_best_parameters(data):
    best_profit = -float('inf')
    best_short_window = None
    best_long_window = None

    for short_window in range(5, 30):  # Adjust the range based on your preferences
        for long_window in range(short_window + 1, 60):  # Adjust the range based on your preferences
            # Calculate profits using current parameters
            current_model = Modelo(data, short_window, long_window)

            # Get the final cumulative profit
            final_profit = current_model['cumulative_profit'].iloc[-1]

            # Update best parameters if the current result is better
            if final_profit > best_profit:
                best_profit = final_profit
                best_short_window = short_window
                best_long_window = long_window

    return best_short_window, best_long_window, best_profit

##################
Dados = corn_data
##################

melhorResultado = find_best_parameters(Dados)
print(melhorResultado)

Otimizado = Modelo(Dados,melhorResultado[0],melhorResultado[1])
print(Otimizado)

plot_strategy(Dados,Otimizado)



