import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.dates as mdates

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

# Function to calculate statistics and daily returns
def calculate_statistics_and_returns(data):
    mean_price = data['Adj Close'].mean()
    std_price = data['Adj Close'].std()
    data['Daily Return'] = data['Adj Close'].pct_change()
    return mean_price, std_price, data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Momentum
def calculate_momentum(data, window=10):
    return data['Adj Close'] - data['Adj Close'].shift(window)

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    exp_short = data['Adj Close'].ewm(span=short_window, adjust=False).mean()
    exp_long = data['Adj Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp_short - exp_long
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    data['SMA'] = data['Adj Close'].rolling(window=window).mean()
    data['Upper Band'] = data['SMA'] + 2 * data['Adj Close'].rolling(window=window).std()
    data['Lower Band'] = data['SMA'] - 2 * data['Adj Close'].rolling(window=window).std()
    return data

# Function to calculate Stochastic Oscillators
def calculate_stochastic_oscillators(data, window=14):
    data['%K'] = (data['Adj Close'] - data['Low'].rolling(window=window).min()) / (
            data['High'].rolling(window=window).max() - data['Low'].rolling(window=window).min()) * 100
    data['%D'] = data['%K'].rolling(window=3).mean()  # You can adjust the window for %D
    return data

# Function to calculate Average Directional Index (ADX)
def calculate_adx(data, window=14):
    delta_high = data['High'].diff()
    delta_low = -data['Low'].diff()
    tr = pd.concat([delta_high, delta_low], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    data['+DI'] = (delta_high / atr * 100).ewm(span=window).mean()
    data['-DI'] = (delta_low / atr * 100).ewm(span=window).mean()
    data['ADX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']) * 100).ewm(span=window).mean()
    return data

# Function to create a candlestick chart
def create_candlestick_chart(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    fig.show()

# Function to generate MACD and RSI-based buy/sell signals
def generate_macd_rsi_signals(data, rsi_overbought=60, rsi_oversold=41):
    signals = pd.DataFrame(index=data.index)

    # Calculate RSI signals
    signals['RSI_Buy'] = (data['RSI'] < rsi_oversold) # Buy when RSI is below oversold levels
    signals['RSI_Sell'] = (data['RSI'] > rsi_overbought) # Sell when RSI is above overbought levels

    # Calculate MACD signals
    signals['MACD_Buy'] = (data['MACD'] > data['Signal Line'])  # Buy when MACD crosses above the signal line
    signals['MACD_Sell'] = (data['MACD'] < data['Signal Line'])  # Sell when MACD crosses below the signal line

    # Combine signals
    signals['Combined_Buy'] = signals['RSI_Buy'] & signals['MACD_Buy']  # Buy when both RSI and MACD indicate buy
    signals['Combined_Sell'] = signals['RSI_Sell'] & signals['MACD_Sell']  # Sell when both RSI and MACD indicate sell

    return signals

# Main analysis and visualization
def main():

    # Step 1: Define stock symbol and date range
    symbol = input('What stock you want: ')
    start_date = input('From what date (YYYY-MM-DD): ')
    end_date = input('To what date (YYYY-MM-DD): ')

    # Step 2: Fetch historical data
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Step 3: Calculate statistics and daily returns
    mean_price, std_price, stock_data = calculate_statistics_and_returns(stock_data)

    # Calculate Bollinger Bands
    stock_data = calculate_bollinger_bands(stock_data)

    # Calculate Stochastic Oscillators
    stock_data = calculate_stochastic_oscillators(stock_data)

    # Calculate ADX
    stock_data = calculate_adx(stock_data)

    # Create subplots with 8 rows and 1 column (including new indicators)
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(12, 15))

    # Plot 1: Daily Returns with Zero Line
    axes[0].plot(stock_data.index, stock_data['Daily Return'])
    axes[0].axhline(y=0, color='r', linestyle='--', label='Zero Line')
    axes[0].set_title('Daily Returns')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Percentage Change')
    axes[0].legend()
    axes[0].grid()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Calculate and Plot 50-Day and 200-Day SMAs
    stock_data['SMA_50'] = stock_data['Adj Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Adj Close'].rolling(window=200).mean()

    # Plot 2: 50-Day and 200-Day SMAs
    axes[1].plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', alpha=0.7)
    axes[1].plot(stock_data.index, stock_data['SMA_50'], label='50-Day SMA', alpha=0.7)
    axes[1].plot(stock_data.index, stock_data['SMA_200'], label='200-Day SMA', alpha=0.7)
    axes[1].set_title(f'{symbol} Stock Price Analysis')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Calculate and Plot RSI
    stock_data['RSI'] = calculate_rsi(stock_data)

    # Plot 3: RSI with Overbought and Oversold Lines
    axes[2].plot(stock_data.index, stock_data['RSI'], label='RSI', color='orange')
    axes[2].axhline(70, color='red', linestyle='--', label='Overbought (70)')
    axes[2].axhline(30, color='green', linestyle='--', label='Oversold (30)')
    axes[2].set_title(f'{symbol} RSI Analysis')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('RSI')
    axes[2].legend()
    axes[2].grid()
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Plot 4: Momentum
    stock_data['Momentum'] = stock_data['Adj Close'] - stock_data['Adj Close'].shift(10)
    axes[3].plot(stock_data.index, stock_data['Momentum'], label='Momentum', color='purple')
    axes[3].set_title(f'{symbol} Momentum Analysis')
    axes[3].set_xlabel('Date')
    axes[3].set_ylabel('Momentum')
    axes[3].legend()
    axes[3].grid()
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Calculate and Plot MACD
    exp_short = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
    exp_long = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = exp_short - exp_long
    stock_data['Sign      al Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # # Plot 5: MACD with Signal Line
    # axes[4].plot(stock_data.index, stock_data['MACD'], label='MACD', color='blue')
    # axes[4].plot(stock_data.index, stock_data['Signal Line'], label='Signal Line', color='red')
    # axes[4].set_title('MACD Analysis')
    # axes[4].set_xlabel('Date')
    # axes[4].set_ylabel('MACD')
    # axes[4].legend()
    # axes[4].grid()

    # Plot 6: Bollinger Bands
    axes[5].plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', alpha=0.7)
    axes[5].plot(stock_data.index, stock_data['SMA'], label='SMA', alpha=0.7)
    axes[5].fill_between(stock_data.index, stock_data['Upper Band'], stock_data['Lower Band'], color='gray', alpha=0.2)
    axes[5].set_title(f'{symbol} Bollinger Bands')
    axes[5].set_xlabel('Date')
    axes[5].set_ylabel('Price')
    axes[5].legend()
    axes[5].grid()
    axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Plot 7: Stochastic Oscillators (%K and %D)
    axes[6].plot(stock_data.index, stock_data['%K'], label='%K', color='blue')
    axes[6].plot(stock_data.index, stock_data['%D'], label='%D', color='red')
    axes[6].axhline(80, color='red', linestyle='--', label='Overbought (80)')
    axes[6].axhline(20, color='green', linestyle='--', label='Oversold (20)')
    axes[6].set_title(f'{symbol} Stochastic Oscillators')
    axes[6].set_xlabel('Date')
    axes[6].set_ylabel('Value')
    axes[6].legend()
    axes[6].grid()
    axes[6].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Plot 8: ADX
    axes[4].plot(stock_data.index, stock_data['ADX'], label='ADX', color='purple')
    axes[4].axhline(25, color='red', linestyle='--', label='Strong Trend (ADX > 25)')
    axes[4].set_title(f'{symbol} ADX')
    axes[4].set_xlabel('Date')
    axes[4].set_ylabel('ADX')
    axes[4].legend()
    axes[4].grid()
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    # Adjust spacing between subplots
    plt.tight_layout()

    # Create a candlestick chart
    create_candlestick_chart(stock_data)

    # Show all plots together
    plt.show()

if __name__ == "__main__":
    main()

#Explanation
# Moving Averages (SMA_20 and SMA_50):
#
# Usage: Moving averages are used to identify trends and potential entry/exit points.
# How to Use:
# When the shorter-term SMA (e.g., SMA_20) crosses above the longer-term SMA (e.g., SMA_50), it's often called a "Golden Cross," suggesting a potential uptrend. This can be a buy signal.
# Conversely, when the shorter-term SMA crosses below the longer-term SMA (a "Death Cross"), it may indicate a potential downtrend and can be a sell signal.

# Relative Strength Index (RSI):
#
# Usage: RSI measures the speed and change of price movements and helps identify overbought or oversold conditions.
# How to Use:
# RSI values above 70 are considered overbought, indicating a potential reversal or correction. It can be a sell signal.
# RSI values below 30 are considered oversold, indicating a potential buying opportunity. It can be a buy signal.
# Combining RSI with other indicators, like moving averages, can provide stronger signals.

# Momentum:
#
# Usage: Momentum indicates the rate of change in stock prices.
# How to Use:
# Positive momentum suggests upward price movement, and negative momentum suggests downward movement.
# Momentum can be used to confirm trends. For example, if prices are rising, and momentum is positive, it may support a buy signal.
# Conversely, if prices are falling, and momentum is negative, it may support a sell signal.

# Moving Average Convergence Divergence (MACD):
#
# Usage: MACD is used to identify changes in the strength, direction, momentum, and duration of a trend.
# How to Use:
# When the MACD line (in blue) crosses above the signal line (in red), it's considered a bullish signal (potential buy).
# When the MACD line crosses below the signal line, it's considered a bearish signal (potential sell).
# Additionally, the distance between the MACD and signal line can indicate the strength of the trend.

# Candlestick Chart:
#
# Usage: Candlestick charts provide a visual representation of price movements.
# How to Use:
# Candlestick patterns, like doji, engulfing, and hammer, can be used to identify potential reversals or continuations.
# For example, a bullish engulfing pattern might indicate a buy signal.

# Generated Signals (RSI and MACD Buy/Sell Signals):
#
# Usage: These signals combine RSI and MACD indicators to generate potential buy and sell signals.
# How to Use:
# "RSI_Buy" and "MACD_Buy" signals can suggest a buy opportunity when they coincide.
# "RSI_Sell" and "MACD_Sell" signals can suggest a sell opportunity when they coincide.
# Combining multiple indicators can enhance the reliability of signals.

