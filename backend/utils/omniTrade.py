import yfinance as yf  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

class OmniTradeAI:  
    def __init__(self, ticker, start_date, end_date):  
        self.ticker = ticker  
        self.start_date = start_date  
        self.end_date = end_date  
        self.data = self.fetch_data()  
        self.signals = None  

    def fetch_data(self):  
        """Fetch historical stock data from Yahoo Finance"""  
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)  
        return data  

    def calculate_sma(self, window=20):  
        """Calculate Simple Moving Average (SMA)"""  
        self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()  

    def calculate_rsi(self, window=14):  
        """Calculate Relative Strength Index (RSI)"""  
        delta = self.data['Close'].diff()  
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  
        rs = gain / loss  
        self.data['RSI'] = 100 - (100 / (1 + rs))  

    def generate_signals(self):  
        """Generate Buy/Sell signals based on SMA & RSI"""  
        self.calculate_sma()  
        self.calculate_rsi()  

        # Buy Signal: Price crosses above SMA & RSI < 30 (Oversold)  
        self.data['Buy_Signal'] = np.where(  
            (self.data['Close'] > self.data['SMA_20']) & (self.data['RSI'] < 30),  
            1,  # Buy  
            0   # No action  
        )  

        # Sell Signal: Price crosses below SMA & RSI > 70 (Overbought)  
        self.data['Sell_Signal'] = np.where(  
            (self.data['Close'] < self.data['SMA_20']) & (self.data['RSI'] > 70),  
            -1,  # Sell  
            0    # No action  
        )  

        self.signals = self.data[['Close', 'SMA_20', 'RSI', 'Buy_Signal', 'Sell_Signal']]  

    def backtest(self, initial_capital=10000):  
        """Simulate trading based on signals"""  
        if self.signals is None:  
            self.generate_signals()  

        positions = []  
        cash = initial_capital  
        shares = 0  

        for idx, row in self.signals.iterrows():  
            if row['Buy_Signal'] == 1 and cash > 0:  
                shares_bought = cash // row['Close']  
                shares += shares_bought  
                cash -= shares_bought * row['Close']  
                positions.append(('BUY', idx, row['Close'], shares, cash))  
            elif row['Sell_Signal'] == -1 and shares > 0:  
                cash += shares * row['Close']  
                positions.append(('SELL', idx, row['Close'], 0, cash))  
                shares = 0  

        final_value = cash + (shares * self.signals['Close'].iloc[-1])  
        return positions, final_value  

    def plot_signals(self):  
        """Visualize trading signals"""  
        plt.figure(figsize=(12, 6))  
        plt.plot(self.data['Close'], label='Price', alpha=0.5)  
        plt.plot(self.data['SMA_20'], label='20-Day SMA', alpha=0.75)  

        # Plot Buy/Sell signals  
        plt.scatter(  
            self.data.index[self.data['Buy_Signal'] == 1],  
            self.data['Close'][self.data['Buy_Signal'] == 1],  
            label='Buy Signal', marker='^', color='green', alpha=1  
        )  
        plt.scatter(  
            self.data.index[self.data['Sell_Signal'] == -1],  
            self.data['Close'][self.data['Sell_Signal'] == -1],  
            label='Sell Signal', marker='v', color='red', alpha=1  
        )  

        plt.title(f'{self.ticker} - Trading Signals')  
        plt.legend()  
        plt.show()  

if __name__ == "__main__":  
    # Example Usage  
    stock = OmniTradeAI("AAPL", "2023-01-01", "2024-01-01")  
    stock.generate_signals()  
    trades, final_value = stock.backtest()  
    print(f"Final Portfolio Value: ${final_value:.2f}")  
    stock.plot_signals()  