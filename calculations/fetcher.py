
from project_paths import project_file
import yfinance as yf
import pandas as pd

NSE_STOCKS = [
    # IT
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    # Finance
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    # Healthcare
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
    # Energy
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS",
    # Industrials
    "LT.NS", "SIEMENS.NS", "ABB.NS", "BEL.NS", "HAL.NS"
]

raw_data = yf.download(
    tickers=NSE_STOCKS,
    start="2019-01-01",
    end="2025-12-31",
    progress=True
)

if "Adj Close" in raw_data.columns:
    prices = raw_data["Adj Close"].copy()
else:
    prices = raw_data["Close"].copy()

cleaned_prices = prices.ffill().bfill()

output_file = project_file("nse_cleaned_prices.csv")
cleaned_prices.to_csv(output_file)

print(f"\nDone! Cleaned price data saved to '{output_file}'.")
print(f"Total Trading Days: {cleaned_prices.shape[0]}")
print(f"Total Stocks: {cleaned_prices.shape[1]}")
print("\nSample Preview:")
print(cleaned_prices.iloc[:5, :5])  # Show first 5 rows and 5 stocks