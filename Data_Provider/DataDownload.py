from polygon import RESTClient
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

client = RESTClient(api_key="75yK0LRV7d5rFsSUOg0dvChWr5mJzT4x")

class StockDataProcessor:
    def __init__(self, api_key):
        """
        Initialize the StockDataProcessor with an API key.

        Parameters:
        api_key (str): Your API key for accessing stock data.
        """
        self.api_key = api_key
        self.client = RESTClient(api_key)  # 初始化REST客户端

    def fetch_stock_data(self, ticker, date_start, date_end, interval="minute", multiplier=1):
        """
        Fetch high-frequency stock data (e.g., 1-minute or 15-minute intervals) for a single ticker.

        Parameters:
        ticker (str): Stock ticker symbol.
        date_start (str): Start date in "YYYY-MM-DD" format.
        date_end (str): End date in "YYYY-MM-DD" format.
        interval (str): Data interval (e.g., "minute").
        multiplier (int): Multiplier for the interval (e.g., 15 for 15-minute data).

        Returns:
        pd.DataFrame: DataFrame containing stock data with columns Date, Open, High, Low, Close, Volume, VWAP, Trades.
        """
        aggs = []

        # 使用分页获取数据
        for a in self.client.list_aggs(
                ticker,
                multiplier,
                interval,
                date_start,
                date_end,
                limit=50000  # 设置每页的上限，Polygon API会自动分页
        ):
            aggs.append(a)

        # 检查是否有数据返回
        if aggs:
            df = pd.DataFrame([{
                'Date': pd.to_datetime(a.timestamp, unit='ms'),
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume,
                'VWAP': a.vwap,  # 确保 VWAP 存在
                'Transaction': a.transactions  # 确保 Trades 存在
            } for a in aggs])
            return df
        else:
            print(f"No results found for {ticker} in the given date range.")
            return pd.DataFrame()

    def fetch_data_multithreaded(self, ticker, date_start, date_end, interval="second", multiplier=1, chunk_size=30):
        """
        Fetch data using multithreading by splitting the date range into smaller chunks.

        Parameters:
        ticker (str): Stock ticker symbol.
        date_start (str): Start date in "YYYY-MM-DD" format.
        date_end (str): End date in "YYYY-MM-DD" format.
        interval (str): Data interval (e.g., "second").
        multiplier (int): Multiplier for the interval.
        chunk_size (int): Number of days per chunk for each thread to handle.

        Returns:
        pd.DataFrame: Concatenated DataFrame of all chunks.
        """
        # Convert start and end dates to datetime objects
        start_date = datetime.strptime(date_start, "%Y-%m-%d")
        end_date = datetime.strptime(date_end, "%Y-%m-%d")

        # Generate date ranges for each chunk
        date_ranges = []
        while start_date < end_date:
            chunk_end_date = min(start_date + timedelta(days=chunk_size), end_date)
            date_ranges.append((start_date.strftime("%Y-%m-%d"), chunk_end_date.strftime("%Y-%m-%d")))
            start_date = chunk_end_date + timedelta(days=1)

        # Use ThreadPoolExecutor to fetch data for each chunk in parallel
        data_frames = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.fetch_stock_data, ticker, start, end, interval, multiplier)
                for start, end in date_ranges
            ]

            # Collect results as they complete
            for future in as_completed(futures):
                data_frames.append(future.result())

        # Concatenate all data chunks into one DataFrame
        return pd.concat(data_frames, ignore_index=True).sort_values(by='Date').reset_index(drop=True)


    def fetch_and_process_stocks(self, tickers, date_start, date_end):
        """
        Fetch and process stock data for multiple tickers.

        Parameters:
        tickers (list): List of stock ticker symbols.
        date_start (str): Start date in "YYYY-MM-DD" format.
        date_end (str): End date in "YYYY-MM-DD" format.

        Returns:
        dict: A dictionary with stock tickers as keys and processed DataFrames as values.
        """
        stock_data = {}

        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            df = self.fetch_stock_data(ticker, date_start, date_end)

            if not df.empty:
                calculator = FactorCalculator(df)
                result_df = calculator.all_factors()
                stock_data[ticker] = result_df
                print(f"Data for {ticker} fetched and processed successfully.")
            else:
                print(f"Failed to fetch data for {ticker} because dataframe is empty.")

        return stock_data
stock_processor = StockDataProcessor(api_key="75yK0LRV7d5rFsSUOg0dvChWr5mJzT4x")
data = stock_processor.fetch_stock_data('SPY', '2019-01-01', '2024-11-11', interval='second', multiplier=1)