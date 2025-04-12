import pandas as pd
import numpy as np
from pathlib import Path

from data_utils import (SP500_tickers, data_dirs, _get_timestamp_, fetch_tickers_data_multithread, load_minute_data_df)



class TickerData:
    def __init__(self, tickers:list, start_date, end_date, start_time, end_time, resample_freq):
        self.tickers = self.check_tickers(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.start_time = start_time
        self.end_time = end_time
        self.resample_freq = resample_freq
        self.tickers_data = None

    def check_tickers(self, tickers):
        return [ticker for ticker in tickers if ticker in SP500_tickers]

    def load_tickers_data(self):
        self.tickers_data = load_minute_data_df(tickers=self.tickers
                                               , start_date=self.start_date
                                               , end_date=self.end_date
                                               , start_time=self.start_time
                                               , end_time=self.end_time
                                               , resample_freq=self.resample_freq)

    def create_samples(self, df, label='return'):
        X_window = self.X_window  # 过去 10 条数据作为输入
        Y_window = self.Y_window
        X_samples, Y_samples = [], []

        features_list = df.columns.tolist()
        for i in range(len(df) - X_window - Y_window + 1):
            X_samples.append(df.iloc[i:i + X_window].values)  # 过去 X_window 作为输入
            Y_samples.append(df.iloc[i + X_window: i + X_window + Y_window][label].values)  # 未来 Y_window 作为输出
        return np.array(X_samples), np.array(Y_samples)


    def load_data_sample(self, label='return', overnight_return=False):
        """
        选择需要的数据样本长度 X-->Y
        label: 预测的项目 'close' 'return' or others
        :return:
        """
        if not overnight_return:
            #
            X_all, Y_all = [], []
            for ticker in self.tickers:
                ticker_df = self.tickers_data[self.tickers_data['ticker'] == ticker]

                for day, daily_df in ticker_df.groupby(ticker_df.index.date):
                    X_daily, Y_daily = self.create_samples(daily_df, label=label)
                    X_all.append(X_daily)
                    Y_all.append(Y_daily)
        else:
            X_all, Y_all = [], []
            for ticker in self.tickers:
                ticker_df = self.tickers_data[self.tickers_data['ticker'] == ticker]
                X_ticker, Y_ticker = self.create_samples(ticker_df)
                X_all.append(X_ticker)
                Y_all.append(Y_ticker)

        X_train = np.vstack(X_all)
        Y_train = np.vstack(Y_all)
        self.X_train, self.Y_train = X_train, Y_train

        return X_train, Y_train

if __name__ == '__main__':
    start_date, end_date = '2020-01-01', '2020-02-01'
    data = fetch_tickers_data_multithread(tickers=['AAPL', 'MSFT', 'TSLA'], start_date=start_date, end_date=end_date, frequency='minute')
    data = _get_timestamp_(data)