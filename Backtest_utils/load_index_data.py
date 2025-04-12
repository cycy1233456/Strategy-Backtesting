import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from Backtest_utils.data_utils import (data_dirs, _get_timestamp_, _calculate_vwap_, _calculate_twap_,
                                _calculate_skew_, _calculate_var_, _calculate_kurtosis_, _calculate_down_std_share_,
                                _calculate_flow_in_ratio_, remove_weekends_and_holidays)

def read_index_file(file_path: str, index, start_date, end_date):
    file_path = Path(file_path).joinpath(index + '.csv')
    if not file_path.exists():
        return pd.DataFrame()

    use_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Transaction']

    # 将 start_date 和 end_date 转换为时间戳
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000) + 86399999

    # 使用 chunksize 分块读取数据
    chunksize = 10000  # 根据实际情况调整 chunksize
    chunks = []

    for chunk in pd.read_csv(file_path, usecols=use_cols, chunksize=chunksize):
        chunk = chunk[(chunk['Date'] >= start_timestamp) & (chunk['Date'] <= end_timestamp)]
        chunks.append(chunk)

    # 合并所有块
    data = pd.concat(chunks)

    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'VWAP', 'transaction']
    return data


def _padding_data_(df_filtered):
    # 填充缺失数据
    # 填充OHLC为前一个时间点的close，交易量为0
    df_filtered['close'] = df_filtered['close'].ffill()  # close 仍然用前一个有效值填充
    df_filtered['open'] = df_filtered['open'].fillna(df_filtered['close'].shift())
    df_filtered['high'] = df_filtered['high'].fillna(df_filtered['close'].shift())
    df_filtered['low'] = df_filtered['low'].fillna(df_filtered['close'].shift())
    df_filtered['volume'] = df_filtered['volume'].fillna(0)  # 填充交易量为0
    df_filtered['transaction'] = df_filtered['transaction'].fillna(0)
    df_filtered['VWAP'] = df_filtered['VWAP'].fillna(df_filtered['close'].shift())
    # df_filtered['TWAP'] = df_filtered['TWAP'].fillna(df_filtered['close'].shift())
    return df_filtered

def calculate_factors(df, resample_freq, **kwargs):
    # kwargs = {
    #     'VWAP': _calculate_vwap_,
    #     'TWAP': _calculate_twap_,
    # }
    factor_value = {}
    for factor_name, method in kwargs.items():
        factor_value[factor_name] = df.resample(resample_freq).apply(method)
    return pd.DataFrame(factor_value)

def get_time_period_resampled_data(df, resample_freq='5min', start_time="09:30", end_time="16:01"
                                   , time_adjust=True, padding=True, discrete_return=True, extra_factors=True):
    """
    更改数据的频率 可选 X * 秒 分钟 天
    :param df:
    :param resample_freq: X* 秒'S' / 分钟 ’min' / 天 'D'
    :param time_adjust:
    :return:
    """
    df = df.copy().set_index('Datetime')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 的索引必须是 DatetimeIndex")

    df['return'] = df['close'].pct_change()
    # 进行数据频率修改
    resampled_df = df.resample(resample_freq).agg({
        'volume': 'sum',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'transaction': 'sum',
    })
    if extra_factors:
        factor_values = calculate_factors(
            df,
            resample_freq='5min',
            VWAP = _calculate_vwap_,
            TWAP = _calculate_twap_,
            skew=_calculate_skew_,
            kurtosis=_calculate_kurtosis_,
            down_std_share=_calculate_down_std_share_,
            flow_in_ratio=_calculate_flow_in_ratio_,
            variance=_calculate_var_
        )
    else:
        factor_values = calculate_factors(
            df,
            resample_freq='5min',
            VWAP=_calculate_vwap_,
            TWAP=_calculate_twap_)

    resampled_df = pd.concat([resampled_df, factor_values], axis=1)

    # 选取需要的时间段
    resampled_df = resampled_df.between_time(start_time, end_time)
    # 去掉周末和节假日的数据
    resampled_df = remove_weekends_and_holidays(resampled_df)

    # 是否需要补全数据条
    if padding:
        resampled_df = _padding_data_(resampled_df)

    if time_adjust:
        # 每天的时间都进行调整
        resampled_df = resampled_df.groupby(resampled_df.index.date).shift(1).dropna()

    if discrete_return:
        resampled_df['return'] = resampled_df['close'] / resampled_df['close'].shift(1) - 1
        # 离散数据情况用的收益率每天的第一条和最后一条调整 非连续收益--不用隔夜收益率
        adjust_rows = pd.concat([resampled_df.groupby(resampled_df.index.date).head(1)
                                    , resampled_df.groupby(resampled_df.index.date).tail(1)]).sort_index()
        adjust_rows['return'] = adjust_rows['close'] / resampled_df['open'] - 1
        resampled_df.loc[adjust_rows.index, 'return'] = adjust_rows['return']
    else:
        # 非离散 用隔夜收益率
        resampled_df['return'] = resampled_df['close'] / resampled_df['close'].shift(1) - 1

    return resampled_df


def get_different_time_period_data(time_period='morning'):
    """
    需要的时间阶段 可以有早盘 盘前盘后 盘前加盘后 所有交易时间
    :param df:
    :param time_period:
    :return:
    """
    if time_period == 'morning':
        start_time = "09:30"
        end_time = "16:00"
    elif time_period == 'pre_market':
        start_time = "04:00"
        end_time = "09:30"
    elif time_period == 'after_market':
        start_time = "16:00"
        end_time = "20:00"
    elif time_period == 'all_trading':
        start_time = "04:00"
        end_time = "20:00"
    else:
        raise ValueError("Invalid time_period. Choose from 'morning', 'pre_market', 'after_market', 'pre_and_after_market', 'all_trading'")
    return start_time, end_time


class IndexData:
    def __init__(self, root_path, resample_freq='5min', need_time_period='morning', X_window=10, Y_window=5
                 ,start_date='2020-01-01', end_date='2020-02-01', time_adjust=True, padding=True, extra_factors=True):
        """
        数据处理集
        :param root_path: 指数csv路径
        :param resample_freq: 需要的时间
        :param need_time_period: 需要的时间片段
        :param X_window: 使用的 预测 X 的长度
        :param Y_window: 需要的预测 Y 长度
        :param time_adjust: 时间调整 前开后闭
        :param padding: 无交易区间数据条填充
        """
        self.index_pools = ['QQQ', 'SPY', 'DIA', 'GLD', 'IBIT']
        self.root_path = root_path  # 'E:/Second/Indices/2020-2024'
        self.start_date, self.end_date = start_date, end_date
        self.start_time, self.end_time = get_different_time_period_data(need_time_period)

        self.resample_freq = resample_freq
        self.time_adjust = time_adjust
        self.padding = padding
        self.extra_factors = extra_factors
        self.raw_data = pd.DataFrame()
        self.index_data = pd.DataFrame()
        self.data_samples = None
        self.X_window, self.Y_window = X_window, Y_window
        self.X_train, self.Y_train = None, None

    def load_index_data(self, index):
        self.raw_data = read_index_file(self.root_path, index=index,start_date=self.start_date, end_date=self.end_date)
        self.index_data = _get_timestamp_(self.raw_data, time_column='date', time_format='second', set_time_as_index=False)
        self.index_data = get_time_period_resampled_data(self.index_data, resample_freq=self.resample_freq
                                , time_adjust=self.time_adjust, padding=self.padding, extra_factors=self.extra_factors)

    def create_samples(self, df, label='return'):
        X_window = self.X_window  # 过去 X 条数据作为输入
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
            for day, daily_df in self.index_data.groupby(self.index_data.index.date):
                X_daily, Y_daily = self.create_samples(daily_df, label=label)
                X_all.append(X_daily)
                Y_all.append(Y_daily)
            X_train = np.vstack(X_all)
            Y_train = np.vstack(Y_all)
            self.X_train, self.Y_train = X_train, Y_train
        else:
            X_train, Y_train = self.create_samples(self.index_data)
            self.X_train, self.Y_train = X_train, Y_train

        return X_train, Y_train

    def describe_dataset(self):
        """
        描述数据集
        """
        print("### 数据集描述 ###")
        print(f"数据集名称: {self.index_pools}")
        print(f"数据集路径: {self.root_path}")
        print(f"时间范围: {self.start_date} 到 {self.end_date}")
        print(f"时间频率: {self.resample_freq}")
        print(f"时间调整: {self.time_adjust}")
        print(f"填充缺失值: {self.padding}")
        print(f"X_window: {self.X_window}")
        print(f"Y_window: {self.Y_window}")
        print(f"开始时间: {self.start_time}")
        print(f"结束时间: {self.end_time}\n")



if __name__ == '__main__':
    # data = read_index_file(data_dirs['ETFS_second'], index='QQQ',start_date='2020-01-01', end_date='2020-02-01')
    # data = _get_timestamp_(data, time_column='date', time_format='second', set_time_as_index=False)
    # data = get_time_period_resampled_data(data, resample_freq='30S', time_adjust=True, padding=True)

    """ 示例用法 """

    Index_data = IndexData(root_path=data_dirs['ETFS_second'], resample_freq='5min', need_time_period='morning', X_window=10, Y_window=5
                           , time_adjust=True, padding=True, start_date='2020-01-01', end_date='2020-01-10')
    SP_stock_data = IndexData(root_path=data_dirs['Stocks_second'], resample_freq='5min', need_time_period='morning', X_window=10, Y_window=5
                           , time_adjust=True, padding=True, start_date='2020-01-01', end_date='2024-01-10')
    # STOCK DATA SAMPLE ONLY FOR SP500 SECOND
    STOCK_DATA = SP_stock_data
    STOCK_DATA.load_index_data(index='AAPL')
    # index data sample
    self = Index_data
    self.load_index_data(index='QQQ')
    self.resample_freq = '1S'
    self.load_data_sample()