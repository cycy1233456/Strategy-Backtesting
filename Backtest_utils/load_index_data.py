import pandas as pd
import numpy as np
from pathlib import Path
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime

from data_utils import data_dirs, _get_timestamp_

def read_index_file(file_path:str, index, nrows=None):
    file_path = Path(file_path).joinpath(index+'.csv')
    if not file_path.exists():
        return pd.DataFrame()
    use_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume' ,'VWAP', 'Transaction']
    if not nrows:
        data = pd.read_csv(file_path, nrows=nrows, usecols=use_cols)
    else:
        data = pd.read_csv(file_path, usecols=use_cols)
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume' ,'VWAP', 'transaction']
    return data
# data = read_index_file(data_dirs['ETFS_second'], 'SPY', nrows=10)

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

def _calculate_twap_(group):
    return (group['close']).mean()
def _calculate_vwap_(group):
    if 'VWAP' in group.columns:
        return (group['VWAP'] * group['volume']).sum() / group['volume'].sum()
    else:
        return ((group['close']+group['open'])/2 * group['volume']).sum() / group['volume'].sum()


def determine_index_frequency(df):
    """
    确定 DataFrame 索引的时间频率。

    :param df: 包含 DatetimeIndex 的 DataFrame
    :return: 频率字符串，如 'S'（秒）、'T'（分钟）、'H'（小时）
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame 的索引必须是 DatetimeIndex")

    # 计算索引之间的时间差
    time_diffs = df.index.to_series().diff()

    # 统计时间差的唯一值及其出现次数
    diff_counts = time_diffs.value_counts()

    # 获取最常见的非空时间差
    most_common_diff = diff_counts.index[0]

    # 判断频率
    if most_common_diff == pd.Timedelta(seconds=1):
        return 'S'  # 秒频率
    elif most_common_diff == pd.Timedelta(minutes=1):
        return 'T'  # 分钟频率
    elif most_common_diff == pd.Timedelta(hours=1):
        return 'H'  # 小时频率
    else:
        return None  # 无法确定频率

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
    df_filtered['TWAP'] = df_filtered['TWAP'].fillna(df_filtered['close'].shift())
    return df_filtered

def remove_weekends_and_holidays(df):
    """
    去掉周末和节假日的数据
    :param df: 包含 DatetimeIndex 的 DataFrame
    :return: 过滤后的 DataFrame
    """
    us_cal = USFederalHolidayCalendar()
    holidays = us_cal.holidays(start=df.index.min(), end=df.index.max())
    df = df[~df.index.normalize().isin(holidays)]
    df = df[df.index.weekday < 5]  # 去掉周末
    return df

def get_time_period_resampled_data(df, resample_freq='5min', start_time="09:30", end_time="16:01"
                                   , time_adjust=True, padding=True, discrete_return=True):
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

    # 进行数据频率修改
    resampled_df = df.resample(resample_freq).agg({
        'volume': 'sum',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'transaction': 'sum',
    })
    resampled_df['VWAP'] = df.resample(resample_freq).apply(_calculate_vwap_)
    resampled_df['TWAP'] = df.resample(resample_freq).apply(_calculate_vwap_)

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
                 ,start_date='2020-01-01', end_date='2020-02-01', time_adjust=True, padding=True):
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
        self.index_data = pd.DataFrame()
        self.data_samples = None
        self.X_window, self.Y_window = X_window, Y_window
        self.X_train, self.Y_train = None, None

    def load_index_data(self, index):
        data = read_index_file(data_dirs['ETFS_second'], index=index,start_date=self.start_date, end_date=self.end_date)
        data = _get_timestamp_(data, time_column='date', time_format='second', set_time_as_index=False)
        self.index_data = get_time_period_resampled_data(data, resample_freq=self.resample_freq
                                                                 , time_adjust=self.time_adjust, padding=self.padding)

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
            for day, daily_df in self.index_data.groupby(self.index_data.index.date):
                X_daily, Y_daily = self.create_samples(daily_df, label=label)
                X_all.append(X_daily)
                Y_all.append(Y_daily)
        else:
            X_all, Y_all = self.create_samples(self.index_data)

        X_train = np.vstack(X_all)
        Y_train = np.vstack(Y_all)
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
    data = read_index_file(data_dirs['ETFS_second'], index='QQQ',start_date='2020-01-01', end_date='2020-02-01')
    data = _get_timestamp_(data, time_column='date', time_format='second', set_time_as_index=False)
    data = get_time_period_resampled_data(data, resample_freq='30S', time_adjust=True, padding=True)

    """ 示例用法 """

    Index_data = IndexData(root_path=data_dirs['ETFS_second'], resample_freq='5min', need_time_period='morning', X_window=10, Y_window=5
                           , time_adjust=True, padding=True)
    self = Index_data
    self.load_index_data(index='QQQ')
    self.load_data_sample()