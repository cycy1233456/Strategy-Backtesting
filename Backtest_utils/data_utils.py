import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor



# from Data_Processor.minute_agg_process import file_path
from tickers_lists import SP500_tickers
import warnings
warnings.filterwarnings('ignore')

"""
数据调用函数 ************ 对于股票数据 *********  包含多种方法
*******指数数据 在load_index_data.py********
调取历史数据的参数：
@params:
ticker:str / tickers:list
start_date / end_date or (start_date, time_range) (end_date, time_range) :

need_period: 日内时段
need_columns: 需要的数据条
frequency: 频率(minute, day) 
agg: True/False 可以选择将多个tickers的数据合并为一个df 如果need_columns为单一 则输出一个df, columns为ticker
padding: 是否将所有的分钟条数补全
"""

# file_path = 'E:/minute_aggs_v1/2024/11/2024-11-14.csv'  #
# data = pd.read_csv(file_path)

data_dirs = {'minute': "E:/minute_aggs_v1", 'day': "E:/day_aggs_v1", 'SP500':'E:/SP500_minute', 'ETFS_second':'E:/Second/Indices/2020-2024'}



def read_file(file_path, tickers):
    """
    读取单个文件并过滤指定 tickers。

    参数:
        file_path (Path): 文件路径。
        tickers (list of str): 要提取的股票代码。

    返回:
        DataFrame: 过滤后的数据。
    """
    if not file_path.exists():
        return pd.DataFrame()  # 返回空 DataFrame

    # 读取数据
    data = pd.read_csv(file_path)
    if tickers is None:
        return data
    elif isinstance(tickers, list):
        # 过滤指定的 tickers
        return_data = data[data['ticker'].isin(tickers)]
        missing_tickers = set(tickers) - set(return_data['ticker'].unique())
        # if missing_tickers != set():
        #     pass
        #     print(f"这些 tickers 没有被找到: {missing_tickers}")
        return return_data
    elif isinstance(tickers, str):
        return data[data['ticker'].isin(list(tickers))]
    else:
        print("given tickers error")
        return pd.DataFrame()


# 基础文件阅读方式
def fetch_tickers_data_multithread(tickers, start_date, end_date, frequency='minute', max_threads=8):
    """
    使用多线程从存储的CSV文件中提取指定时间段和tickers的数据。

    参数:
        tickers (list of str): 要提取的股票代码。
        start_date (str): 起始日期，格式 "YYYY-MM-DD"。
        end_date (str): 结束日期，格式 "YYYY-MM-DD"。
        frequency (str): 数据频率，支持 "minute" 或 "day"。
        max_threads (int): 最大线程数。

    返回:
        DataFrame: 包含指定 tickers 和时间段内的数据，按时间排序。
    """
    # 转换日期为 Datetime 格式
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # 构建所有目标文件路径
    base_dir = data_dirs.get(frequency, None)
    if not base_dir:
        raise ValueError(f"频率 {frequency} 未定义存储路径，请检查配置。")

    file_paths = [
        Path(base_dir) / f"{current_date.year:04d}/{current_date.month:02d}/{current_date.strftime('%Y-%m-%d')}.csv"
        for current_date in pd.date_range(start=start_date, end=end_date, freq='D')
    ]

    # 使用多线程读取文件
    all_data = []
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(read_file, file_path, tickers): file_path for file_path in file_paths}
        for future in futures:
            try:
                data = future.result()
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                print(f"读取文件出错: {futures[future]}, 错误: {e}")

    # 合并所有数据
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.sort_values(by='window_start', inplace=True)  # 根据时间排序
    else:
        result = pd.DataFrame()  # 如果没有数据，返回空 DataFrame
    result.sort_values(['ticker', 'window_start'], inplace=True)
    return result


# 加载分钟频数据 sample
# minute_data = fetch_tickers_data_multithread(tickers, start_date, end_date, frequency='minute')

# 加载日频数据
# daily_data = fetch_tickers_data_multithread(tickers, start_date, end_date, frequency='day')


# 基于基础方式构建
# 分钟数据 --> 选定时段
def _get_timestamp_(df, time_column='window_start', time_format='minute', set_time_as_index=False):
    """
    在df 中添加时间列 Datetime 精确到秒 （美东纽约交易所时间）
    判断时间格式 nanosecond/ ms 找到对应的时间
    如果是Minute 将数据的时间格 + 1 让所有时间戳对应的为前向时间 stamp_t := (t-1min, t] 前开后闭区间
    :param df:
    :param time_column: 时间戳所在列
    :param time_format:
    :return:
    """
    df = df.copy()
    if len(str(df.loc[0, time_column])) == 19:
        time_period = 'ns'
    elif len(str(df.loc[0, time_column])) == 13:
        time_period = 'ms'
        # 从纳秒转换为 UTC 时间
        df['Datetime'] = (
            pd.to_datetime(df[time_column], unit=time_period)
            .dt.tz_localize('UTC')  # 明确时间为 UTC
            .dt.tz_convert('America/New_York')
            .dt.tz_localize(None)
        )
    if time_column != 'Datetime':
        df.drop(columns=[time_column], inplace=True)

    # 输出的时间列为 datetime 时间调整**
    if time_format == 'minute':
        df['Datetime'] = df['Datetime'] + pd.Timedelta(minutes=1)
    elif time_format == 'second':
        df['Datetime'] = df['Datetime'] + pd.Timedelta(seconds=1)
    elif time_format == 'day':
        df['Datetime'] = df['Datetime'].dt.date

    if set_time_as_index:
        df.set_index('Datetime')
    return df


def _get_time_period_data_(df, start_time="09:00", end_time="16:01", padding=True):
    """
    选取给定时间段的数据 并决定是否需要padding
    :param df: 股票数据 / 无法处理指数数据
    :param start_time: 开始时间 default "09:00"
    :param end_time: 结束时间 default "16:01"
    :param padding: 选择是否将缺失的时间窗口补上 9:00, 9:03 --> 9:00, 9:01, 9:02, 9:03
    :return:
    """
    df = df.copy()
    start_time = pd.to_datetime(start_time).time()
    end_time = pd.to_datetime(end_time).time()
    df_filtered = df[
        (df['Datetime'].dt.time >= start_time) &
        (df['Datetime'].dt.time <= end_time)
        ]

    df_filtered.set_index('Datetime', inplace=True)
    trading_days = df_filtered.index.normalize().unique()
    # 生成时间序列，包含所有分钟
    # 确保填充范围内的时间只包含交易日
    full_time_range = pd.DatetimeIndex(
        [time for day in trading_days for time in pd.date_range(
            start=day + pd.Timedelta(hours=start_time.hour),
            end=day + pd.Timedelta(hours=end_time.hour),
            freq='min'
        )]
    )

    if padding:
        df_padded = pd.DataFrame()
        for ticker, df_filled in df_filtered.groupby('ticker'):
            # 重新索引并填充缺失的时间点
            df_filled = df_filled.reindex(full_time_range, method=None)
            # 填充缺失数据
            # 填充OHLC为前一个时间点的close，交易量为0
            df_filled['close'] = df_filled['close'].ffill()  # close 仍然用前一个有效值填充
            df_filled[['ticker', 'window_start']] = df_filled[['ticker', 'window_start']].ffill()
            df_filled['open'] = df_filled['open'].fillna(df_filled['close'].shift())
            df_filled['high'] = df_filled['high'].fillna(df_filled['close'].shift())
            df_filled['low'] = df_filled['low'].fillna(df_filled['close'].shift())
            df_filled['volume'] = df_filled['volume'].fillna(0)  # 填充交易量为0
            df_filled['transactions'] = df_filled['transactions'].fillna(0)

            df_padded = pd.concat([df_padded, df_filled.between_time(start_time, end_time)])
    else:
        df_padded = df_filtered

    return df_padded


def _calculate_twap(group: pd.core.groupby.generic.DataFrameGroupBy, base_price='close'):
    """
    计算 Time-Weighted Average Price (TWAP)
    TWAP = sum(price * duration) / sum(duration)
    """
    return group[base_price].mean()


def _calculate_vwap(group):
    """
    计算 Volume-Weighted Average Price (VWAP)
    """
    volume_sum = group['volume'].sum()
    if volume_sum == 0:
        return np.nan  # 如果成交量为零，返回 NaN 或者 0
    return np.sum((group['close']+group['open'])/2 * group['volume']) / volume_sum


def _resample_data_(df, resample_freq:'str', time_adjust=True):
    """
    目前输入的df index ：t代表 (t-1, t]的时间条
    resample过程中 会以传入的第一个时间节点 t_start 为开始 向后对 t_start + resample_freq的时间进行 aggregate
    输出的df时间index为 t_start（但实际上是从[t_start-1, t_start-1+freq]的聚合） 如果修改后则变为 从[t, t+freq]的聚合
    example: resample_freq=60min time_point=16:00
            -->if adjusted: (15:00, 16:00]  --> if not adjust: (14:59, 15:59]
    :param df: index
    :param resample_freq:
    :return: resampled_data with ohlc TWAP VWAP
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return 'ERROR: NOT DATETIME INDEX'
    df = df.copy()
    if time_adjust:
        origin = 'end'
    else:
        origin = 'start'
    # df = df.set_index('Datetime')
    resampled_df = df.groupby('ticker').resample(resample_freq, origin=origin).agg({
        'volume': 'sum',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'window_start': 'first',
        'transactions': 'sum'
    })

    # 计算 TWAP 和 VWAP
    twap_column = df.groupby('ticker').resample(resample_freq, origin=origin).apply(_calculate_twap)
    # 用start为标签的情况 直接返回一个multiindex 如果用end的话需要用一个stack来变成multi index
    twap_column = twap_column if origin=='start' else twap_column.stack()
    vwap_column = df.groupby('ticker').resample(resample_freq, origin=origin).apply(_calculate_vwap)
    vwap_column = vwap_column if origin == 'start' else vwap_column.stack()

    resampled_df['TWAP'] = twap_column
    resampled_df['VWAP'] = vwap_column

    # 重置索引
    resampled_df.reset_index(inplace=True)
    # 去掉 NaN 数据
    resampled_df.dropna(axis=0, inplace=True)
    resampled_df.set_index('level_1', inplace=True)
    return resampled_df


def load_minute_data_df(tickers:list, start_date, end_date, start_time, end_time, resample_freq):
    """
    复合(主): 读取分钟函数 可以指定每天的多少时间内的数据 并通过resample返还
    :param tickers: tickers_pool
    :param start_date: str
    :param end_date: str
    :param start_time: str
    :param end_time: str
    :param resample_freq: str(x)+'min' x为需要的分钟间隔时间
    :return: 从start_date-end_date下每日的start_time-end_time中不间断数据（通过padding）并resample为需要的时间
    """
    # 读取原始文件
    minute_data = fetch_tickers_data_multithread(tickers=tickers, start_date=start_date, end_date=end_date,
                                                 frequency='minute')
    # 增加一列时间
    minute_data = _get_timestamp_(minute_data)
    # 提取一个固定时间段的每日数据
    need_minute_data = _get_time_period_data_(df=minute_data, start_time=start_time, end_time=end_time, padding=True)
    if resample_freq is None:
        return minute_data
    else:
        resampled_minute_data = _resample_data_(need_minute_data, resample_freq=resample_freq)
        return resampled_minute_data


def load_minute_data_point(tickers:list, date_point, minute_point, resample_freq: str):
    """
    单一时间点 数据条 可用与交易下单时给定时间点
    :param tickers:
    :param date_point: 指定日
    :param minute_point: 指定时间点 分钟点
    :param resample_freq: 给定了窗口时间 可以算未来多少min的VWAP和 TWAP
    :return:
    """
    start_date, end_date = date_point, date_point
    start_time = minute_point
    if resample_freq.endswith('min'):
        minutes_to_add = max(60, int(resample_freq[0:-3]))
        end_time = (pd.to_datetime(minute_point, format='%H:%M') + pd.Timedelta(minutes=minutes_to_add)).strftime('%H:%M')

    elif resample_freq.endswith('T'):
        minutes_to_add = max(60, int(resample_freq[0:-1]))
        end_time = (pd.to_datetime(minute_point, format='%H:%M') + pd.Timedelta(minutes=minutes_to_add)).strftime('%H:%M')

    else:
        end_time = minute_point

    minute_data = load_minute_data_df(tickers=tickers, start_date=start_date, end_date=end_date
                                      , start_time=start_time, end_time=end_time, resample_freq=resample_freq)
    minute_data = minute_data.between_time(minute_point, minute_point)
    return minute_data


def load_daily_data_point(start_date, end_date):
    #TODO 构建一个可以调取每日收盘时的VWAP的函数 或者TWAP 或者每日任何一个时间点上的 VWAP 相比上一个单个时间点的扩展
    # load_minute_data_df(start_date, end_date, start_time, end_time, resample_freq)
    pass


def load_period_minute_data(tickers:list, start_date, end_date, resample_freq: str, period:str):
    trade_periods = {
        'inner_day': ('09:00', '16:01'),  # Regular market hours (e.g., NYSE hours)
        'pre_market': ('04:00', '09:00'),  # Pre-market trading hours
        'after_market': ('16:00', '20:00'),  # After-market trading hours
        'overnight': ('20:00', '04:00'),  # Overnight trading (for continuous markets like crypto)
        'full_day': ('00:00', '23:59'),  # Whole day (24-hour period)
        'early_trade': ('06:00', '09:00'),  # Early trading window
        'late_trade': ('16:00', '18:00')  # Late trading window
    }
    try:
        start_time, end_time = trade_periods[period]
    except KeyError:
        start_time, end_time = trade_periods['inner_day']
    minute_data = load_minute_data_df(tickers, start_date, end_date, start_time, end_time, resample_freq)
    return minute_data


def agg_tickers_minute_data(need_column:str, tickers, start_date, end_date, start_time, end_time, resample_freq: str):
    """
    :param need_column: 'close' 'open' 'VWAP'
    :return: df(index=datetime, columns=tickers) 数值为need_column
    """
    minute_data = load_minute_data_df(tickers, start_date, end_date, start_time, end_time, resample_freq)
    return pd.DataFrame({ticker: ticker_df[need_column] for ticker, ticker_df in minute_data.groupby('ticker')})


def load_daily_data(tickers, start_date, end_date):
    daily_data = fetch_tickers_data_multithread(tickers=tickers, start_date=start_date, end_date=end_date, frequency='day')
    daily_data.dropna(inplace=True)
    # ADD COLUMN 'Datetime' choose format of day
    daily_data = _get_timestamp_(daily_data, time_format='day')
    return daily_data


def load_current_tickers(start_date, end_date)->pd.DataFrame:
    """
    输出给定时间段内都有哪些tickers 如果当日有则为1 没有则为nan
    :param start_date:
    :param end_date:
    :return: df[column=tickers, index=date] 如果当日有数据则为1 否则为nan
    """
    daily_data = load_daily_data(tickers=None, start_date=start_date, end_date=end_date)
    current_tickers = pd.DataFrame(columns=daily_data['ticker'].drop_duplicates()
                                   , index=daily_data['Datetime'].unique())
    for trade_date in current_tickers.index:
        current_tickers.loc[trade_date, daily_data[daily_data['Datetime']==trade_date]['ticker']] = 1
    return current_tickers


def load_useful_tickers(start_date, end_date, given_tickers=None):
    """
    去掉那些在一开始有数值后面变成 NaN 的列，同时保留那些一开始是 NaN 但后面有数值的列
    :param start_date:
    :param end_date:
    :param given_tickers: 可以选定一些 tickers来进行操作 将Pool限定在这些里面
    :return: df
    """
    current_tickers = load_current_tickers(start_date=start_date, end_date=end_date)

    if isinstance(given_tickers, list):
        current_tickers = current_tickers.loc[:, given_tickers]
    # 标记需要删除的列
    cols_to_drop = []

    for col in current_tickers.columns[1:]:  # 从第二列开始检查
        # 检查该列的首个非NaN值是否出现在第一个位置
        if current_tickers[col].iloc[0] is not None and current_tickers[col].isna().any():  # 只有在开头有值且后续有NaN时
            cols_to_drop.append(col)

    # 删除那些符合条件的列
    current_tickers = current_tickers.drop(columns=cols_to_drop)
    return current_tickers


def agg_tickers_daily_data(tickers, start_date, end_date, need_columns:str):
    """
    可以将所有的tickers的某一个数据并在一起
    if need columns为list且为多个，则输出一个dict{column:df}
    :param need_columns: 'close' 'open' 'high'  'close'
    :return: df(index=datetime, columns=tickers) 数值为need_column
    """
    daily_data = load_daily_data(tickers, start_date, end_date)
    daily_data.set_index('Datetime', inplace=True)
    if isinstance(need_columns, list) and len(need_columns) == 1:
        agg_data = pd.DataFrame({ticker: ticker_df[need_columns[0]]
                                 for ticker, ticker_df in daily_data.groupby('ticker')})
    elif isinstance(need_columns, str):
        agg_data = pd.DataFrame({ticker: ticker_df[need_columns]
                                 for ticker, ticker_df in daily_data.groupby('ticker')})
    elif isinstance(need_columns, list) and len(need_columns) > 1:
        agg_data = {}
        for need_column in need_columns:
            agg_data[need_column] = pd.DataFrame({ticker: ticker_df[need_columns]
                                 for ticker, ticker_df in daily_data.groupby('ticker')})
    else:
        print('not find the columns')
        agg_data = pd.DataFrame()

    return agg_data


def read_sp500_minute_data(start_date, end_date, load=True):
    """
    从本地中读取SP500的连续时间数据
    :param start_date:
    :param end_date:
    :param load: 从本地读取
    :return:
    """
    if load:
        minute_data = pd.DataFrame()
        for year in range(int(start_date[0:4]), int(end_date[0:4])):
            file_path = data_dirs['SP500'] + '/' + year + '.csv'
            minute_data = pd.concat([minute_data, pd.read_csv(file_path)])
        minute_data.sort_values('ticker', inplace=True)
        minute_data = minute_data[(minute_data['Datetime']>=pd.to_datetime(start_date))
                                                & (minute_data['Datetime']<=pd.to_datetime(end_date))]
        minute_data.set_index('Datetime', inplace=True)

    else:
        minute_data = load_minute_data_df(tickers=tickers, start_date=start_date, end_date=end_date
                                          , start_time='9:00', end_time='16:00', resample_freq=None)
        # minute_data.to_csv(file_path)
    return minute_data


def point_to_point_return(tickers, start_date, end_date, pattern, opening_point, cleaning_point, minute_data
                          , accuracy='close', period_length=10, future=1):
    """
    C2C :获取每只股票的每日收益率 每个日期 t 对应的是 close_t+1/ close_1 -1 即今天收盘到明天的收益率
    P2P : Point to Point 可以指定时间点来进行收益率的计算 需要用到分钟数据
    :param tickers: 股票池子
    :param start_date: 起始日期
    :param end_date:
    :param pattern: 'C2C' 'P2P'
    :param opening_point: 开仓时间点
    :param cleaning_point: 清仓时间点
    :param accuracy: 需要以 VWAP T分钟的均价来计算开仓价格？
    :param accuracy: 需要以 T分钟的均价来计算价格
    :param future: 需要未来几天的收益率 future = X means close_t+X/ close_t
    :return: 单只股票收益率时间序列 其中 index t 对应的数据是 ：从t时间点开仓 到t+future清仓的收益率 用log计算
            'C2C': {'close_df':close_df, 'return_df':future_return}
            'P2P':{'open_df':opening_data, 'clean_df':cleaning_data, 'return_df':future_return}
    """
    if pattern == 'C2C':
        # 在收盘时开仓
        if accuracy == 'close':
            # 调取每日收盘价信息
            close_df = agg_tickers_daily_data(
                tickers=tickers, start_date=start_date, end_date=end_date, need_columns='close')
        elif accuracy == 'VWAP' or accuracy == 'TWAP':
            # 从分钟连续总表中提取时间点 从16:00 - 16:00+period_length
            depend_point = f'16:{period_length}'
            # 参照时间是5分钟
            opening_point = (pd.to_datetime(depend_point) - pd.Timedelta(minutes=period_length-1)).strftime('%H:%M')
            # 在给定了minute_data下 直接查找需要的位置
            need_data = minute_data.between_time(opening_point, depend_point)
            if accuracy == 'TWAP':
                # 计算开仓价格
                close_df = need_data.groupby('ticker').resample('D')['close'].mean().dropna()
                close_df = close_df.dropna().unstack().T
            else:
                # 开仓的point
                close_df = need_data.groupby('ticker').resample('D').apply(lambda x:(x['close']*x['volume']).sum()/(x['volume'].sum()))
                close_df = close_df.dropna().unstack().T

        else:
            print('error pattern --> simple C2C')
            close_df = agg_tickers_daily_data(
                tickers=tickers, start_date=start_date, end_date=end_date, need_columns='close')

        if isinstance(close_df, pd.DataFrame):
            future_return = np.log(close_df.shift(-future)/close_df)
            return {'close_df':close_df, 'return_df':future_return}

    elif pattern == 'P2P':
        # 如果选择了点对点 需要两个df 分别记录两个时间点
        if accuracy == 'close':
            time_index_data = minute_data.set_index('Datetime')
            opening_data = time_index_data.between_time(opening_point, opening_point)
            cleaning_data = time_index_data.between_time(cleaning_point, cleaning_point)
            opening_data = opening_data.groupby('ticker').resample('D')['close'].last().dropna().unstack().T
            cleaning_data = cleaning_data.groupby('ticker').resample('D')['close'].first().dropna().unstack().T
            future_return = np.log(cleaning_data.shift(-future) / opening_data)

        elif accuracy == 'VWAP' or accuracy == 'TWAP':
            # 给定时间点 如16:00 则需要的时间段为从16:00:00-16:09:59
            opening_point = (pd.to_datetime(opening_point) + pd.Timedelta(minutes=1)).strftime('%H:%M')
            open_depend_point = (pd.to_datetime(opening_point) + pd.Timedelta(minutes=period_length-1)).strftime('%H:%M')
            open_need_data = minute_data.between_time(opening_point, open_depend_point)

            cleaning_point = (pd.to_datetime(cleaning_point) + pd.Timedelta(minutes=1)).strftime('%H:%M')
            clean_depend_point = (pd.to_datetime(cleaning_point) + pd.Timedelta(minutes=period_length-1)).strftime('%H:%M')
            clean_need_data = minute_data.between_time(cleaning_point, clean_depend_point)

            if accuracy == 'TWAP':
                # 计算开仓价格
                opening_data = open_need_data.groupby('ticker').resample('D')['close'].mean().dropna()
                opening_data = opening_data.dropna().unstack().T
                # 计算平仓价格
                cleaning_data = clean_need_data.groupby('ticker').resample('D')['close'].mean().dropna()
                cleaning_data = cleaning_data.dropna().unstack().T
                future_return = np.log(cleaning_data.shift(-future) / opening_data)
            else:
                # 开仓的point
                close_df = need_data.groupby('ticker').resample('D').apply(lambda x:(x['close']*x['volume']).sum()/(x['volume'].sum()))
                close_df = close_df.dropna().unstack().T

        return {'open_df':opening_data, 'clean_df':cleaning_data, 'return_df':future_return}

    # 一个df(columns = tickers, index=Date) 里面的数可以是close 或者是 VWAP定义
    # 如果不提供日频agg数据 则自行查询



if __name__ == '__main__':
    tickers = SP500_tickers # 指定的股票代码
    start_date = "2021-01-01"
    end_date = "2021-12-31"
    data = read_index_file(data_dirs['ETFS_second'], index='QQQ', nrows=10000)
    # close_df = agg_tickers_daily_data(SP500_tickers, start_date, end_date, 'close')
