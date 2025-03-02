# 单品种的时序因子测试
# 日频交易策略；日内交易策略
import pandas as pd
import numpy as np

# 每个index代表这个时间窗口结束 9:01代表 9:00-9:01
data.set_index('Date', inplace=True)
data.index = data.index + pd.Timedelta(minutes=1)

# 定义一个单品种择时策略，即在不同时间点上有买入卖出的一条仓位序列
def MA_Strategy(data, win_long, win_short, loss_ratio=999, base_price='Close'):
    """
    基于长短期均线的交易策略，支持部分持仓
    参数:
        pdatas: DataFrame，包含日期（DateTime）、收盘价（CLOSE）等列
        win_long: 长期均线窗口
        win_short: 短期均线窗口
        lossratio: 止损率, 默认为 999 表示不止损
    返回:
        stats: 策略整体表现
        result_peryear: 每年表现
        transactions: 交易记录
        pdatas: 带有交易标记和净值计算的原始数据
        position_df: 持仓记录，时间为索引
    """
    data = data.copy()
    #
    data = data.between_time("9:00", "16:00")
    # 计算长短均线
    lma = data[base_price].rolling(window=win_long, min_periods=1).mean()
    sma = data[base_price].rolling(window=win_short, min_periods=1).mean()

    # 定义买入和卖出信号  这个信号是在一个时间窗口结束时发出的 作用于下一个时间窗口  因此仓位的改变从下一个时间窗口开始
    buy_signal = ((sma > lma) & (sma.shift(1) <= lma.shift(1))).shift(1).fillna(False)
    sell_signal = ((sma < lma) & (sma.shift(1) >= lma.shift(1))).shift(1).fillna(False)

    # 初始化持仓状态
    data['position'] = 0

    # 计算持仓变化：买入为 +1，卖出为 -1
    data['position_change'] = 0
    data.loc[buy_signal, 'position_change'] = 1
    data.loc[sell_signal, 'position_change'] = -1

    # 计算累计持仓
    data['position'] = data['position_change'].replace(0, np.NAN)
    data['position'] = data['position'].fillna(method='ffill').fillna(0)  # 防止持仓变成负值

    # 创建持仓 DataFrame
    position_df = data[['position']]

    return position_df


# 简单回测方法
def calculate_metrics(data, position_df, fee_rate=0.0001):
    """
    计算收益率序列、价值曲线及策略相关评估指标
    参数:
        data: 原始价格数据，包含列 ['Date', 'CLOSE']
        position_df: 持仓记录，时间为索引，包含列 ['position']
        fee_rate: 每次交易的手续费比例，默认为 0.001 (0.1%)
    返回:
        results: 字典，包含换手率、平均持仓时间、超额收益的 Sharpe 比率、Calmar 比率、最大回撤
        nav_curve: 策略的价值曲线
        benchmark_curve: 基准价值曲线
    """
    # 确保索引对齐
    data = data.sort_index()
    position_df = position_df.sort_index()

    # 确保数据与持仓对齐
    merged = data
    merged['position'] = position_df['position']

    # 收益率计算
    merged['ret'] = merged['Close'].pct_change().fillna(0)
    merged['strategy_ret'] = merged['position'] * np.log(merged['VWAP'] / merged['VWAP'].shift(1))

    # 交易信号和手续费
    merged['trade'] = merged['position'].diff().abs()  # 计算换手率
    merged['transaction_cost'] = merged['trade'] * fee_rate
    merged['strategy_ret_net'] = merged['strategy_ret'] - merged['transaction_cost']

    # 策略净值曲线
    merged['nav'] = (1 + merged['strategy_ret_net']).cumprod()
    merged['benchmark_nav'] = (1 + merged['ret']).cumprod()

    # 评估指标计算
    excess_ret = merged['strategy_ret_net'] - merged['ret']  # 超额收益
    sharpe = excess_ret.mean() / excess_ret.std(ddof=1) * np.sqrt(252) if excess_ret.std() > 0 else 0  # 年化 Sharpe 比率
    max_drawdown = ((merged['nav'] / merged['nav'].cummax()) - 1).min()  # 最大回撤
    calmar = (merged['nav'].iloc[-1] ** (252 / len(merged)) - 1) / abs(
        max_drawdown) if max_drawdown < 0 else 0  # Calmar 比率
    turnover = merged['trade'].sum() / len(merged)  # 平均换手率
    avg_holding_time = (merged['position'] > 0).sum() / max(merged['trade'].sum(), 1)  # 平均持仓时间

    # 结果汇总
    results = {
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'turnover_rate': turnover,
        'average_holding_time': avg_holding_time,
    }

    return results, merged['nav'].iloc[-1], merged['benchmark_nav'].iloc[-1]

for win_short in [30,60,90,120,150,180]:
    win_long = win_short * 5

    position_df = MA_Strategy(data, win_long, win_short)
    position_df.replace(-1, 0)
    print(calculate_metrics(data, position_df, fee_rate=0.000))
