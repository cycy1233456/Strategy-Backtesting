import numpy as np
import pandas as pd
import matplotlib as plt
import pandas_market_calendars as mcal
import datetime
import dateutil


class Context:
    def __init__(self, cash, start_date, end_date):
        self.start_date = start_date  # 开始日期，结束日期，账户现金
        self.end_date = end_date
        self.cash = cash
        self.cost = None  # 手续费设置
        self.positions = {}  # 持仓信息,字典
        self.benchmark = None  # 基准收益对比
        nyse = mcal.get_calendar('NYSE')
        self.schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        self.date_range = self.schedule

        # self.dt=datetime.datetime.strptime("%y-%m-%d",start_date)   #循环所在时间
        self.dt = dateutil.parser.parse(start_date)  # 转化为时间对象  ***需要从开始日期开始的后一个交易日

context = Context(1000000,'2019-01-03','2023-08-01')

# TODO:获取历史数据,count为往前天数回测时间

def get_index_data(index, start_date, end_date) -> pd.DataFrame:
    data = pro.query('daily', ts_code=index, start_date=start_date, end_date=end_date)
    # data = pd.DataFrame({"close":data.Data[0],"open":data.Data[1]},index=pd.to_datetime(data.Times)).dropna()
    return data


# security为股票代码，field为所需要列数据，count为交易天数
def attribute_history(security, count, fields=('open', 'close', 'high', 'low', 'volume')):
    # 获取最后一天 timedelta变化范围为一天
    end_date = (context.dt - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    # 结束日往前数count个交易日，到开始时间 timestamp转strftime设定类型
    start_date = (trade_cal[(trade_cal['is_open'] == 1) & (trade_cal['cal_date'] <= end_date)][-count:].iloc[0, :][
        'cal_date']).strftime('%Y-%m-%d')
    print(end_date, start_date)
    return attribute_daterange_history(security, start_date, end_date, fields)


def attribute_month_history(security, count, fields=('open', 'close', 'high', 'low', 'volume')):
    end_date = (context.dt - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (trade_cal[(trade_cal['is_open'] == 1) & (trade_cal['cal_date'] <= end_date)][-count:].iloc[0, :][
        'cal_date']).strftime('%Y-%m-%d')
    print(end_date, start_date)
    return attribute_daterange_history(security, start_date, end_date, fields)


# TODO:取值 从网页，文件获取历史数据
def attribute_daterange_history(security, start_date, end_date, fields=('open', 'close', 'high', 'low', 'volume')):
    df = get_index_data(security, start_date, end_date)  # 若本地没有，则连接tushare
    return df  # 返回所需列


# 选取一只股票作为基准
def set_benchmark(security):
    context.benchmark = security


# 初始化函数，设定基准
def initialize(context):
    pass


# 股票池确定
stock_pool = pro.stock_basic(exchange='SSE', list_status='L',
                             fields='ts_code,symbol,name,industry,list_date,delist_date')


def get_used_stocks(stock_pool):
    used_or_not = (stock_pool["list_date"] <= "2010") | (stock_pool["delist_date"] == None)
    return stock_pool[used_or_not == True]["ts_code"].values


stock_pool = get_used_stocks(stock_pool)

trade_cal = pro.query('trade_cal', start_date='20100101', end_date='20231231')
trade_cal.index = pd.to_datetime(trade_cal["cal_date"])
adjust_days = trade_cal[trade_cal["is_open"] == 1].resample("Q").first()["cal_date"].values


# 设定每日行为
def handle_data(context):
    # 每个季度第一个交易日
    adjust_or_not = context.dt.strftime('%Y%m%d') in adjust_days
    stock_scores = pd.DataFrame(index=stock_pool, columns=["F_score"])

    if adjust_or_not:
        for stock in stock_pool:
            stock_scores.loc[stock, "F_score"] = get_stock_F_Score(context, stock)

        used_stocks = stock_scores.query("F_score == {}".format(stock_scores["F_score"].max())).index
        used_stocks_num = len(used_stocks)
        current_positions = list(context.positions.keys())

        if len(context.positions) == 0:
            pass
        else:
            for stock in current_positions:
                if stock not in used_stocks:
                    order_target_value(stock, 0)
        for stock in used_stocks:
            order_target_value(stock, context.cash / used_stocks_num)


    else:
        pass


# 最大回撤，传入净值时间序列
def max_drawdown(df, column):
    max_drawdown = 0
    peak = df['value'][0]
    for net_worth in df['value'][1:]:
        if net_worth > peak:
            peak = net_worth
        drawdown = peak - net_worth
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    max_drawdown_ratio = max_drawdown / peak
    print('最大回撤：{}'.format(max_drawdown))
    print('最大回撤率：{}'.format(max_drawdown_ratio))
    calmar_ratio(df, max_drawdown_ratio, column)


def calmar_ratio(df, max_drawdown_ratio, column):
    calmar_ratio = df[column].mean() / max_drawdown_ratio
    print('calmar ratio:', calmar_ratio)


def sharpe_ratio(df, column):
    sharpe_ratio = df[column].mean() / df[column].std()
    print('夏普比率：', sharpe_ratio)
    # return sharpe_ratio


def statistic(df, column):
    max_drawdown(df, column)
    sharpe_ratio(df, column)


# 用户初始化
initialize(context)


# TODO:模拟每天的运行
# Dataframe记录每个交易日的收益，收益率
def run():
    # 初始账户价值
    init_value = context.cash

    # index_date = pd.to_datetime(context.date_range).strftime('%Y-%m-%d')
    plt_df = pd.DataFrame(columns=['value'])  # 用于记录账户价值，索引为时间，有账户价值和基准价值
    # print(plt_df)

    last_price = {}
    # 每日执行一次
    value = 0
    for dt in tqdm.tqdm(pd.to_datetime(context.date_range).strftime('%Y%m%d')):
        dt = dateutil.parser.parse(dt)
        print(f"Processing data for date: {dt}")
        context.dt = dt  # 变成时间对象传入context

        handle_data(context)  # 用户操作

        # 今日操作后，收盘时账户价值
        context.cash = float(context.cash)

        # 遍历所有持仓查询今日信息
        for stock in context.positions:
            today_data = get_today_data(stock)
            # 考虑停牌情况，此时get_today_data会返回空值,价格按照停牌前一交易日的价格
            if len(today_data) == 0:
                p = last_price[stock]
            else:
                p = today_data['close']  # 收盘价计算今日总价
                last_price[stock] = p

            value = value + float(p) * float(context.positions.get(stock, 0))  # 账户价值即为股票价值与现金之和

        # 在dt索引到当前交易日，在账户价值一列中添加今日资产价值
        plt_df.loc[dt, 'value'] = float(value)
        # print(context.positions)

    # 与初始价格算收益率
    plt_df['revenue'] = (plt_df['value'] - init_value) / init_value
    plt_df['R'] = plt_df['value'] / init_value

    ##基准历史初始价格，计算收益率
    ##添加基准列
    # bm_df = attribute_daterange_history(context.benchmark, context.start_date, context.end_date)   #获取基准的历史数据
    # bm_df.index = context.date_range   #修改为同一index
    # init_benchmark = bm_df['open'][0]
    #
    ##benchmark_r = pd.DataFrame(benchmark_r,index=pd.to_datetime(context.date_range).strftime('%Y-%m-%d'))
    #
    # plt_df['benchmark_ratio'] = (bm_df['open'] - init_benchmark) / init_benchmark
    # plt_df['benchmark_R'] = bm_df['open'] / init_benchmark
    ##两者净值收益率作图
    # plt_df[['revenue','benchmark_ratio']].plot()
    return plt_df

    # print(plt_df)


# print(context.benchmark)
# 主函数运行
df = run()

# 量化回测指标
statistic(df, 'revenue')