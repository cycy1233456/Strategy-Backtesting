import numpy as np
import pandas as pd
import matplotlib as plt
import tqdm
import Backtest_utils.data_utils
import seaborn as sns
from scipy.stats import spearmanr

from Backtest_utils.Factors_Construct import future_return


class FactorAnalysis:
    def __init__(self, bin_return_df):
        """
        初始化因子分析类
        :param bin_return_df: DataFrame，行是时间，列是不同因子分组的收益率
        """
        self.bin_return_df = bin_return_df

    def calculate_ic(self, factor_values):
        """
        快速计算 IC（信息系数）
        :param factor_values: DataFrame，行是时间，列是不同分组对应的因子值
        :return: Series，时间序列的 IC 值
        """
        # 确保索引对齐
        common_index = self.bin_return_df.index.intersection(factor_values.index)
        returns = future_return.loc[common_index]
        factor_values = factor_values.loc[common_index]

        # 按行计算 Spearman Rank Correlation
        ic_values = returns.corrwith(factor_values, axis=1, method=lambda x, y: spearmanr(x, y)[0])

        # 转换为 Series
        ic_series = pd.Series(ic_values, index=common_index, name="IC")
        return ic_series

    def calculate_rank_ic(self, factor_value):
        """
        快速计算 Rank IC（排序信息系数）
        :param factor_values: DataFrame，行是时间，列是不同分组对应的因子值
        :return: Series，时间序列的 Rank IC 值
        """
        # 对因子值和收益率进行排序后再计算 IC
        ranked_returns = future_return.rank(axis=1, pct=True)
        ranked_factors = factor_value.rank(axis=1, pct=True)

        # 计算排序 IC
        rank_ic_values = ranked_returns.corrwith(ranked_factors, axis=1, method=lambda x, y: spearmanr(x, y)[0])

        # 转换为 Series
        rank_ic_series = pd.Series(rank_ic_values, index=self.bin_return_df.index, name="Rank IC")
        return rank_ic_series.mean()


    def calculate_group_rank_ic(self, factor_value):
        """
        快速计算 Rank IC（排序信息系数）
        :param factor_values: DataFrame，行是时间，列是不同分组对应的因子值
        :return: Series，时间序列的 Rank IC 值
        """
        # 对因子值和收益率进行排序后再计算 IC
        ranked_returns = self.bin_return_df.rank(axis=1, pct=True)
        ranked_factors = self.group_factor_mean.rank(axis=1, pct=True)

        # 计算排序 IC
        rank_ic_values = ranked_returns.corrwith(ranked_factors, axis=1, method=lambda x, y: spearmanr(x, y)[0])

        # 转换为 Series
        rank_ic_series = pd.Series(rank_ic_values, index=self.bin_return_df.index, name="Rank IC")
        return rank_ic_series.mean()


    def calculate_long_short(self, compound=False):
        """
        计算多空组合的日收益率和价值曲线 单利复利 主单利
        :return: DataFrame，包含多空组合的日收益率和价值曲线
        """
        long_short_return = self.bin_return_df.iloc[:, 0] - self.bin_return_df.iloc[:, -1]  # 最多组 - 最少组
        if compound:
            long_short_nav = (1 + long_short_return).cumprod()  # 累积净值
        else:
            long_short_nav = 1 + long_short_return.cumsum()
        return pd.DataFrame({"Return": long_short_return, "NAV": long_short_nav})

    @staticmethod
    def max_drawdown(df, column):
        """
        计算最大回撤和最大回撤率
        :param df: DataFrame，包含净值时间序列
        :param column: str，分析的列名
        :return: 最大回撤值，最大回撤率
        """
        drawdown = df[column].cummax() - df[column]
        drawdown_ratio = drawdown / (drawdown + df[column])
        print(f"最大回撤：{drawdown.max()}")
        print(f"最大回撤率：{drawdown_ratio.max():.2%}")
        return drawdown_ratio.max()

    @staticmethod
    def calmar_ratio(df, max_drawdown_ratio, column):
        """
        计算 Calmar 比率
        :param df: DataFrame，包含净值时间序列
        :param max_drawdown_ratio: 最大回撤率
        :param column: str，分析的列名
        :return: Calmar 比率
        """
        annual_return = pow(1+df[column].sum(), (252 / df.shape[0]) / 7)-1  # 假设每日收益率的均值
        calmar_ratio = annual_return / max_drawdown_ratio
        print(f"Calmar 比率：{calmar_ratio:.2f}")
        return calmar_ratio

    @staticmethod
    def sharpe_ratio(df, column):
        """
        计算夏普比率
        :param df: DataFrame，包含净值或收益时间序列
        :param column: str，分析的列名
        :return: 夏普比率
        """
        annual_return = pow(1+df[column].sum(), (252 / df.shape[0]) / 7)-1   # 年化收益率
        annual_volatility = df[column].std() * np.sqrt(252)  # 年化波动率
        sharpe_ratio = annual_return / annual_volatility
        print(f"夏普比率：{sharpe_ratio:.2f}")
        return sharpe_ratio


    def calculate_turnover(self, holdings):
        """
        计算换手率
        :param holdings: DataFrame，行是时间，列是分组持仓比例
        :return: 换手率
        """
        turnover = holdings.diff().abs().sum(axis=1) / 2  # 每日持仓变动总和的一半
        avg_turnover = turnover.mean()
        print(f"平均换手率: {avg_turnover:.2%}")
        return turnover

    def calculate_average_holding_period(self, holdings):
        """
        计算平均持仓时间
        :param holdings: DataFrame，行是时间，列是分组持仓比例
        :return: 平均持仓时间
        """
        holding_days = (holdings != 0).astype(int).cumsum()  # 计算累计持仓天数
        avg_holding_period = holding_days.mean(axis=0)
        print("平均持仓时间（天）：")
        print(avg_holding_period)
        return avg_holding_period


    def statistic(self, df, column):
        """
        统计多空组合的关键指标，包括最大回撤、夏普比率和 Calmar 比率
        :param df: DataFrame，包含净值时间序列
        :param column: str，分析的列名
        """
        print(f"\n统计分析 - {column}:")
        max_dd, max_dd_ratio = self.max_drawdown(df, column)
        calmar = self.calmar_ratio(df, max_dd_ratio, column)
        sharpe = self.sharpe_ratio(df, column)
        return {"Max Drawdown": max_dd, "Max Drawdown Ratio": max_dd_ratio,
                "Calmar Ratio": calmar, "Sharpe Ratio": sharpe}

    def analyze_long_short(self):
        """
        分析多空组合的价值曲线及统计指标
        """
        long_short_df = self.calculate_long_short()

        # 绘制价值曲线
        plt.figure(figsize=(10, 6))
        plt.plot(long_short_df["NAV"], label="Long-Short NAV")
        plt.title("Long-Short Portfolio Net Asset Value")
        plt.xlabel("Date")
        plt.ylabel("Net Asset Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 打印统计指标
        stats = self.statistic(long_short_df, "NAV")
        return stats


