import pandas as pd
import numpy as np
from pathlib import Path
import pytz
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor


file_path = 'E:/minute_aggs_v1/2024/11/2024-11-14.csv' #
# data = pd.read_csv(file_path)

# Define UTC and US/Eastern time zones
utc = pytz.utc
eastern = pytz.timezone('US/Eastern')
# data['timestamp'] = pd.to_datetime(data['window_start'], unit='ns', utc=True).dt.tz_convert(eastern)
# check data completeness
# data.groupby('ticker').count()


# 数据和保存路径
main_data_dir = Path('E:/minute_aggs_v1/')
output_dir = Path('E:/minute_continuous/')
temp_dir = Path('E:/minute_temp/')
output_dir.mkdir(parents=True, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)


def save_monthly_data_to_temp(year, month):
    """分阶段保存每个月的数据到临时文件"""
    data_dir = main_data_dir / f'{year:04d}/{month:02d}/'
    if not data_dir.exists():
        return

    csv_files = sorted(data_dir.glob('*.csv'))
    for file in csv_files:
        data = pd.read_csv(file)
        data['window_start'] = pd.to_datetime(data['window_start'], unit='ns', utc=True)

        # 按股票代码分组并保存
        for ticker, df in data.groupby('ticker'):
            temp_file = temp_dir / f'{ticker}_{year}.csv'
            if temp_file.exists():
                df.to_csv(temp_file, mode='a', header=False, index=False)
            else:
                df.to_csv(temp_file, index=False)


def merge_temp_files_to_yearly(year):
    """将临时文件合并为年度文件"""
    temp_files = sorted(temp_dir.glob(f'*_{year}.csv'))
    for temp_file in temp_files:
        ticker = temp_file.stem.split('_')[0]  # 获取股票代码
        yearly_file = output_dir / f'{ticker}_{year}.csv'

        # 读取并按时间排序后保存
        df = pd.read_csv(temp_file)
        df['window_start'] = pd.to_datetime(df['window_start'])
        df = df.sort_values(by='window_start')
        df.to_csv(yearly_file, index=False)

        # 删除临时文件
        temp_file.unlink()


# 主流程：按年处理所有数据
def process_yearly_data(year):
    print(f"Processing data for year: {year}")
    # 阶段 1：逐月保存临时数据
    for month in range(1, 13):
        save_monthly_data_to_temp(year, month)

    # 阶段 2：合并临时文件为年度文件
    merge_temp_files_to_yearly(year)
    print(f"Completed data processing for year: {year}")


# 示例：处理 2023 年的数据
process_yearly_data(2023)
