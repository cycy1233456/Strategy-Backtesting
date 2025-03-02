import os
import gzip
import shutil
from botocore.config import Config
import boto3

# Initialize a session using your credentials
session = boto3.Session(
   aws_access_key_id='baecf22d-da28-4fa1-9f6f-ade187bce92d',
   aws_secret_access_key='75yK0LRV7d5rFsSUOg0dvChWr5mJzT4x',
)

# Create a client with your session and specify the endpoint
s3 = session.client(
   's3',
   endpoint_url='https://files.polygon.io',
   config=Config(signature_version='s3v4'),
)

def download_and_extract_day_aggs(bucket_name, prefix, local_base_dir, start_year=2020):

    """
    下载和解压 S3 中从指定年份开始的文件至本地。

    参数:
        bucket_name (str): S3 Bucket 名称。
        prefix (str): S3 上的文件路径前缀。
        local_base_dir (str): 本地存储文件的根目录。
        start_year (int): 起始年份，默认从 2020 开始。
        endpoint_url (str): 自定义 S3 端点 URL。
        aws_access_key (str): AWS Access Key。
        aws_secret_key (str): AWS Secret Key。
    """
    # 创建本地存储根目录
    os.makedirs(local_base_dir, exist_ok=True)

    # 分页器列出所有对象
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            # 获取文件路径
            object_key = obj['Key']

            # 筛选符合年份条件的文件
            try:
                year = int(object_key.split('/')[2])
            except (IndexError, ValueError):
                continue
            if year < start_year or not object_key.endswith('.gz'):  # 跳过非目标年份和非 .gz 文件
                continue

            # 构造本地存储路径
            relative_path = os.path.relpath(object_key, prefix)  # 相对路径
            local_gz_path = os.path.join(local_base_dir, relative_path)
            local_csv_path = local_gz_path[:-3]  # 去掉 ".gz"

            # 创建必要的文件夹
            os.makedirs(os.path.dirname(local_gz_path), exist_ok=True)

            # 下载文件
            print(f"Downloading: {object_key} -> {local_gz_path}")
            s3.download_file(bucket_name, object_key, local_gz_path)

            # 解压缩文件
            print(f"Extracting: {local_gz_path} -> {local_csv_path}")
            with gzip.open(local_gz_path, 'rb') as gz_file:
                with open(local_csv_path, 'wb') as csv_file:
                    shutil.copyfileobj(gz_file, csv_file)

            # 删除原始 .gz 文件（可选）
            os.remove(local_gz_path)
            print(f"Completed: {local_csv_path}")


# 使用示例
bucket_name = 'flatfiles'
prefix = 'us_stocks_sip/day_aggs_v1/'  # 文件前缀
local_base_dir = 'E:/day_aggs_v1/'  # 本地存储根目录
start_year = 2020  # 起始年份

# 调用函数
download_and_extract_day_aggs(bucket_name, prefix, local_base_dir, start_year)


