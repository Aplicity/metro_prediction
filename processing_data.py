import pandas as pd
import numpy as np
import os
import time


# 读取数据
df_07 = pd.read_csv('./source_data/SPTCC-20160701.csv', encoding='gbk', header = None)
df_08 = pd.read_csv('./source_data/SPTCC-20160802.csv', encoding='gbk', header = None)
df_09 = pd.read_csv('./source_data/SPTCC-20160901.csv', encoding='gbk', header = None)
columns = ['卡ID', '刷卡日期' , '刷卡时间', '刷卡站点', '刷卡乘车类型', '刷卡扣钱', '是否优惠']
df_07.columns = columns
df_08.columns = columns
df_09.columns = columns

df = pd.concat([df_07,df_08,df_09]) # 把三个数据表连接在一起
del df_07,df_08,df_09
df = df.query("刷卡乘车类型 == '地铁'") # 只筛选地铁数据

# 特征工程
df['是否进站'] = df['刷卡扣钱'].apply(lambda x : 1 if x == 0 else 0) # 无扣钱算进站
df['DateTime'] = df['刷卡日期'] + " " + df["刷卡时间"] # 转化为完整的时间
df['DateTime'] = pd.to_datetime(df['DateTime'])  # 转换数据时间类型
df['月份'] = df["DateTime"].dt.month
df['日'] = df["DateTime"].dt.day
df['时'] = df["DateTime"].dt.hour
df['分'] = df["DateTime"].dt.minute
df['星期序数'] = df["DateTime"].dt.dayofweek # 周一为0、周二为1....
df['是否周末'] = df['星期序数'].apply(lambda x : 1 if x >=5 else 0) # 周六日则转换为1
df['第x个五分钟'] = df['分'].apply(lambda x : x // 5)
df['进出站状态'] = df['是否进站'].replace(0,-1)  # 进站为1，出站为-1
df.to_csv('./source_data/source_data.csv',index = None ) # 保存数据，方便后面使用


# 转换数据类型为字符串
df['月份'] = df['月份'].astype(str)
df['日'] = df['日'].astype(str)
df['时'] = df['时'].astype(str)
df['第x个五分钟'] = df['第x个五分钟'].astype(str)

# 统计每个站点、每个月份、每日、每小时、每5分钟的人流量
count_df = df.groupby(['刷卡站点','月份', '日', '时', '第x个五分钟' ])['进出站状态'].count().to_frame()
count_df.to_csv('./source_data/count_df.csv', encoding='gbk') # 把统计数据表保存到本地

