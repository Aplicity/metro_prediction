import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor
import tqdm
import warnings
from pyecharts import Bar , Timeline
warnings.filterwarnings('ignore')
import re
import math
import time
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
PI = math.pi
import json
import numpy as np

## 读取统计数据
df = pd.read_csv("./source_data/count_df.csv", encoding='gbk')
df = df.sort_values(by=['刷卡站点', '月份', '日', '时', '第x个五分钟'])

# 构建预测的输入数据 一天24小时，一小时12*5分钟
hours = []
minuteOfHours = []
for hour in range(24):
    for minuteOfHour in range(12):
        hours.append(hour)
        minuteOfHours.append(minuteOfHour)
test_df = pd.DataFrame({"时": hours, "第x个五分钟": minuteOfHours})

# 数据所有地铁站集合set
stations = set(df['刷卡站点'])

# 创建一个叫 KNN_output 的文件夹，用于存放模型预测数据
try:
    os.mkdir("KNN_output")
except:
    pass

# 对每个地铁站数据进行训练、预测
for station in tqdm.tqdm(stations):
    mask = df['刷卡站点'] == station
    temp_df = df[mask]
    X = temp_df[['时', '第x个五分钟']]
    y = temp_df['进出站状态']
    clf = KNeighborsRegressor() # 构建KNN模型
    clf.fit(X, y) # 模型训练
    test_df = pd.DataFrame({"时": hours, "第x个五分钟": minuteOfHours}) # 构建预测的输入数据 一天24小时，一小时12*5分钟
    y_pred = clf.predict(test_df) # 预测一天对客流量
    pred_df = test_df
    pred_df['客流量'] = y_pred # 添加预测数据
    pred_df.to_csv("./knn_output/{}.csv".format(station), index=None, encoding='gbk') # 把预测数据保存


## 对预测结果用pyecharts可视化
DFs = []
for fileName in os.listdir('./knn_output/'): # 对预测数据进行遍历
    temp_df = pd.read_csv('./knn_output/' + fileName, encoding='gbk')
    station = fileName.split('.')[0]
    temp_df['站点'] = station
    DFs.append(temp_df)
pred_df = pd.concat(DFs)
del DFs

timeline = Timeline(is_auto_play=True, timeline_bottom=0, width=1400, height=700)

for hour in range(24):
    for minuteOfHour in range(12):
        mask = (pred_df['时'] == hour) & (pred_df['第x个五分钟'] == minuteOfHour)
        temp_df = temp_df.sort_values('站点')
        temp_df = pred_df[mask]
        attrs = temp_df['站点']
        values = temp_df['客流量']

        bar = Bar("24小时上海地铁客流量", "以每5分钟为统计单位", width=1400, height=600)
        bar.add("", attrs, values)
        timeline.add(bar, '{}时 ：{}～{}分钟'.format(hour, 5 * minuteOfHour, 5 * minuteOfHour + 5))
timeline.render('KNN_24小时上海地铁客流量.html')

## 对预测结果用pyplot可视化
DFs = [] # 读取预测数据
for fileName in os.listdir('./knn_output/'):
    temp_df = pd.read_csv('./knn_output/' + fileName , encoding='gbk')
    station = fileName.split('.')[0]
    temp_df['站点'] = station
    DFs.append(temp_df)
pred_df = pd.concat(DFs)  # 把所有预测数据合并到一个数据表中
del DFs

def drop_num(station):
    '''
    把完整到站点名称删去"x号线"
    :param station: 站点名称
    :return: 修正后到站点名称
    '''
    re_line_num = re.findall("\d+号线",station )[0]
    n = len(re_line_num)
    return station[n:]

pred_df['站点'] = pred_df['站点'].apply(drop_num) # 把完整到站点名称删去"x号线"

# 以下是处理站点坐标位置的功能函数
def _transformlat(coordinates):
    lng = coordinates[:, 0] - 105
    lat = coordinates[:, 1] - 35
    ret = -100 + 2 * lng + 3 * lat + 0.2 * lat * lat + \
        0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
    ret += (20 * np.sin(6 * lng * PI) + 20 *
            np.sin(2 * lng * PI)) * 2 / 3
    ret += (20 * np.sin(lat * PI) + 40 *
            np.sin(lat / 3 * PI)) * 2 / 3
    ret += (160 * np.sin(lat / 12 * PI) + 320 *
            np.sin(lat * PI / 30.0)) * 2 / 3
    return ret


def _transformlng(coordinates):
    lng = coordinates[:, 0] - 105
    lat = coordinates[:, 1] - 35
    ret = 300 + lng + 2 * lat + 0.1 * lng * lng + \
        0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
    ret += (20 * np.sin(6 * lng * PI) + 20 *
            np.sin(2 * lng * PI)) * 2 / 3
    ret += (20 * np.sin(lng * PI) + 40 *
            np.sin(lng / 3 * PI)) * 2 / 3
    ret += (150 * np.sin(lng / 12 * PI) + 300 *
            np.sin(lng / 30 * PI)) * 2 / 3
    return ret


def gcj02_to_wgs84(coordinates):
    """
    GCJ-02转WGS-84
    :param coordinates: GCJ-02坐标系的经度和纬度的numpy数组
    :returns: WGS-84坐标系的经度和纬度的numpy数组
    """
    ee = 0.006693421622965943  # 偏心率平方
    a = 6378245  # 长半轴
    lng = coordinates[:, 0]
    lat = coordinates[:, 1]
    is_in_china = (lng > 73.66) & (lng < 135.05) & (lat > 3.86) & (lat < 53.55)
    _transform = coordinates[is_in_china]  # 只对不在国内的坐标做偏移

    dlat = _transformlat(_transform)
    dlng = _transformlng(_transform)
    radlat = _transform[:, 1] / 180 * PI
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * PI)
    mglat = _transform[:, 1] + dlat
    mglng = _transform[:, 0] + dlng
    coordinates[is_in_china] = np.array(
        [_transform[:, 0] * 2 - mglng, _transform[:, 1] * 2 - mglat]).T
    return coordinates


def bd09_to_gcj02(coordinates):
    """
    BD-09转GCJ-02
    :param coordinates: BD-09坐标系的经度和纬度的numpy数组
    :returns: GCJ-02坐标系的经度和纬度的numpy数组
    """
    x_pi = PI * 3000 / 180
    x = coordinates[:, 0] - 0.0065
    y = coordinates[:, 1] - 0.006
    z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * x_pi)
    theta = np.arctan2(y, x) - 0.000003 * np.cos(x * x_pi)
    lng = z * np.cos(theta)
    lat = z * np.sin(theta)
    coordinates = np.array([lng, lat]).T
    return coordinates


def bd09_to_wgs84(coordinates):
    """
    BD-09转WGS-84
    :param coordinates: BD-09坐标系的经度和纬度的numpy数组
    :returns: WGS-84坐标系的经度和纬度的numpy数组
    """
    return gcj02_to_wgs84(bd09_to_gcj02(coordinates))


def mercator_to_bd09(mercator):
    """
    墨卡托转BD-09
    :param coordinates: GCJ-02坐标系的经度和纬度的numpy数组
    :returns: WGS-84坐标系的经度和纬度的numpy数组
    """
    MCBAND = [12890594.86, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
    MC2LL = [[1.410526172116255e-08, 8.98305509648872e-06, -1.9939833816331, 200.9824383106796, -187.2403703815547,
              91.6087516669843, -23.38765649603339, 2.57121317296198, -0.03801003308653, 17337981.2],
             [-7.435856389565537e-09, 8.983055097726239e-06, -0.78625201886289, 96.32687599759846, -1.85204757529826,
              -59.36935905485877, 47.40033549296737, -16.50741931063887, 2.28786674699375, 10260144.86],
             [-3.030883460898826e-08, 8.98305509983578e-06, 0.30071316287616, 59.74293618442277, 7.357984074871,
              -25.38371002664745, 13.45380521110908, -3.29883767235584, 0.32710905363475, 6856817.37],
             [-1.981981304930552e-08, 8.983055099779535e-06, 0.03278182852591, 40.31678527705744, 0.65659298677277,
              -4.44255534477492, 0.85341911805263, 0.12923347998204, -0.04625736007561, 4482777.06],
             [3.09191371068437e-09, 8.983055096812155e-06, 6.995724062e-05, 23.10934304144901, -0.00023663490511,
              -0.6321817810242, -0.00663494467273, 0.03430082397953, -0.00466043876332, 2555164.4],
             [2.890871144776878e-09, 8.983055095805407e-06, -3.068298e-08, 7.47137025468032, -3.53937994e-06,
              -0.02145144861037, -1.234426596e-05, 0.00010322952773, -3.23890364e-06, 826088.5]]

    x = np.abs(mercator[:, 0])
    y = np.abs(mercator[:, 1])
    coef = np.array([MC2LL[index] for index in (
        np.tile(y.reshape((-1, 1)), (1, 6)) < MCBAND).sum(axis=1)])
    return converter(x, y, coef)


def converter(x, y, coef):
    x_temp = coef[:, 0] + coef[:, 1] * np.abs(x)
    x_n = np.abs(y) / coef[:, 9]
    y_temp = coef[:, 2] + coef[:, 3] * x_n + coef[:, 4] * x_n ** 2 + coef[:, 5] * x_n ** 3 + coef[:,
                                                                                                  6] * x_n ** 4 + coef[:,
                                                                                                                       7] * x_n ** 5 + coef[
        :,
        8] * x_n ** 6
    x[x < 0] = -1
    x[x >= 0] = 1
    y[y < 0] = -1
    y[y >= 0] = 1
    x_temp *= x
    y_temp *= y
    coordinates = np.array([x_temp, y_temp]).T
    return coordinates

# 加载上海地铁数据信息
with open("station_info_json.json", 'r') as fr:
    station_info_json = json.load(fr)

mapbox_access_token = "pk.eyJ1IjoibHVrYXNtYXJ0aW5lbGxpIiwiYSI6Im\
Npem85dmhwazAyajIyd284dGxhN2VxYnYifQ.HQCmyhEXZUTz3S98FMrVAQ"

# 定义figure
layout = go.Layout(autosize=True,
                   hovermode='closest',
                   mapbox=dict(
                       accesstoken=mapbox_access_token,
                       bearing=0,
                       center=dict( lat=31.234941, lon=121.47824),
                       pitch=0,
                       zoom=10 ),
                   )

def show_fig(hour , minuteOfHour):
    color = ('blue', 'green', 'yellow', 'purple', 'orange', 'red', 'violet',
                 'navy', 'crimson', 'cyan', 'magenta', 'maroon', 'peru', 'pink')  # 可按需增加

    data = []  # 绘制数据
    marked = set()
    cnt = 0
    for line in station_info_json['content']:
        uid = line['line_uid']
        if uid in marked:  # 由于线路包括了来回两个方向，需要排除已绘制线路的反向线路
            continue
        plots = []  # 站台墨卡托坐标
        plots_name = []  # 站台名称
        for plot in line['stops']:
            plots.append([plot['x'], plot['y']])
            plots_name.append(plot['name'])
        plot_mercator = np.array(plots)
        plot_coordinates = bd09_to_wgs84(mercator_to_bd09(plot_mercator))  # 站台经纬度

        plots_size = []
        for name in plots_name:
            mask = (pred_df["站点"] == name) & (pred_df["时"] == hour) & (pred_df["第x个五分钟"] == minuteOfHour)
            try:
                flow = pred_df[mask]['客流量'].values[0]
            except:
                flow = 0.1
            plots_size.append(flow / 10)
    #     print(line['line_name'])

        # 设置标记点的参数
        data.append(
            go.Scattermapbox(
                lon=plot_coordinates[:, 0],  # 站台经度
                lat=plot_coordinates[:, 1],  # 站台纬度
                mode='markers+lines',
                name=line['line_name'],  # 线路名称，显示在图例（legend）上
                text=plots_name,  # 各个点的名称，鼠标悬浮在点上时显示
                # 设置标记点的参数
                marker = go.scattermapbox.Marker(
                    size = plots_size,  # 这个用来控制站点绘制的大小
                    color=color[cnt] ),))
        marked.add(uid)  # 添加已绘制线路的uid
        marked.add(line['pair_line_uid'])  # 添加已绘制线路反向线路的uid
        cnt = (cnt + 1) % len(color)
    fig = dict(data=data, layout=layout)
    # py.iplot(fig)
    py.plot(fig, filename='Shanghai_{}时_第{}个5分.html'.format(hour , minuteOfHour))  # 生成html文件并打开

show_fig(7 , 0) # 可视化 7时- 0～5分（第0个5分）的预测数据


