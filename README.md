# metro_prediction
上海地铁预测可视化

## 需求说明
对上海地铁各个站点对客流量进行预测，并进行可视化

## 文件说明
* source_data -- 存放源数据以及清洗整理完的数据（数据较大，请移步百度云下载：链接:https://pan.baidu.com/s/1RPJgffHIDJxl8rIzYqPaeQ  密码:59e2）
  - SPTCC-20160901.csv ： 2016年09月01日公共交通数据（源数据）
  - SPTCC-20160802.csv ： 2016年08月02日公共交通数据（源数据）
  - SPTCC-20160701.csv ： 2016年07月01日公共交通数据（源数据）
  - source_data.csv ： 上面3个数据合并后，并经过清洗后的数据（代码输出文件）
  - count_df.csv ： 以每个站点、每个时间段为分组，统计5分钟内的客流量（代码输出文件）
  
* data -- 存放9月1日各个站点各个时间段的客流量（代码输出文件）
  - 2号线中山公园.csv
  - 16号线新场.csv
  - ...
  - ...
  - 10号线上海动物园.csv
  
* KNN_output -- 存放基于KNN输出的预测客流量（代码输出文件）
  - 2号线中山公园.csv
  - 16号线新场.csv
  - ...
  - ...
  - 10号线上海动物园.csv

* LSTM_output -- 存放基于LSTM输出的预测客流量（代码输出文件）
  - 2号线中山公园.csv
  - 16号线新场.csv
  - ...
  - ...
  - 10号线上海动物园.csv
  
 * station_info_json.json ：上海地铁信息，主要为线路名称、站点坐标（代码输出文件）
  
以下为代码文件，请按顺序执行
* processing_data.py -- 合并源数据并清洗，执行后生成 source_data/source_data.csv和source_data/count_df.csv文件
* get_shanghai_station_info.py -- 通过爬虫获取上海地铁信息，以json格式保存到本地，即station_info_json.json
* predict_for_KNN.py -- 基于KNN预测一天24小时*12（每5分钟）个时间段的客流量并可视化，输出 KNN_output
* predict_for_LSTM.py -- 基于LSTM预测一天24小时*12（每5分钟）个时间段的客流量并可视化，输出 LSTM_output

以下文件为可视化输出
* KNN_24小时上海地铁客流量.html -- 基于pyecharts生成含时间条的柱状图（模型KNN）
* Shanghai_7时_第0个5分.html -- 基于pyplot生成07:00～07:05的各站点Scattermapbox（模型KNN）
* LSTM_24小时上海地铁客流量.html -- 基于pyecharts生成含时间条的柱状图（模型LSTM）
* LSTM_Shanghai_7时_第0个5分.html -- 基于pyplot生成07:00～07:05的各站点Scattermapbox（模型LSTM）

* images
 - pyechart_show.png -- 基于pyecharts可视化截图
 - pyplot_show.png -- 基于pyplot可视化截图

## 效果展示
### pyecharts版本
![image](https://github.com/Aplicity/metro_prediction/blob/master/images/pyechart_show.png)

### pygplot版本
![image](https://github.com/Aplicity/metro_prediction/blob/master/images/pyplot_show.png)

## Note
* 源数据为公共交通数据，包括公交等，如果只研究地铁方面，需要单独筛选。
* 如想可视化其他城市，则修改get_shanghai_station_info.py中的城市代码，全国各个城市对应的代码见《BaiduMap_cityCode.txt》
