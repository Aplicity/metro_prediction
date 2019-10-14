import requests
import json
import time

mapbox_access_token = "pk.eyJ1IjoibHVrYXNtYXJ0aW5lbGxpIiwiYSI6Im\
Npem85dmhwazAyajIyd284dGxhN2VxYnYifQ.HQCmyhEXZUTz3S98FMrVAQ"

null = None  # 将json中的null定义为None
city_code = 289  # 上海的城市编号
station_info = requests.get('http://map.baidu.com/?qt=bsi&c=%s&t=%s' % (city_code, int(time.time() * 1000)))
station_info_json = eval(station_info.content)  # 将json字符串转为python对象

with open("station_info_json.json", 'w') as fr:
    json.dump(station_info_json, fr)

