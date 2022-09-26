#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 下午9:41
# @Author  : firstelfin
# @File    : json_str2excel.py


import json
import pandas as pd

data = {
    "name": [10],
    "labels": [json.dumps({"test": "train", "中国": "世界第一"})]
}

data = pd.DataFrame(data)
data.to_csv("../oildetect.csv", header=False, index=False)
