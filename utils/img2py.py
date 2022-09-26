#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 下午1:44
# @Author  : firstelfin
# @File    : img2py.py


import numpy as np
import base64
from io import BufferedReader
from flask import request, Flask, jsonify
from werkzeug.datastructures import FileStorage
import cv2 as cv


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json;charset=utf-8"


def is_base64_str(input_str: str):

    return True if input_str and "base64,/" in input_str\
                   and input_str.startswith("data:image/") else False


@app.route("/infer", methods=["POST"])
def test():
    """
    https://blog.51cto.com/u_15549234/5139108数据转换博客
    Returns:
    """
    res = {
        "code": 200,
        "msg": "",
        "data": {}
    }
    img_file = request.files.get("imgFile")
    if not img_file:
        img_file = request.form.get("imgFile")
    img_base = request.form.get("imgBase64")
    if isinstance(img_file, FileStorage):
        res["msg"] = "入参是一个文件\n"
        # todo: 进行FileStorage对象转numpy数据
        img_buff = BufferedReader(img_file)
        img_byte = BufferedReader.read(img_buff)
        img_arr = np.frombuffer(img_byte, dtype=np.uint8)
        img = cv.imdecode(img_arr, 1)
        del img_buff, img_byte, img_arr
    elif is_base64_str(img_base):
        res["msg"] = "入参是base64编码\n"
        img_decode = base64.b64decode(img_base.split(";base64,")[-1])
        img_arr = np.fromstring(img_decode, dtype=np.uint8)
        img = cv.imdecode(img_arr, 1)
        # img = cv.imdecode(img_decode, 1)
        # del img_decode, img_arr
    else:
        res["msg"] = "入参是 [图片路径, 文件夹, 摄像头, 视频] \n"
        img = img_file  # 图片路径、文件夹、摄像头、视频

    return jsonify(res)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=10016, debug=True)
