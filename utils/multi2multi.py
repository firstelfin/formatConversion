#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/18 下午2:26
# @Author  : firstelfin
# @File    : multi2multi.py

import os
import random
import shutil
import json
from random import randint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
from bs4 import BeautifulSoup
import cv2 as cv

from tool import colorstr

seed = randint(2022, 2025)

labels_name = {
    "白烟": "0",
    "灰烟": "0",
    "黑烟": "0",
    "火焰": "1",
    "黄焰": "1",
}


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_dir', default='./data', type=str,
                        help="数据集的根目录.")
    parser.add_argument('--new_dir', type=str, default='', help="新建数据集的根目录.")
    parser.add_argument('--delete', action="store_true", help="是否删除无标注内容的图像和标注文件.")
    parser.add_argument("--sampler", type=float, default=0., help="采样间隔")

    opt = parser.parse_args()
    return opt


class Xml2Yolo(object):
    """
    PascalVOC标注数据转为AOC数据格式
    """

    def __init__(self):
        self.opt = get_opt()

    @classmethod
    def read_xml(cls, name):
        if not os.path.exists(name):
            return {"object": []}
        with open(name, "r+", encoding="utf-8-sig") as f:
            data = f.read()
            f.close()
        soup = BeautifulSoup(data, 'lxml')
        objects = soup.find_all("object")
        res = {
            "height": int(soup.select("size height")[0].text),
            "width": int(soup.select("size width")[0].text),
            "channel": int(soup.select("size depth")[0].text),
            "object": []
        }
        for obj in objects:
            name = str(obj.find("name").text)
            name = name.encode().decode("utf-8-sig")
            difficult = int(obj.find("difficult").text)
            x_min = int(obj.select("bndbox xmin")[0].text) / res["width"]
            y_min = int(obj.select("bndbox ymin")[0].text) / res["height"]
            x_max = int(obj.select("bndbox xmax")[0].text) / res["width"]
            y_max = int(obj.select("bndbox ymax")[0].text) / res["height"]
            x, y = (x_max + x_min) / 2, (y_max + y_min) / 2
            w, h = (x_max - x_min), (y_max - y_min)
            res["object"].append([name, difficult, str(x), str(y), str(w), str(h)])
        return res

    @classmethod
    def rename(cls, index, img=True):
        """
        对图片和标注文件重命名.
        Args:
            index: 当前处理对象在数据集中的索引编号
            img: 对象是否为图片
        Returns: 新的文件名
        """
        suffix = "jpg" if img else "txt"
        return f"smokefire_industai{index: 06d}.{suffix}"

    @classmethod
    def produce_txt(cls, data):
        res = []
        for d in data["object"]:
            res.append(" ".join([labels_name[d[0]]] + d[2:]) + "\n")
        return res

    @classmethod
    def save_yolo_txt(cls, filename, txt):
        with open(filename, "w+", encoding="utf-8") as f:
            f.writelines(txt)
            f.close()
        pass

    @classmethod
    def sampler(cls, data, fps):
        if isinstance(fps, int) and fps > 1:
            sample_seed = True
        else:
            sample_seed = False
        res = []
        num = 0
        for d in data:
            if sample_seed and num % fps == 1:
                res.append(d)
                continue
            if random.random() < fps:
                res.append(d)
        return res

    def trans(self, start=0):
        origin_dir = self.opt.origin_dir
        new_dir = self.opt.new_dir
        # origin_images = os.listdir(origin_dir + "images/")
        origin_labels = os.listdir(origin_dir + "/labels/")
        if new_dir == "":
            change_datasets = False
        else:
            change_datasets = True
            if not os.path.exists(new_dir + "/images"):
                os.mkdir(new_dir + "/images")
            if not os.path.exists(new_dir + "/labels"):
                os.mkdir(new_dir + "/labels")

        # 对图像标注进行采样
        if self.opt.sampler:
            origin_labels = self.sampler(origin_labels, self.opt.sampler)

        for i, img in enumerate(origin_labels):
            old_name = img.split(".")[0]
            new_label_name = self.rename(i + start, False)
            # old_label
            old_label = old_name + ".xml"
            label = self.read_xml(origin_dir + "/labels/" + old_label)
            new_label_txt = self.produce_txt(label)
            # 是否删除旧图像、旧标注，当图片无标注信息时
            if not new_label_txt and self.opt.delete:
                os.remove(origin_dir + f"images/{old_name}.jpg")
                if os.path.exists(origin_dir + "labels/" + img):
                    os.remove(origin_dir + "labels/" + img)
                continue

            if change_datasets:
                # copy 图片到指定文件夹
                shutil.copy(origin_dir + f"/images/{old_name}.jpg",
                            new_dir + f"/images/{self.rename(i + start)}")
                self.save_yolo_txt(new_dir + "/labels/" + new_label_name, new_label_txt)
            else:
                self.save_yolo_txt(origin_dir + "/labels/" + new_label_name, new_label_txt)
        pass
    pass


if __name__ == '__main__':
    xml2yolo = Xml2Yolo()
    xml2yolo.trans()
