#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/18 下午2:26
# @Author  : firstelfin
# @File    : multi2multi.py

import os
import random
import shutil
import json
import munch
from random import randint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
from bs4 import BeautifulSoup
import cv2 as cv
from xml.dom import minidom
from pathlib import Path

from tool import colorstr

seed = randint(2022, 2025)

labels_name = {
    "白烟": "0",
    "灰烟": "0",
    "黄烟": "0",
    "黑烟": "0",
    "火焰": "1",
    "黄焰": "1",
    "烟": "0",
    "火": "1",
    "烟雾": "0",
    "OilDetect": "0"
}

valid_list = ["黄焰", "黄烟"]
print_list = []


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

    Examples
        >>> xml2yolo = Xml2Yolo(
        >>>     origin_dir="/home/industai/sda2/datatsets/smokefire_industai/smokefire_industaiv2/",
        >>>     new_dir="/home/industai/sda2/datatsets/smokefire_industai/middlev2/", delete=False, sampler=0
        >>> )
        >>> xml2yolo.trans(0)
    """

    def __init__(self, origin_dir="", new_dir="", delete=False, sampler=0):
        self.opt = {
            "origin_dir": origin_dir,
            "new_dir": new_dir,
            "delete": delete,
            "sampler": sampler
        }
        self.opt = munch.munchify(self.opt)

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
        return f"oildetect{index:06d}.{suffix}"

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
            old_name, old_suffix = img.split(".")
            new_label_name = self.rename(i + start, False)
            # old_label
            old_label = old_name + ".xml"
            label = self.read_xml(origin_dir + "/labels/" + old_label)
            # for ntl in label["object"]:
            #     if ntl[0] in valid_list:
            #         print_list.append(old_name)
            new_label_txt = self.produce_txt(label)

            # 是否删除旧图像、旧标注，当图片无标注信息时
            if not new_label_txt and self.opt.delete:
                os.remove(origin_dir + f"images/{old_name}.jpg")
                if os.path.exists(origin_dir + "labels/" + img):
                    os.remove(origin_dir + "labels/" + img)
                continue

            if change_datasets:
                # copy 图片到指定文件夹
                if Path(origin_dir + f"images/{old_name}.jpg").exists():
                    img_suffix = "jpg"
                else:
                    img_suffix = "JPG"
                shutil.copy(origin_dir + f"/images/{old_name}.{img_suffix}",
                            new_dir + f"/images/{self.rename(i + start)}")
                self.save_yolo_txt(new_dir + "/labels/" + new_label_name, new_label_txt)
            else:
                self.save_yolo_txt(origin_dir + "/labels/" + new_label_name, new_label_txt)
        pass
    pass


class Yolo2Xml(object):
    """
    Yolo数据格式转换为PascalVOC格式
    yolo数据存放格式：
    - 数据集
        |- images
        |- labels
        |- classes.txt(可存放在labels里面)

    Examples
        >>> yolo = Yolo2Xml("/home/industai/sda2/datatsets/charging_station/charging/")
        >>> yolo.trans()
    """
    class_names = {
        "0": "烟",
        "1": "火"
    }

    def __init__(self, root_dir):

        self.root_dir = root_dir
        pass

    @classmethod
    def add_txt_element(cls, xml_data, name, doc, root):
        """
        向根节点root添加name节点，并添加文本内容
        Args:
            xml_data: 要渲染的数据
            name: 节点名
            doc: Document
            root: 要操作的根节点
        Returns: Document
        """
        folder = doc.createElement(name)
        folder_node = doc.createTextNode(xml_data[name])
        folder.appendChild(folder_node)
        root.appendChild(folder)
        return doc

    @classmethod
    def produce_xml(cls, xml_data, out="xml_labels/test.xml"):
        """
        基于PascalVOC格式的字典数据生成xml数据，并保存
        Args:
            xml_data: PascalVOC格式的字典数据
            out: 保存xml文件的路径
        Returns: None
        """
        doc = minidom.Document()
        root = doc.createElement("annotation")
        doc.appendChild(root)
        # 添加第一级标签
        cls.add_txt_element(xml_data, "folder", doc, root)
        cls.add_txt_element(xml_data, "filename", doc, root)
        cls.add_txt_element(xml_data, "path", doc, root)
        cls.add_txt_element(xml_data, "segmented", doc, root)
        # 添加第二级公共标签
        source = doc.createElement("source")
        root.appendChild(source)
        cls.add_txt_element(xml_data["source"], "database", doc, source)
        size_node = doc.createElement("size")
        root.appendChild(size_node)
        cls.add_txt_element(xml_data["size"], "width", doc, size_node)
        cls.add_txt_element(xml_data["size"], "height", doc, size_node)
        cls.add_txt_element(xml_data["size"], "depth", doc, size_node)
        # 添加object标签
        for object_i in xml_data["object"]:
            object_node = doc.createElement("object")
            root.appendChild(object_node)
            cls.add_txt_element(object_i, "name", doc, object_node)
            cls.add_txt_element(object_i, "pose", doc, object_node)
            cls.add_txt_element(object_i, "truncated", doc, object_node)
            cls.add_txt_element(object_i, "difficult", doc, object_node)
            bndbox = doc.createElement("bndbox")
            object_node.appendChild(bndbox)
            cls.add_txt_element(object_i["bndbox"], "xmin", doc, bndbox)
            cls.add_txt_element(object_i["bndbox"], "ymin", doc, bndbox)
            cls.add_txt_element(object_i["bndbox"], "xmax", doc, bndbox)
            cls.add_txt_element(object_i["bndbox"], "ymax", doc, bndbox)
            pass

        with open(out, "w+", encoding="utf-8") as f:
            doc.writexml(f, indent="\t", addindent="\t", newl="\n", encoding="utf-8")
            f.close()
        return

    @classmethod
    def read_txt(cls, root_dir, name, img_size):
        """
        读取yolo格式的标注文件，将其转换为PascalVOC格式的字典
        Args:
            root_dir: 待处理的数据集根目录
            name: 待处理的文件名
            img_size: 图片大小
        Returns: 文件存在返回格式化的数据, 不存在返回False
        """
        try:
            with open(root_dir + os.sep + "labels" +
                      os.sep + name.split('.')[0] + ".txt", "r+", encoding="utf-8") as f:
                data = f.readlines()
                f.close()
        except FileNotFoundError:
            return False
        read_result = {
            "folder": "smokefire_industai",
            "filename": f"{name}",
            "path": f"{root_dir}/images/{name}",
            "source": {
                "database": "UnKnown"
            },
            "size": {
                "width": str(img_size[0]),
                "height": str(img_size[1]),
                "depth": str(img_size[2])
            },
            "segmented": "0",
            "object": []
        }
        for d in data:
            class_id, x, y, w, h = d.split(" ")
            h = h[:-1]
            middle_object = {
                "name": cls.class_names[class_id],
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {
                        "xmin": str(round((float(x) - 0.5 * float(w)) * img_size[0])),
                        "ymin": str(round((float(y) - 0.5 * float(h)) * img_size[1])),
                        "xmax": str(round((float(x) + 0.5 * float(w)) * img_size[0])),
                        "ymax": str(round((float(y) + 0.5 * float(h)) * img_size[1]))
                    }
            }
            read_result["object"].append(middle_object)
        return read_result

    def trans(self, out="xml_lables"):
        if not os.path.exists(self.root_dir + os.sep + out):
            os.mkdir(self.root_dir + os.sep + out)
        origin_images = os.listdir(self.root_dir + "/images/")
        my_bar = tqdm(origin_images, desc=colorstr("Yolo2Xml trans:") + f"{self.root_dir}")
        for name in my_bar:
            img = cv.imread(self.root_dir + "/images/" + name)
            h, w, c = img.shape
            del img
            xml_data = self.read_txt(self.root_dir, name, [w, h, c])
            if xml_data:
                self.produce_xml(xml_data,
                                 out=self.root_dir + os.sep + out + os.sep + name.split(".")[0] + ".xml")
            my_bar.set_postfix({colorstr("image"): name, "status": not isinstance(xml_data, bool)})
            pass
        pass

    pass


if __name__ == '__main__':
    xml2yolo = Xml2Yolo(
        origin_dir="/home/industai/sda2/datasets/smokefire_industai/20220817smoke/",
        new_dir="/home/industai/sda2/datasets/smokefire_industai/20220817smoke/yolo_text/",
        delete=False, sampler=0
    )
    xml2yolo.trans(30000)
    print(print_list)
    # yolo = Yolo2Xml("/home/industai/sda2/datatsets/charging_station/charging/")
    # yolo.trans()
    pass

