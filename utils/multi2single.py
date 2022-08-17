#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 上午11:04
# @Author  : firstelfin
# @File    : multi2single.py
"""
YOLO格式的数据集转化为COCO格式的数据集
"""

import os
import json
from random import randint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import cv2 as cv

from tool import colorstr

seed = randint(2022, 2025)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data', type=str,
                        help="数据集的根目录, include images and labels and [classes.txt].")
    parser.add_argument('--save_path', type=str, default='./train.json',
                        help="不切分时, 最终保存文件的相对路径")
    parser.add_argument('--split_files', action='store_true', help="随机切分数据为3个子集, 默认比例为8:1:1")
    parser.add_argument('--split_file', action='store_true',
                        help="从train.txt val.txt test.txt文件获取切分数据子集")
    parser.add_argument('--split_root', type=str, default="./", help="train.txt val.txt test.txt文件所在路径.")
    parser.add_argument('--ratio', type=tuple, default=(0.8, 0.1, 0.1),
                        help=f"Split ratio of data set: {colorstr('train:valid:test')}")

    opt = parser.parse_args()
    return opt


class Yolo2Coco(object):
    """
    YOLO格式的数据集转化为COCO格式的数据集
    """

    def __init__(self):
        self.opt = get_opt()
        pass

    @classmethod
    def split_random1(cls, img_paths, ratio=(0.8, 0.1, 0.1)):
        """
        数据集采用如下格式时切分数据集：
        data
         - images
            - *.jpg
         - labels
            - *.txt
         - classes.txt
         classes.txt可在label中
        :param img_paths: 待切分的图片数据集
        :param ratio: train、valid、test的比例
        :return: train_img、valid_img、test_img
        """
        assert int(sum(ratio)) == 1, f"the sum of ratio excepted 1, got {ratio}."
        train_img, valid_test = train_test_split(img_paths, test_size=1-ratio[0], random_state=seed)
        valid_img, test_img = train_test_split(valid_test, test_size=ratio[2]/(ratio[1]+ratio[2]), random_state=seed)
        print(f'The number of split: train-{colorstr(len(train_img))} '
              f'valid-{colorstr(len(valid_img))} test-{colorstr(len(test_img))}')
        return train_img, valid_img, test_img

    @classmethod
    def split_random2(cls, root_dir, phases=("train", "valid", "test")):
        """
        train、valid、test都从txt文件获取对应的样本
        :param phases: 待处理文件的名字
        :param root_dir: train.txt、valid.txt、test.txt的根目录
        :return: train_img, valid_img, test_img
        """
        images = []
        for p in phases:
            assert os.path.exists(root_dir+f"{p}.txt"), f"{p}.txt not exists."
            with open(root_dir+f"{p}.txt", "r+", encoding="utf-8-sig") as f:
                images.append(f.readlines())
                f.close()
        return images

    @classmethod
    def save_json(cls, json_data, out_dir, name=""):
        filename = out_dir + f"/{name}.json"
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(json_data, f)
            f.close()
        print(colorstr("FileSave") + f":{filename}")
        pass

    def trans(self):
        print(colorstr("Loading dataset") + f" from {self.opt.root_dir}...")
        assert os.path.exists(self.opt.root_dir)
        origin_images = os.path.join(self.opt.root_dir, "images")
        origin_labels = os.path.join(self.opt.root_dir, "labels")
        # 验证classes.txt是否在根目录下
        if os.path.exists(self.opt.root_dir + "/classes.txt"):
            origin_class = os.path.join(self.opt.root_dir, "classes.txt")
        else:
            origin_class = os.path.join(origin_labels, "classes.txt")
        assert os.path.exists(origin_class), f"{colorstr('NotFondFileError')}: {origin_class} not exists."
        # 读入类别名词文件
        with open(origin_class, "r+", encoding="utf-8-sig") as f:
            names = f.read().strip().split()
            f.close()
        # 读入图片名词列表
        images = os.listdir(origin_images)
        # images = glob.glob(origin_images + "/*")
        train_img, valid_img, test_img = [], [], []
        train_data, valid_data, test_data, datasets = {}, {}, {}, {}

        if self.opt.split_files or self.opt.split_file:
            # 初始化coco格式的数据保存变量
            train_data = {"categories": [], "annotations": [], "images": []}
            valid_data = {"categories": [], "annotations": [], "images": []}
            test_data = {"categories": [], "annotations": [], "images": []}

            # 建立类别名与编码的对应关系，类别从0开始
            for i, name in enumerate(names):
                train_data["categories"].append({"id": i, "name": name, "supercategory": "mark"})
                valid_data["categories"].append({"id": i, "name": name, "supercategory": "mark"})
                test_data["categories"].append({"id": i, "name": name, "supercategory": "mark"})
                pass
            # 切分数据集
            if self.opt.split_files:
                train_img, valid_img, test_img = self.split_random1(images, ratio=self.opt.ratio)
            elif self.opt.split_file:
                if self.opt.split_root == "./":
                    split_root = self.opt.root_dir
                else:
                    split_root = os.path.join(self.opt.root_dir, self.opt.split_root)
                train_img, valid_img, test_img = self.split_random2(split_root)
        else:
            datasets = {"categories": [], "annotations": [], "images": []}
            for i, name in enumerate(names):
                datasets["categories"].append({"id": i, "name": name, "supercategory": "mark"})
            pass

        # yolo格式数据转换
        ann_cnt_id = 0
        my_bar = tqdm(images, desc=f"{colorstr(self.opt.root_dir)}数据集转换进行中...")
        for i, img in enumerate(my_bar):
            # 获取标注文件
            label_file = img.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
            # 读取图像的高和宽
            im = cv.imread(origin_images + "/" + img)
            height, width, _ = im.shape
            # 判定图像被随机分配到哪一部分
            if self.opt.split_file or self.opt.split_files:
                if img in train_img:
                    datasets = train_data
                elif img in valid_img:
                    datasets = valid_data
                elif img in test_img:
                    datasets = test_data
            # 添加图像的基本信息
            datasets["images"].append({
                "file_name": img,
                "id": i,
                "width": width,
                "height": height
            })
            # 标注文件不存在时跳过循环
            if not os.path.exists(os.path.join(origin_labels, label_file)):
                continue

            # 读取标注文件
            with open(os.path.join(origin_labels, label_file), "r+", encoding="utf-8-sig") as f:
                label_list = f.readlines()
                f.close()
            for label in label_list:
                label = label.strip().split()
                x, y, w, h = float(label[1]), float(label[2]), float(label[3]), float(label[4])
                # yolo 2 coco
                x1, y1, x2, y2 = (x - w / 2) * width, (x - h / 2) * height, \
                                 (x + w / 2) * width, (x + h / 2) * height

                # 标签序号从0开始计数
                name_id = int(label[0])
                ins_width = max(0, x2 - x1)
                ins_height = max(0, y2 - y1)
                datasets["annotations"].append({
                    "area": ins_width * ins_height,
                    "bbox": [x1, y1, ins_width, ins_height],
                    "id": ann_cnt_id,
                    "image_id": i,
                    "category_id": name_id,
                    "iscrowd": 0,
                    "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_cnt_id += 1
        print(colorstr("End of data conversion..."))
        # 开始保存转换后的数据
        save_data = {
            "train": train_data,
            "val": valid_data,
            "test": test_data,
            "other": datasets,
        }
        out_dir = os.path.join(self.opt.root_dir, "annotations")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if train_data:
            for p in ["train", "val", "test"]:
                self.save_json(save_data[p], out_dir, p)
        else:
            self.save_json(save_data["other"], out_dir, self.opt.save_path)
            pass


if __name__ == '__main__':
    y2c = Yolo2Coco()
    y2c.trans()
