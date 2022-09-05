#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 上午11:19
# @Author  : firstelfin
# @File    : img_fuse.py

import random
import numpy as np
import cv2 as cv
from pathlib import Path, PosixPath
from tqdm import tqdm
from sklearn.cluster import KMeans
import warnings

from utils import colorstr

EPSILON = 1e-5
warnings.filterwarnings("ignore")


class ImgFuse(object):
    """对两个源图片进行像素级融合

    Examples::
        charging_fuse = ImgFuse(
            origin_dir="/home/industai/sda2/datatsets/charging_station/charging_images/",
            render_dir="/home/industai/project/datasets/VJshi/set/images/",
            out_dir="/home/industai/sda2/datatsets/charging_station/charging/",
            shuffle=False
        )

        charging_fuse.img_fuse(1800, start_num=21342, prefix_img="/images/smokefire_industai",
                               prefix_label="/labels/smokefire_industai")
    """

    CHANGE_SUFFIX = {
        "mp4": ".mov",
        "mov": ".mp4"
    }
    fire = False

    def __init__(self, origin_dir, render_dir, out_dir, shuffle):
        """
        :param origin_dir: 源图根目录
        :param render_dir: 渲染根目录
        :param out_dir:    保存文件的根目录
        :param shuffle:    是否随机采样
        """
        self.origin_dir = origin_dir
        self.render_dir = render_dir
        self.out_dir = out_dir
        self.shuffle = shuffle

    @classmethod
    def point_wise_add(cls, img1, img2):
        """
        img2是灰度图，RGB值是相等的
        """
        weights = np.zeros_like(img1, dtype=np.float16)
        max_num = img2.max()
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                weights[i, j, :] = (img2[i, j, 0] + EPSILON) / (max_num + EPSILON)
        new_img = img2 * weights + img1 * (1 - weights)
        return new_img.astype(np.uint8)

    @classmethod
    def bbox_of_obj(cls, img):
        """
        根据图片寻找连通区域，并计算其边界
        """
        img3 = cv.medianBlur(img, 3)  # 可选操作
        H, W, _ = img.shape

        # 进行膨胀腐蚀操作，闭运算: 寻找目标对象的边界框
        bbox_img = cls.close(img3[:, :, 1], kernel_size=[5, 3], iterations=[7, 5])
        kmeans = cls.get_channel_centers(img3[:, :, :1])
        threshold = kmeans[0] + np.std(img3[:, :, :1])
        bbox_img[bbox_img < threshold] = 0
        bbox_img[bbox_img >= threshold] = 255
        conn = cls.connected_components(bbox_img)
        # 对连通区域进行可视化标记
        # 生成标签列表
        labels = []
        for v in conn[2]:
            x1, y1, w, h, ares, code = v
            x2, y2 = x1 + w, y1 + h
            cv.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2, lineType=8)
            labels.append([int(cls.fire), x1 / W, y1 / H, x2 / W, y2 / H])
            pass
        return img, labels

    @classmethod
    def connected_components(cls, img, labels=2, connectivity=8, l_type=cv.CV_32S):
        """
        寻找连通区域, 控制面积小于7%的目标; todo:目标包含在另一个目标中的剔除
        return: result = (
            int obj_number,
            np.ndarray connected_image,
            np.ndarray bbox_info,
            np.ndarray center_of_mass
        )
        """
        # conn = cv.connectedComponents(img, labels, connectivity, ltype=l_type)
        conn = cv.connectedComponentsWithStats(img, connectivity=connectivity, ltype=l_type)
        conn_img = conn[1]
        # 寻找背景编码
        background_code = None
        for i in range(conn_img.max() + 1):
            if not img[conn_img == i].sum():
                background_code = i
                break
        if background_code is not None:
            delete_code = [background_code]
        else:
            delete_code = []

        # 连通区域筛选
        ares = img.shape[0] * img.shape[1]
        box_obj = conn[2]
        if box_obj.shape[0] > 2:
            threshold = ares * 2 / 100
        else:
            threshold = ares * 0.5 / 100
        candidate = []
        for i, values in enumerate(box_obj):
            if i in delete_code or values[-1] < threshold:
                delete_code.append(i)
                continue
            candidate.append(i)
        for i in delete_code:
            conn_img[conn_img == i] = min(delete_code)
        result = (
            len(candidate),
            conn_img,
            np.hstack([box_obj[candidate], np.array(candidate).reshape(-1, 1)]) if candidate else [],
            conn[3][candidate]
        )
        return result

    @classmethod
    def bilateral_filter(cls, img):
        """
        双边滤波
        """
        bi = cv.bilateralFilter(img, 9, 75, 75)
        return bi

    @classmethod
    def get_channel_centers(cls, img):
        centers = []
        for i in range(img.shape[-1]):
            ki_means = KMeans(n_clusters=2).fit(np.reshape(img[:, :, i], [-1, 1]))
            std = np.std(img[:, :, i])
            middle = max(25, np.mean(ki_means.cluster_centers_.flatten()) - std)
            centers.append(middle)
        return centers

    @classmethod
    def erode(cls, img, kernel_size=3, iterations=1):
        """
        腐蚀操作
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion = cv.erode(img, kernel, iterations=iterations)
        return erosion

    @classmethod
    def dilate(cls, img, kernel_size=3, iterations=1):
        """
        膨胀操作
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        expand = cv.dilate(img, kernel, iterations=iterations)
        return expand

    @classmethod
    def close(cls, img, kernel_size=(3, 3), iterations=(1, 1)):
        """
        闭运算, 获取烟火对象的区域.
        """
        img = cls.dilate(img, kernel_size[0], iterations[0])
        img = cls.erode(img, kernel_size[1], iterations[1])
        return img

    @classmethod
    def get_random_img(cls, origin, render, shuffle=True, number=0.):
        """
        随机获取两个图片的路径, 默认是随机采样
        :param origin:   源图片集
        :param render:   渲染图片集
        :param shuffle:  是否随机采样
        :param number:   第几次采样
        :return: origin_img, render_img
        """
        if shuffle:
            origin_index = random.randint(0, len(origin) - 1)
            render_index = random.randint(0, len(render) - 1)
        else:
            origin_index = number % len(origin)
            render_index = number % len(render)
        return origin[origin_index], render[render_index]

    @classmethod
    def test_img_fuse(cls, name1, name2):
        img1 = cv.imread(filename=name1, flags=1)  # shape=(480, 640, 3)
        img1_h, img1_w, _ = img1.shape
        img2 = cv.imread(filename=name2, flags=1)  # shape=(1152, 2048, 3)
        img2 = cv.resize(img2, dsize=(img1_w, img1_h), interpolation=cv.INTER_LINEAR)
        add_img = cls.point_wise_add(img1, img2)

        new_img2, labels = cls.bbox_of_obj(img2)
        for label in labels:
            cv.rectangle(add_img, (int(label[1] * img1_w), int(label[2] * img1_h)),
                         (int(label[3] * img1_w), int(label[4] * img1_h)),
                         color=(0, 255, 0), thickness=2, lineType=8)
        concat = np.hstack([add_img, img2])
        cv.imshow("img fuse", concat)
        cv.waitKey(0)

    @classmethod
    def fuse(cls, name1, name2, out, number, show=False,
             start_num=0, prefix_img="/images/smokeFireFuse",
             prefix_label="/labels/smokeFireFuse", suffix=".jpg"):
        """
        对两张图片进行融合

        Args:
            suffix: 融合图片命名的后缀
            prefix_img: 融合图片的前缀
            prefix_label: 融合图片的标注文件前缀
            start_num: 融合图片的起始编码
            number: 融合的图片编码
            name1: 源图片
            name2: 渲染图片
            out:   图片保存路径
            show:  是否展示

        return: None.
        """
        if isinstance(name1, PosixPath):
            name1 = name1.__str__()
        if isinstance(name2, PosixPath):
            name2 = name2.__str__()
        img1 = cv.imread(filename=name1, flags=1)  # shape=(480, 640, 3)
        img1_h, img1_w, _ = img1.shape
        img2 = cv.imread(filename=name2, flags=1)  # shape=(1152, 2048, 3)
        img2 = cv.resize(img2, dsize=(img1_w, img1_h), interpolation=cv.INTER_LINEAR)
        add_img = cls.point_wise_add(img1, img2)

        new_img2, labels = cls.bbox_of_obj(img2)

        # for label in labels:
        #     cv.rectangle(add_img, (int(label[1] * img1_w), int(label[2] * img1_h)),
        #                  (int(label[3] * img1_w), int(label[4] * img1_h)),
        #                  color=(0, 255, 0), thickness=2, lineType=8)
        # todo: 将渲染后的图片保存到 out/images文件夹
        cv.imwrite(out + f"{prefix_img}{number + start_num:06d}{suffix}", add_img)
        # todo: 将渲染后的标注保存到 out/labels文件夹
        with open(out + f"{prefix_label}{number + start_num:06d}.txt", "w+", encoding="utf-8") as f:
            for label in labels:
                center_x, center_y = (label[1] + label[3]) / 2, (label[2] + label[4]) / 2
                w, h = label[3] - label[1], label[4] - label[2]
                f.write(" ".join([str(label[0]), str(center_x), str(center_y), str(w), str(h)]) + "\n")
            f.close()
        if show:
            concat = np.hstack([add_img, img2])
            cv.imshow("img fuse", concat)
            cv.waitKey(0)
        return None

    def img_fuse(self, number=1e4, start_num=0, prefix_img="/images/smokeFireFuse",
                 prefix_label="/labels/smokeFireFuse", suffix=".jpg"):
        """
        多张图片融合
        Args:
            number: 要生成的图像数量
            start_num: 保存图片的起始编号
            prefix_img: 融合图片的前缀
            prefix_label: 融合图片的标注文件前缀
            suffix: 融合图片的后缀

        return: None
        """
        total_num = number
        origin = list(Path(self.origin_dir).iterdir())
        # origin = glob.glob(self.origin_dir)
        render = list(Path(self.render_dir).iterdir())
        # render = glob.glob(self.render_dir)
        assert type(number) == int and number > 0, "number必须是大于0的整数"
        my_bar = tqdm(range(int(number), 0, -1), desc=colorstr("bright_magenta", "图片渲染:"))
        for i in my_bar:
            origin_p, render_p = self.get_random_img(origin, render, number=i)
            self.fuse(origin_p, render_p, self.out_dir, total_num - i, start_num=start_num,
                      prefix_img=prefix_img, prefix_label=prefix_label, suffix=suffix)
            my_bar.set_postfix({colorstr("bright_blue", "images index"): i})
        pass


if __name__ == '__main__':
    # 实例化一个COCO数据集渲染对象
    charging_fuse = ImgFuse(
        origin_dir="/home/industai/sda2/datatsets/charging_station/charging_images/",
        render_dir="/home/industai/project/datasets/VJshi/set/images/",
        out_dir="/home/industai/sda2/datatsets/charging_station/charging/",
        shuffle=False
    )
    charging_fuse.img_fuse(1800, start_num=21342, prefix_img="/images/smokefire_industai",
                           prefix_label="/labels/smokefire_industai")
