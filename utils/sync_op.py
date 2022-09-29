#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/28 下午5:06
# @Author  : firstelfin
# @File    : sync_op.py

from random import random
import shutil
from pathlib import Path


def sync_img_label_from_source(
        sample_ratio, root_dir: Path, new_dir: Path,
        save_img: str, save_label: str, old_img="images",
        old_label="labels", label_suffix=".txt", op="copy"
):
    """
    从一个文件夹下同步复制(移动)图片与标注文本。支持任意对应的数组，唯一要求是文件名的stem是一样的。

    ------
    Args:
        sample_ratio: 采样率，控制要复制多少数据对
        root_dir: 源数据根目录
        new_dir: 新生成数据的根目录
        save_img: 新生成图像文件夹的名字
        save_label: 新生成标注文件夹的名字
        old_img: 源文件夹图片文件夹的名字
        old_label: 源文件夹标注文件夹的名字
        label_suffix: 标注文件的后缀
        op: 要进行的同步操作['copy', 'move']

    ------
    Examples:
        >>> sync_img_label_from_source(
        >>>     sample_ratio=0.75,
        >>>     root_dir=Path("/home/industai/sda2/datasets/smokefire_industai/smokefire_datasets"),
        >>>     new_dir=Path("/home/industai/sda2/datasets/smokefire_industai/smokefire_add/charging/"),
        >>>     save_img="images",
        >>>     save_label="labels"
        >>> )

    Returns: bool

    """
    images = (root_dir / old_img).iterdir()  # 待处理的所有图片
    for img in images:
        if random() < 1 - sample_ratio:
            continue
        name = img.stem + label_suffix
        label = root_dir / old_label / name
        if label.exists():
            # 移动图片、标注数据
            getattr(shutil, op)(img.__str__(), new_dir / save_img / f"{img.name}")
            getattr(shutil, op)(label.__str__(), new_dir / save_label / f"{label.name}")
    return True


if __name__ == '__main__':
    sync_img_label_from_source(
        sample_ratio=0.75,
        root_dir=Path("/home/industai/sda2/datasets/smokefire_industai/smokefire_datasets"),
        new_dir=Path("/home/industai/sda2/datasets/smokefire_industai/smokefire_add/charging/"),
        save_img="images",
        save_label="labels"
    )
    pass
