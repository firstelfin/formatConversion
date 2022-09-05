#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 上午11:39
# @Author  : firstelfin
# @File    : tool.py

import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def colorstr(*x):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = x if len(x) > 1 else ('blue', 'bold', x[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def add_prefix(split_data, root_dir):
    return [root_dir + "/images/" + sd + "\n" for sd in split_data]


def save_txt(root_dir, split_data, mode="train", year="2017"):
    with open(root_dir + f"{mode}{year}.txt", "w+", encoding="utf-8") as f:
        f.writelines(split_data)
        f.close()
    pass


def split_datasets(path_root, split_ratio, year):
    """
    切分数据集，生成train.txt、valid.txt、test.txt

    Args:
    ------
        path_root: 数据集的根目录\n
        split_ratio: 切分比例\n
        year: 年份\n

    Examples
    ------
        >>> split_datasets(
        >>>     path_root="/home/industai/sda2/datatsets/smokefire_industai/",
        >>>     split_ratio=[0.8, 0.1, 0.1],
        >>>     year=2022
        >>> )

    Returns: 切分后的数据集分布
    """
    all_images_paths = os.listdir(path_root + "/images/")
    train, valid_test = train_test_split(all_images_paths, train_size=split_ratio[0], random_state=2022)
    valid, test = train_test_split(valid_test,
                                   train_size=split_ratio[1] / sum(split_ratio[1:]), random_state=2022)
    train = add_prefix(train, path_root)
    save_txt(path_root, train, year=year)
    valid = add_prefix(valid, path_root)
    save_txt(path_root, valid, "valid", year=year)
    test = add_prefix(test, path_root)
    save_txt(path_root, test, "test", year=year)
    return train, valid, test


def rename_signal_file(root_path, new_name):
    root_path.rename(new_name)
    pass


def rename_files(root_dir, out_prefix="smokefire_industai", out_suffix=".jpg", out_index=0):
    """
    将图片、文件夹下的所有图片重命名

    Args:
    ------
        root_dir: 目标根路径\n
        out_prefix: 输出前缀\n
        out_suffix: 输出后缀\n
        out_index: 输出编码\n

    Examples
    ------
        >>> rename_files("/home/industai/sda2/datatsets/charging_station/chargingFP/", out_index=23143)

    Returns: next code index.
    """
    root_path = Path(root_dir)
    if root_path.is_dir():
        for file in root_path.iterdir():
            rename_signal_file(file, "/" + root_path.absolute().__str__() + os.sep
                               + f"{out_prefix}{out_index:06d}{out_suffix}")
            out_index += 1
            pass
        pass
    elif root_path.is_file():
        rename_signal_file(root_path, "/" + root_path.parent.absolute().__str__() + os.sep
                           + f"{out_prefix}{out_index:06d}{out_suffix}")
        out_index += 1
        pass
    return out_index


def write_yolov5_txt(root_dir, prefix="/home/industai/sda2/datatsets/charging_station",
                     out="/home/industai/sda2/datatsets/smokefire_industai/valid_charging.txt"):
    """
    将root_dir下的所有文件写入yolo训练文件train.txt这种格式

    Args:
    ------
        root_dir: 源文件夹\n
        prefix: 文件前缀\n
        out: 输出路径

    Examples
    -------
        >>> write_yolov5_txt(
        >>>     root_dir="/home/industai/sda2/datatsets/charging_station/chargingFP/",
        >>>     prefix="/home/industai/sda2/datatsets/smokefire_industai/images/"
        >>> )

    Returns:
    """
    write_path = Path(root_dir).iterdir()
    if prefix is None:
        prefix = root_dir
    with open(out, "w+", encoding="utf-8") as f:
        for file in write_path:
            f.write(prefix + os.sep + file.name + "\n")
        f.close()
    pass


if __name__ == '__main__':
    # rename_files("/home/industai/sda2/datatsets/charging_station/chargingFP/", out_index=23143)
    # write_yolov5_txt(
    #     root_dir="/home/industai/sda2/datatsets/charging_station/chargingFP/",
    #     prefix="/home/industai/sda2/datatsets/smokefire_industai/images/"
    # )
    split_datasets(
        path_root="",
        split_ratio=[0.8, 0.1, 0.1],
        year=2022
    )
    pass
