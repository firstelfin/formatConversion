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
        path_root: 数据集的根目录
        split_ratio: 切分比例
        year: 年份
    Returns: 切分后的数据集分布
    """
    all_images_paths = os.listdir(path_root + "/images/")
    train, valid_test = train_test_split(all_images_paths, train_size=split_ratio[0], random_state=2022)
    valid, test = train_test_split(valid_test,
                                   test_size=split_ratio[-1] / sum(split_ratio[1:]), random_state=2022)
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


if __name__ == '__main__':
    rename_files("/home/industai/sda2/datatsets/charging_images", out_index=21342)
