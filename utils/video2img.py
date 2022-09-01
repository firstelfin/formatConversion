#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/31 下午2:55
# @Author  : firstelfin
# @File    : video2img.py


import os
from pathlib import PurePath, Path
import cv2 as cv
from tqdm import tqdm

from utils.tool import colorstr


class Video2Img(object):
    """
    Video2Img类主要解决从视频中采样图片，支持
        - 多个源文件路径
        - 不同的视频文件
        - 采样频率多样化
    Examples
        >>> vi = Video2Img(
        >>>     root_dir="/home/industai/sda2/datatsets/charging_station/charging_station",
        >>>     out_dir="/home/industai/sda2/datatsets/charging_station/charging_images"
        >>> vi.set_sampler(3)
        >>> not_use_index = vi.trans(
        >>>     out_prefix="smokefire_industai",
        >>>     out_index=0
        >>>     )
        >>> print(not_use_index)
    )
    """

    suffixes = [".mp4", ".mov"]
    sampler = [1]

    def __init__(self, root_dir: [list, str], out_dir: [str]):
        self.root_dir = root_dir if isinstance(root_dir, list) else [root_dir]
        # 文件验证、文件夹验证
        self.files = []
        self.dirs = []
        print(colorstr('bright_green', "开始文件验证："))
        for project_name in self.root_dir:
            if Path(project_name).is_file():
                self.files.append(project_name)
            elif Path(project_name).is_dir():
                self.dirs.append(project_name)
            else:
                assert colorstr('bright_red', f"{project_name}不是合法的文件路径或文件夹")
        print(colorstr('bright_green', "开始文件夹视频获取："))
        for video_dir in self.dirs:
            video_files = Path(video_dir).iterdir()
            # 验证文件是否为视频文件，后缀验证
            for file in video_files:
                if self.suffix_valid(file.name):
                    self.files.append(file.__str__())
        del self.dirs
        # 对输出进行路径验证
        if not Path(out_dir).is_dir():
            Path(out_dir).mkdir()
        self.out_dir = out_dir
        pass

    @classmethod
    def suffix_valid(cls, name):
        """
        验证name是否拥有合法的后缀名
        Args:
            name: str, 可以是文件路径，也可以是文件名
        Returns: True/False
        """
        name_path = PurePath(name)
        if name_path.suffix in cls.suffixes:
            return True
        return False

    def suffix_add(self, suffixes):
        self.suffixes += suffixes

    @classmethod
    def set_sampler(cls, frequency=1):
        """
        改变采样策略
        Args:
            frequency: 采样间隔时间, 单位秒
        Returns: None
        """
        cls.sampler = [frequency]

    @classmethod
    def get_sampler(cls, fps=1, fps_index=0):
        """
        采样规则方法, 可以实现在不同帧率, 不同索引下给出不同的采样间隔
        Args:
            fps: 视频帧率
            fps_index: 视频采样图片索引
        Returns: 采样间隔
        """
        return cls.sampler[0]

    @classmethod
    def read_video_save_img(cls, name, out, out_prefix="smokefire_industai", out_index=0):
        """
        读取视频name, 根据cls.get_sampler的采样策略进行视频图像采样, 采样得到的图片保存到out,
            文件名前缀是out_prefix,编号是从out_index开始.
        Args:
            name: 视频文件路径
            out: 输出图片文件夹
            out_prefix: 输出文件前缀
            out_index: 输出文件的起始编号
        Returns: 输出文件未使用编号的下界
        """
        cap = cv.VideoCapture(name)
        # 获取视频属性
        fps = int(cap.get(cv.CAP_PROP_FPS))  # 帧率
        video_status = cap.isOpened()
        fps_index = 0
        sampler_code = cls.get_sampler(fps, fps_index)  # 控制你的采样策略
        while video_status:
            img_status, frame = cap.read()
            if not img_status:
                break
            if fps_index % int(fps * sampler_code) == 0:
                cv.imwrite(out + os.sep + f"{out_prefix}{out_index:06d}.jpg", frame)
                sampler_code = cls.get_sampler(fps, fps_index)
                fps_index = 0
                out_index += 1
                pass
            fps_index += 1
        cap.release()  # 释放视频
        return out_index + 1

    def trans(self, out_prefix="", out_index=0):
        """
        对所有视频、文件夹下的视频进行视频截图
        Args:
            out_prefix: 输出图片文件的前缀
            out_index: 输出图片的起始编号
        Returns: 输出文件未使用编号的下界
        """
        my_bar = tqdm(self.files, desc=colorstr("bright_yellow", "视频截图"))
        for file_name in my_bar:
            out_index = self.read_video_save_img(file_name, self.out_dir,
                                                 out_prefix=out_prefix, out_index=out_index)
            my_bar.set_postfix({colorstr("bright_blue", "Name"): file_name,
                                colorstr("bright_blue", "Index"): out_index})
        return out_index

    pass


if __name__ == '__main__':
    pass
