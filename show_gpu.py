#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/19 上午9:41
# @Author  : firstelfin
# @File    : show_gpu.py
import matplotlib.pyplot as plt

# todo: 1e3~1e8之间的数组大小进行执行时间测试
# todo: 加速比的测试

cpu_float = [52.0129, 461.242, 4616.74, 46194.7, 462647, 4625473.9]
gpu_float = [0.79872, 0.798624, 3.63533, 60.0875, 534.72, 5372.62]
gpu_float_ = [0.828672, 0.79856, 7.80778, 55.1086, 572.557, 5556.64]
x = [1000, 10000, 100000, 1000000, 10000000, 100000000]
x2 = [1,2,3,4,5,6]
x1 = [128*((i+127) / 128) for i in x]
gpu = [gpu_float[i] / x[i] for i in range(6)]

ratio = [cpu_float[i] / gpu_float[1] for i in range(6)]

fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)
# ax1.plot(x, cpu_float)
# ax2.plot(x, gpu_float)
# ax3.plot(x1, gpu_float)
# ax4.plot(x1, ratio)
plt.plot(x2, ratio)
plt.xticks(x2, ("1e3", "1e4", "1e5", "1e6", "1e7", "1e8"))
plt.show()
ratio = [cpu_float[i] / gpu_float[1] for i in range(6)]

fig2 = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)
# ax1.plot(x, cpu_float)
# ax2.plot(x, gpu_float)
# ax3.plot(x1, gpu_float)
# ax4.plot(x1, ratio)
plt.plot(x2, gpu)
plt.xticks(x2, ("1e3", "1e4", "1e5", "1e6", "1e7", "1e8"))
plt.show()
