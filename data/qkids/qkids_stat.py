#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: qkids_stat.py
@time: 2024/12/19 15:44
@desc:
"""

import numpy as np

stat = {
    "right_master":
        {'mean': [33.3797699, -17.85713797, 71.77358903, 10.2435088, -90.92793028, 15.7188264, 1599.68513374],
         'std': [19.80147228, 15.35986512, 27.46804917, 21.27188275, 3.49398177, 15.93463831, 806.48648815],
         'max': [86.331, 41.91, 150.817, 69.411, -61.578, 92.586, 3350.],
         'min': [-9.826, -66.719, -9.157, -47.873, -114.594, -60.527, 268.]},
    "right_puppet":
        {'mean': np.array(
            [33.48150376, -17.91478238, 72.09764807, 10.49017056, -90.93237872, 15.70857256, 1622.94954054]),
            'std': np.array(
                [19.83900605, 15.37413166, 27.42137184, 21.16549558, 3.47063697, 15.88744476, 788.56620906]),
            'max': np.array([86.018, 40.794, 145.584, 66.479, -61.641, 92.465, 3348.]),
            'min': np.array([-9.805, -66.797, -8.956, -47.034, -114.592, -60.467, 337.])
        }
}


class Normalizer:

    def __init__(self):
        self.stat = stat
        self.mean = stat['right_puppet']['mean']
        self.std = stat['right_puppet']['std']
        self.max = stat['right_puppet']['max']
        self.min = stat['right_puppet']['min']

    # def encode(self, x):
    #     return (x - self.mean) / self.std
    #
    # def decode(self, x):
    #     return x * self.std + self.mean

    # encode data to [-1, 1]
    def encode(self, x):
        if len(x) == 7:
            out = 2 * (x - self.min) / (self.max - self.min) - 1
            out[6] = x[6] / 3350
            return out
        else:
            raise Exception

    def decode(self, x):
        if len(x) == 7:
            out = (x + 1) / 2 * (self.max - self.min) + self.min
            out[6] = x[6] * 3350
            return out
        else:
            raise Exception
