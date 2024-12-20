#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: compute_stat.py
@time: 2024/12/19 15:16
@desc:
"""
import os
import glob
import pickle

import numpy as np


def collect_task_qpos(f: str):
    print('[TASK]', f)
    right_master, right_puppet = [], []
    for epi in glob.glob(os.path.join(f, "*.pkl")):
        print('[EPISODE]', epi)
        data = pickle.load(open(epi, 'rb'))
        for e in data['data']:
            right_master.append(e['right_master'])
            right_puppet.append(e['right_puppet'])

    return {"right_master": np.stack(right_master), "right_puppet": np.stack(right_puppet)}


def show(data):
    for k, v in data.items():
        print(k, v.shape)
        stat = {'mean': np.mean(v, axis=0), 'std': np.std(v, axis=0), 'max': np.max(v, axis=0),
                'min': np.min(v, axis=0)}
        print(stat)


def run():
    data = {"right_master": [], "right_puppet": []}
    for i in glob.glob(os.path.join(raw_data_path, "*")):
        task_data = collect_task_qpos(i)
        for k, v in task_data.items():
            data[k].append(v)

    con_data = {}
    for k, v in data.items():
        con_data[k] = np.concatenate(v)

    show(con_data)


if __name__ == '__main__':
    raw_data_path = "/mnt/d4t/data/lerobot"
    run()
