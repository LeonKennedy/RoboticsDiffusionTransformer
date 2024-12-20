#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: create_hdf5.py
@time: 2024/12/6 14:23
@desc:
"""

import glob
import os
import cv2
import h5py
import pickle
import numpy as np
from qkids_stat import Normalizer


def _preprocess(raw):
    master = []
    puppet = []
    image_top = []
    image_right = []
    data = raw['data']
    for e in data:
        master.append(norm.encode(e['right_master']))
        puppet.append(norm.encode(e['right_puppet']))
        image_right.append(cv2.cvtColor(e['camera']['RIGHT'], cv2.COLOR_BGR2RGB))
        image_top.append(cv2.cvtColor(e['camera']['TOP'], cv2.COLOR_BGR2RGB))

    image_right = np.stack(image_right)
    # image_right = np.moveaxis(image_right, -1, 1)

    image_top = np.stack(image_top)
    # image_top = np.moveaxis(image_top, -1, 1)
    return np.stack(master), np.stack(puppet), image_top, image_right


def process_episode(episode, writer):
    act, state, img_top, img_right = _preprocess(episode)

    action = writer.create_dataset("action", act.shape)
    action[...] = act

    obs = writer.create_group("observations")
    qpos = obs.create_dataset("qpos", state.shape)
    qpos[...] = state

    img = obs.create_group("images")
    cam_high = img.create_dataset("cam_high", img_top.shape, dtype=np.uint8)
    cam_high[...] = img_top

    cam_right = img.create_dataset("cam_right_wrist", img_right.shape, dtype=np.uint8)
    cam_right[...] = img_right


def process_task(path: str):
    print(path)
    hdf5_path = os.path.join(output_dir, os.path.basename(path))
    os.makedirs(hdf5_path, exist_ok=True)
    for epi in glob.glob(os.path.join(path, "*.pkl")):
        out_path = os.path.join(hdf5_path, os.path.basename(epi).replace('.pkl', '.hdf5'))
        # if os.path.exists(out_path):
        #     continue
        data = pickle.load(open(epi, 'rb'))
        print(epi)
        with h5py.File(out_path, "w") as writer:
            process_episode(data, writer)


def run():
    for i in glob.glob(os.path.join(raw_data_path, "*")):
        task_data = process_task(i)


if __name__ == '__main__':
    raw_data_path = "/mnt/d4t/data/lerobot"
    output_dir = "/mnt/d2t/code/RoboticsDiffusionTransformer/data/dataset/qkids"
    norm = Normalizer()
    run()
