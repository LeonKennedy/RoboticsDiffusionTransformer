#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: data_to_images.py
@time: 2024/12/20 15:06
@desc:
"""
import os
import pickle
import cv2
import sys

def main(name: str):
    raw = pickle.load(open(name, 'rb'))
    data = raw['data']
    for i, e in enumerate(data):
        img = e['camera']['TOP']
        cv2.imwrite(f"images/{i}.jpg", img)


if __name__ == '__main__':
    try:
        filename = sys.argv[1]
    except IndexError as e:
        filename = '/mnt/d4t/data/lerobot/put_puzzle/10_09_16_10_31.pkl'
    os.makedirs("images", exist_ok=True)
    main(filename)
