#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: graper.py
@time: 2024/5/14 11:18
@desc:
"""
from serial import Serial


class Grasper:
    status: int
    OPEN = 1
    CLOSE = 2

    def __init__(self, serial_port: Serial, gid: int, min_v: int = 1350, max_v: int = 1800):
        self._min_v = min_v
        self._max_v = max_v
        self._id = gid
        self._s = serial_port
        self.status = -1
        self.loose()

    def _set_pwm(self, v: str):
        # 000P1800T1000!
        # data = b'#001P' + v.encode() + b'T1000!'
        data = f'#00{self._id}P{v}T0500!'.encode('utf-8')
        # print("send data:", data)
        self._s.write(data)

    def _set(self, ratio: float):
        v = self._min_v + int((self._max_v - self._min_v) * ratio)
        self._set_pwm(str(v))

    def clamp(self):
        if self.status != self.CLOSE:
            self._set(1)
            self.status = self.CLOSE

    def loose(self):
        if self.status != self.OPEN:
            self._set(0)
            self.status = self.OPEN

    def change(self):
        if self.status == self.OPEN:
            self.clamp()
        elif self.status == self.CLOSE:
            self.loose()
