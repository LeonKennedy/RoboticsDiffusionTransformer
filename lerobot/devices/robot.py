#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: robot.py
@time: 2024/10/9 11:27
@desc:
"""
import abc
from typing import List

import numpy as np

from .arm import build_right_arm, build_two_arm


class Robot:

    @abc.abstractmethod
    def move_start_position(self, master: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def move_master_to_puppet(self):
        raise NotImplementedError

    @abc.abstractmethod
    def clear_uart(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_end_torque_zero(self):
        raise NotImplementedError

    @abc.abstractmethod
    def follow(self, bit_width: float) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def lock(self):
        raise NotImplementedError

    # eval
    @abc.abstractmethod
    def read_puppet_state(self) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def set_state(self, state, bit_width: float):
        raise NotImplementedError


class SingleArmRobot(Robot):

    def __init__(self):
        self.arm_right = build_right_arm()

    def move_master_to_puppet(self):
        rm, rp = self.arm_right.get_all_angle()
        self.arm_right.master.move_to1(rp)

    def clear_uart(self):
        self.arm_right.clear_uart()

    def set_end_torque_zero(self):
        self.arm_right.master.set_end_torque_zero()

    def move_start_position(self, master: bool):
        self.arm_right.move_start_position(master)

    def follow(self, bit_width: float):
        episode = {}
        right_master_angles, right_trigger_angle, right_puppet_angles, right_grasper_angle = self.arm_right.follow(
            bit_width)
        episode["right_master"] = right_master_angles + [right_trigger_angle]
        episode["right_puppet"] = right_puppet_angles + [right_grasper_angle]
        return episode

    def lock(self):
        self.arm_right.lock()

    def read_puppet_state(self) -> List:
        return self.arm_right.read_puppet_state()

    def set_state(self, state: np.ndarray, bit_width: float):
        angle, grasper = state[:6], state[6]
        self.arm_right.puppet.move_to(angle, bit_width)
        self.arm_right.grasper.set_angle(grasper)


class TwoArmRobot(Robot):

    def __init__(self):
        self.arm_left, self.arm_right = build_two_arm()

    def move_start_position(self, master: bool):
        self.arm_left.move_start_position(master)
        self.arm_right.move_start_position(master)

    def move_master_to_puppet(self):
        lm, lp = self.arm_left.get_all_angle()
        self.arm_left.master.move_to1(lp)
        rm, rp = self.arm_right.get_all_angle()
        self.arm_right.master.move_to1(rp)

    def clear_uart(self):
        if self.arm_left:
            self.arm_left.clear_uart()
        if self.arm_right:
            self.arm_right.clear_uart()

    def set_end_torque_zero(self):
        if self.arm_left:
            self.arm_left.master.set_end_torque_zero()
        if self.arm_right:
            self.arm_right.master.set_end_torque_zero()

    def follow(self, bit_width: float):
        episode = {}
        if self.arm_left:
            left_master_angles, left_trigger_angle, left_puppet_angles, left_grasper_angle = self.arm_left.follow(
                self.bit_width)
            episode["left_master"] = left_master_angles + [left_trigger_angle]
            episode["left_puppet"] = left_puppet_angles + [left_grasper_angle]
        if self.arm_right:
            right_master_angles, right_trigger_angle, right_puppet_angles, right_grasper_angle = self.arm_right.follow(
                self.bit_width)
            episode["right_master"] = right_master_angles + [right_trigger_angle]
            episode["right_puppet"] = right_puppet_angles + [right_grasper_angle]
        return episode


def build_robot(dim: int) -> Robot:
    if dim == 7:
        return SingleArmRobot()
    elif dim == 17:
        return TwoArmRobot()
    else:
        raise NotImplementedError
