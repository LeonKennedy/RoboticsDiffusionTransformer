import sys
import time
from typing import List, Tuple

from loguru import logger

try:
    import arm_robot as robot
except ModuleNotFoundError:
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    from . import arm_robot as robot


class ServoGroup:

    def __init__(self, dr: robot.arm_robot, id_list: List):
        self.dr = dr
        self.id_list = id_list

    def free(self):
        self.dr.set_torques(self.id_list, [0] * len(self.id_list))
        # return self.dr.free()

    def lock(self):
        return self.dr.lock()

    def _check(self, sid: int):
        if sid in self.id_list:
            return True
        else:
            logger.warning(f"servo {sid} not in {self.id_list}")
            return False

    def set_zero_position(self):
        for i in self.id_list:
            self.dr.set_zero_position(i)
            time.sleep(1)

    def move_to(self, angles: List, bit_width: float = 10):
        self.dr.set_angles(self.id_list, angles, 10, bit_width, 0)

    def move_to1(self, angles):
        self.dr.set_angles(self.id_list, angles, 10, 10, 1)

    def set_torque(self, sid: int, val: float):
        if self._check(sid):
            self.dr.set_torque(sid, val)


class Puppet(ServoGroup):
    pass


class Master(ServoGroup):

    def set_end_torque_zero(self):
        assert self.id_list[-1] == 6
        self.set_torque(6, 0)


''''主臂参数'''''
# robot.dimensions #
arm_six_axes_l_1 = 120.933  # 主臂l1杆的长度，单位mm
arm_six_axes_l_2 = 106.066  # 主臂l2杆的长度，单位mm
arm_six_axes_l_3 = 45.1  # 主臂l3杆的长度，单位mm
arm_six_axes_d_3 = 54.37  # 主臂d3杆的长度，单位mm
arm_six_axes_d_4 = 60.85  # 主臂d4杆的长度，单位mm
arm_six_axes_l_p = 129  # 工具参考点到电机输出轴表面的距离，单位mm（所有尺寸参数皆为mm）
arm_six_axes_l_p_mass_center = 50  # 工具（负载）质心到 6 号关节输出面的距离
''' robot.weight '''
arm_six_axes_G_1 = 0.0346  # 主臂杆件2重量，单位kg
arm_six_axes_G_2 = 0.2779  # 主臂关节3重量，单位kg
arm_six_axes_G_3 = 0.0313  # 主臂杆件3重量，单位kg
arm_six_axes_G_4 = 0.4175  # 主臂关节4+关节5重量，单位kg
arm_six_axes_G_5 = 0.2369  # 主臂关节6，单位kg
arm_six_axes_G_p = 0.0723  # 负载重量，单位kg，所有重量单位皆为kg

arm_six_axes_joints_torque_factor_1 = 0.1  # 主臂关节1力矩修正系数
arm_six_axes_joints_torque_factor_2 = 0.75  # 主臂关节2力矩修正系数
arm_six_axes_joints_torque_factor_3 = 0.45  # 主臂关节3力矩修正系数
arm_six_axes_joints_torque_factor_4 = 0.2  # 主臂关节4力矩修正系数
arm_six_axes_joints_torque_factor_5 = 0.1  # 主臂关节5力矩修正系数
arm_six_axes_joints_torque_factor_6 = 0.0  # 主臂关节6力矩修正系数

COM_LEFT = 'COM3'
COM_RIGHT = 'COM5'
UART_baudrate = 115200


def init_dr(com: str) -> robot.arm_robot:
    dr = robot.arm_robot(L_p=arm_six_axes_l_p, L_p_mass_center=arm_six_axes_l_p_mass_center, G_p=arm_six_axes_G_p,
                         com=com,
                         uart_baudrate=UART_baudrate)
    dr.L = [arm_six_axes_l_1, arm_six_axes_l_2, arm_six_axes_l_3, arm_six_axes_d_3, arm_six_axes_d_4 + arm_six_axes_l_p]

    dr.G = [arm_six_axes_G_1, arm_six_axes_G_2, arm_six_axes_G_3, arm_six_axes_G_4, arm_six_axes_G_5, arm_six_axes_G_p]
    dr.torque_factors = [arm_six_axes_joints_torque_factor_1, arm_six_axes_joints_torque_factor_2,
                         arm_six_axes_joints_torque_factor_3, arm_six_axes_joints_torque_factor_4,
                         arm_six_axes_joints_torque_factor_5, arm_six_axes_joints_torque_factor_6]
    return dr


def build_master_and_puppet(com_name: str, master_ids: List[int], puppet_ids: List[int]) -> Tuple[Master, Puppet]:
    dr = init_dr(com_name)
    master = Master(dr, master_ids)
    puppet = Puppet(dr, puppet_ids)
    return master, puppet


if __name__ == '__main__':
    dr = init_dr(COM_LEFT)
