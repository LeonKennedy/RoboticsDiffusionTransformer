from .grasper import build_grasper, Grasper
from .constants import GRASPER_NAME, GRASPER_MOTOR_ID_LEFT, GRASPER_MOTOR_ID_RIGHT


def build_two_grasper():
    graspers = build_grasper(GRASPER_NAME, [GRASPER_MOTOR_ID_LEFT, GRASPER_MOTOR_ID_RIGHT])
    return graspers[0], graspers[1]


def build_left_grasper():
    graspers = build_grasper(GRASPER_NAME, [GRASPER_MOTOR_ID_LEFT])
    return graspers[0]


def build_right_grasper():
    graspers = build_grasper(GRASPER_NAME, [GRASPER_MOTOR_ID_RIGHT])
    return graspers[0]
