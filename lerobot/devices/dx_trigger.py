import math
import time
from typing import Tuple
import enum

from dynamixel_sdk import GroupBulkRead, GroupSyncRead, PortHandler, PacketHandler, COMM_SUCCESS

from .constants import TRIGGER_MOTOR_ID_LEFT, TRIGGER_MOTOR_ID_RIGHT

class ReadAttribute(enum.Enum):
    TEMPERATURE = 146
    VOLTAGE = 145
    VELOCITY = 128
    POSITION = 132
    CURRENT = 126
    PWM = 124
    HARDWARE_ERROR_STATUS = 70
    HOMING_OFFSET = 20
    BAUDRATE = 8


MIN_PWM, MAX_PWM = 50, 800


class Dynamixel:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_VELOCITY_LIMIT = 44
    ADDR_GOAL_PWM = 100
    OPERATING_MODE_ADDR = 11
    POSITION_I = 82
    POSITION_P = 84
    ADDR_ID = 7

    def __init__(self, port_handler: PortHandler):
        self.portHandler = port_handler
        # self.portHandler.LA
        self.packetHandler = PacketHandler(2)
        # if not self.portHandler.setBaudRate(self.config.baudrate):
        #     raise Exception(f'failed to set baudrate to {self.config.baudrate}')

    def __del__(self):
        if self.portHandler.is_open:
            self.portHandler.closePort()

    def _read_value(self, motor_id, attribute: ReadAttribute, num_bytes: int, tries=10):
        try:
            if num_bytes == 1:
                value, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler,
                                                                                     motor_id,
                                                                                     attribute.value)
            elif num_bytes == 2:
                value, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler,
                                                                                     motor_id,
                                                                                     attribute.value)
            elif num_bytes == 4:
                value, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler,
                                                                                     motor_id,
                                                                                     attribute.value)
        except Exception:
            if tries == 0:
                raise Exception
            else:
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        if dxl_comm_result != COMM_SUCCESS:
            if tries <= 1:
                # print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                raise ConnectionError(f'dxl_comm_result {dxl_comm_result} for servo {motor_id} value {value}')
            else:
                print(f'dynamixel read failure for servo {motor_id} trying again with {tries - 1} tries')
                time.sleep(0.02)
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        elif dxl_error != 0:  # # print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            # raise ConnectionError(f'dxl_error {dxl_error} binary ' + "{0:b}".format(37))
            if tries == 0 and dxl_error != 128:
                raise Exception(f'Failed to read value from motor {motor_id} error is {dxl_error}')
            else:
                return self._read_value(motor_id, attribute, num_bytes, tries=tries - 1)
        return value

    def _write_value(self, motor_id, attribute, value):
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, attribute, value)
        if dxl_comm_result != COMM_SUCCESS:
            raise ConnectionError(f'dxl_comm_result {dxl_comm_result} for servo {motor_id} value {value}')
        if dxl_error != 0:
            raise Exception(f'Failed to write value from motor {motor_id} error is {dxl_error}')

    def read_temperature(self, motor_id: int):
        return self._read_value(motor_id, ReadAttribute.TEMPERATURE, 1)

    def read_current(self, motor_id: int):
        current = self._read_value(motor_id, ReadAttribute.CURRENT, 2)
        if current > 2 ** 15:
            current -= 2 ** 16
        return current

    def read_position(self, motor_id: int):
        pos = self._read_value(motor_id, ReadAttribute.POSITION, 4)
        if pos > 2 ** 31:
            pos -= 2 ** 32
        # print(f'read position {pos} for motor {motor_id}')
        return pos

    def read_position_degrees(self, motor_id: int) -> float:
        return (self.read_position(motor_id) / 4096) * 360

    def read_position_radians(self, motor_id: int) -> float:
        return (self.read_position(motor_id) / 4096) * 2 * math.pi

    def write_pwm(self, motor_id: int, value):
        limit_value = min(max(MIN_PWM, value), MAX_PWM)
        self._write_value(motor_id, Dynamixel.ADDR_GOAL_PWM, limit_value)


RANGE = {2: (1700, 1700 + 600), 4: (1700, 1700 + 600)}


class Trigger(Dynamixel):

    def __init__(self, port_handler: PortHandler, motor_id: int):
        super().__init__(port_handler)
        self.motor_id = motor_id
        self.range = RANGE[self.motor_id]

    def read(self) -> float:
        cur = self.position()
        ratio = round((self.range[1] - cur) / (self.range[1] - self.range[0]), 2)
        return max(0, min(1, ratio))

    def position(self) -> int:
        return self.read_position(self.motor_id)

    def set_pwm(self, value):
        self.write_pwm(self.motor_id, round(value))


def build_trigger(device_name="COM10") -> Tuple[Trigger, Trigger]:
    ph = PortHandler(device_name)
    if not ph.openPort():
        raise Exception(f'Failed to open port {device_name}')
    return Trigger(ph, 2), Trigger(ph, 4)


def build_left_trigger(device_name="COM10") -> Trigger:
    ph = PortHandler(device_name)
    if not ph.openPort():
        raise Exception(f'Failed to open port {device_name}')
    return Trigger(ph, TRIGGER_MOTOR_ID_LEFT)


def build_right_trigger(device_name="COM10") -> Trigger:
    ph = PortHandler(device_name)
    if not ph.openPort():
        raise Exception(f'Failed to open port {device_name}')
    return Trigger(ph, TRIGGER_MOTOR_ID_RIGHT)


if __name__ == '__main__':
    # d = Dynamixel("COM10")
    l, r = build_trigger("COM10")
    while 1:
        print("left", l.position(), l.read(), "right", r.position(), r.read())
        time.sleep(1 / 20)
