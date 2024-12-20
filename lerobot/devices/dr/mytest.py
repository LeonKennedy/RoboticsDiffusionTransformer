import time
import arm_robot as robot  # 忽略这里的报错
import sys
from io import StringIO

uart_baudrate = 115200  # 串口波特率，与CAN模块的串口波特率一致
com = 'COM5'  # 在这里输入 COM 端口号（Windows 系统）
'''注意：主臂从下到上直到手爪电机 ID 号必须是依次 1,2,3,4,5,6,7'''
id_list_master = [1, 2, 3, 4, 5, 6]  # 设定主臂关节电机的 ID 号（含爪子）
id_list_slaver = [7, 8, 9, 10, 11, 12]  # 设定从臂关节电机的 ID 号（含爪子）


def init_dr():
    dr = robot.arm_robot(L_p=arm_six_axes_l_p, L_p_mass_center=arm_six_axes_l_p_mass_center, G_p=arm_six_axes_G_p,
                         com=com,
                         uart_baudrate=uart_baudrate)
    dr.L = [arm_six_axes_l_1, arm_six_axes_l_2, arm_six_axes_l_3, arm_six_axes_d_3, arm_six_axes_d_4 + arm_six_axes_l_p]

    dr.G = [arm_six_axes_G_1, arm_six_axes_G_2, arm_six_axes_G_3, arm_six_axes_G_4, arm_six_axes_G_5, arm_six_axes_G_p]
    dr.torque_factors = [arm_six_axes_joints_torque_factor_1, arm_six_axes_joints_torque_factor_2,
                         arm_six_axes_joints_torque_factor_3, arm_six_axes_joints_torque_factor_4,
                         arm_six_axes_joints_torque_factor_5, arm_six_axes_joints_torque_factor_6]
    return dr


'''先将两台臂运动到合适姿态（也可以在开机前摆到合适位置）'''

# angles_init = [0, 15, -90, 0, 0, 0]
# dr.set_angles(id_list=id_list_master[:6], angle_list=angles_init, speed=10, param=10, mode=1)
# time.sleep(1)
# dr.set_angles(id_list=id_list_slaver[:6], angle_list=angles_init, speed=10, param=10, mode=1)
# time.sleep(5) # 等待运动到合适位置
#
# dr.set_torques(id_list=[id_list_master[5], id_list_master[6]], torque_list=[0, 0], param=0, mode=0) # 放松主臂6号关节和手抓电机
# dr.set_torque_limit(id_num=id_list_slaver[6], torque_limit=0.3) # 设置从臂手爪的最大夹持力

max_duration = 0


def get_all():
    global max_duration
    st = time.time()
    while (time.time() - st < 100):
        id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        angle_speed_torque_list = dr.get_angle_speed_torque_all(id_list=id_list)
        if angle_speed_torque_list:
            duration = time.time() - st
            if duration > max_duration:
                max_duration = duration
            print("duration:", duration, "max", max_duration)
            angles = [i[0] for i in angle_speed_torque_list]
            master, puppet = angles[:6], angles[6:]
            print('[M]', master, '\n[P]', puppet)
            return master, puppet
        else:
            time.sleep(0.002)
    raise Exception("out of time")


def set_zero_master():
    for i in id_list_master:
        dr.set_zero_position(i)
        time.sleep(1)


def set_zero_puppet():
    for i in id_list_slaver:
        dr.set_zero_position(i)
        time.sleep(1)


def free_puppet():
    dr.set_torques(id_list_slaver, [0, 0, 0, 0, 0, 0], mode=0)


def master_gravity():
    dr.set_torque(6, 0, param=0, mode=0)
    while 1:
        angle_list = get_all()
        master_angles, puppet_angles = angle_list[:6], angle_list[6:]
        dr.gravity_compensation(pay_load=0, F=[0, 0, 0], angle_list=master_angles)  # 主臂重力补偿


tmp_buffer = StringIO()


def gravity(angles):
    sys.stdout = tmp_buffer
    dr.gravity_compensation(pay_load=0, F=[0, 0, 0], angle_list=angles)
    sys.stdout = sys.__stdout__


def follow():
    '''开始主从操作'''
    N = 0
    dr.set_torque(6, 0, param=0, mode=0)
    time.sleep(3)
    id_list = id_list_master + id_list_slaver
    print("id_list:", id_list)
    start = time.time()
    while (time.time() - start < t):
        angle_speed_torque_list = dr.get_angle_speed_torque_all(id_list=id_list)
        if angle_speed_torque_list is None:
            time.sleep(0.001)
        else:
            angle_list = [i[0] for i in angle_speed_torque_list]
            print('[M]', angle_list[:6], '\n[P]', angle_list[6:])
            # dr.gravity_compensation(pay_load=0, F=[0, 0, 0], angle_list=angle_list[:6])
            gravity(angle_list[:6])
            dr.set_angles(id_list=id_list_slaver, angle_list=angle_list[:6], speed=10, param=10, mode=0)
            '''适当调整pid后可使用下面的代码'''
            # bit_wideth1 = 1 / (time.time() - start_over) / 2 # 计算在 t>n 情况下的指令发送频率的一半
            # dr.set_angles(id_list=id_list_slaver, angle_list=slaver_angle_list, speed=20, param=bit_wideth1, mode=0)
            # start_over = time.time()
            '''适当调整pid后可使用上面的代码'''
            N += 1
            time.sleep(0.002)

    print(N, "Done!")
    print("FPS", N / (time.time() - start))


def set_puppet(angles):
    dr.set_angles(id_list_slaver, angles, speed=10, param=10, mode=0)


def one_step():
    master_angle, puppet_angle = get_all()
    dr.gravity_compensation(pay_load=0, F=[0, 0, 0], angle_list=master_angle)
    set_puppet(master_angle)


def init_position():
    ## [M] [-79.387, -0.319, 43.826, -23.292, -87.247, 7.581]
    ## [P] [-80.105, -2.746, 42.464, -22.826, -86.682, 7.581]
    dr.set_angles(id_list_master, [-80, -10, 30, -22, -86, 0], 10, 10, 1)
    time.sleep(1)
    dr.set_angles(id_list_slaver, [-80, -10, 30, -22, -86, 0], 10, 10, 1)


if __name__ == '__main__':
    t = 100  # 主从操作执行时间，单位s
    dr = init_dr()
