import time
import math
import arm_six_axis as am

class gripper(am.arm):
    id_num = 7  # 配置手爪ID号与控制手爪的一体化关节ID号一致
    d = 10  # 内部齿轮直径，单位 mm
    def __int__(self, L_p=0, L_p_mass_center=0, G_p=0, com='', uart_baudrate=115200):
        am.arm.__init__(self, L_p=L_p, L_p_mass_center=L_p_mass_center, G_p=G_p, com=com, uart_baudrate=uart_baudrate)

    def grasp(self, wideth=10, speed=10, force=3):
        '''
        :param wideth: 手爪开合宽度，单位 mm
        :param speed: 手爪开合速度，单位 mm/s
        :param force: 手爪开合力，单位 N
        :return: 无
        '''
        if wideth > 85 or wideth < 0:
            print("请输入正确的开合宽度：0~90，已将 wideth 设置为 90")
            wideth = 85
        if speed > 60 or speed <= 0:
            print("请输入正确的开合速度：0~60，已将 speed 设置为 60")
            speed = 60
        # if force < 120:
        #     print("请输入正确的开合力：>120，已将 force 设置为 120")
        #     force = 120
        angle = - wideth / 2 / (self.d / 2) / math.pi * 180  # 将手爪宽度转换成角度
        w = speed / (self.d / 2) / (math.pi * 2) * 60  # 将手爪开合速度转换为关节转速 r/min
        torque = force * (self.d / 2 / 1000)  # 手爪开合里转换成关节力矩 Nm
        self.set_angle_adaptive(id_num=self.id_num, angle=angle, speed=w, torque=torque)
        return True

    def grasp_done(self):
        """检测手爪动作是否完成

        Args:
            无
        Returns:
            无
        Raises:
            无
        """
        self.position_done(id_num=self.id_num)

    def detect_wideth_grasp(self):
        angle = self.get_angle(id_num=self.id_num)
        wideth = - angle / 180 * math.pi * (self.d / 2) * 2
        return wideth





