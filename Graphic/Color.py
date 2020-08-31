import numpy as np


class HVColorSpaceGenerator:

    def __new__(cls, h_num=256, v_num=256):
        lut = np.ones((v_num, h_num, 3))

        for i in range(h_num):
            h = i / h_num * 360
            lut[:, i, :] = gen_hue(h, sine)

        for v in range(v_num):
            lut[v, :, :] *= v / (v_num-1)

        return lut


def gen_hue(angle, basis):
    r = basis(angle)
    g = basis(angle - 120)
    b = basis(angle - 240)
    return [r, g, b]


def linear(angle):
    angle %= 360
    if 0 <= angle < 60:
        return 1
    elif 60 <= angle < 120:
        return (120 - angle) / 60
    elif 120 <= angle < 180:
        return 0
    elif 180 <= angle < 240:
        return 0
    elif 240 <= angle < 300:
        return (angle - 240) / 60
    elif 300 <= angle < 360:
        return 1


def sine(angle):
    angle %= 360
    if 0 <= angle < 60:
        return 1
    elif 60 <= angle < 120:
        return (1 - np.cos(np.pi * angle / 60)) / 2
    elif 120 <= angle < 180:
        return 0
    elif 180 <= angle < 240:
        return 0
    elif 240 <= angle < 300:
        return (1 - np.cos(np.pi * angle / 60)) / 2
    elif 300 <= angle < 360:
        return 1
