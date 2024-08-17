#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/21 11:31:03
@ Author   : sunyifan
@ Version  : 1.0
"""

import math
import numpy as np


def getWorld2View2_old(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()      # R是cam2world
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T                # R,t均为cam2world，这里求逆得到world2cam
    Rt[:3, 3] = -R@t
    Rt[3, 3] = 1.0

    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)      # 推导基于的假设：cam为左手系 --> 由于f-n>0有NDC为左手系
    P[2, 3] = -(zfar * znear) / (zfar - znear)    # 这里 near --> 0, far --> 1
    return P


def getProjectionMatrix_games101(znear, zfar, fovX, fovY):  # 自增
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right                                 # 这么设置bottom/left，则已默认正交投影时无x,y方向平移

    P = np.zeros((4, 4))                                   
    P[0, 0] = 2.0 * znear / (right - left)        # 2n/w
    P[1, 1] = 2.0 * znear / (top - bottom)        # 2n/h
    P[3, 2] = 1.0                                 
    P[2, 2] =  (znear + zfar) / (znear - zfar)
    P[2, 3] = -2.0*znear*zfar / (znear - zfar)    # 推导基于的假设：cam为左手系 --> 由于n-f<0有NDC为右手系
                                                  # 这里 near --> 1, far --> -1
    return P


def getProjectionMatrix_opengl(znear, zfar, fovX, fovY):    # 自增
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right                                 # 这么设置bottom/left，则已默认正交投影时无x,y方向平移

    P = np.zeros((4, 4))                                   
    P[0, 0] = 2.0 * znear / (right - left)        # 2n/W
    P[1, 1] = 2.0 * znear / (top - bottom)        # 2n/H
    P[3, 2] = -1.0                                
    P[2, 2] = (znear + zfar) / (znear - zfar)
    P[2, 3] = 2.0*znear*zfar / (znear - zfar)     # 推导基于的假设：cam为右手系 --> 转到NDC后是左手系
                                                  # 这里 near --> -1, far --> 1
    return P

# 注：左手系更直观，可使远处物体的z值更大，自然也有map(near) > map(far)；
#     注意z值仅用于光栅化depth-buffer，不影响该点的像素位置(不论它是否有显示)。

if __name__ == "__main__":
    p = [2, 0, -2]
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    projmatrix = getProjectionMatrix(**proj_param)
    print(projmatrix)
