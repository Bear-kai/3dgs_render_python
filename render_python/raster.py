#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/21 11:31:03
@ Author   : sunyifan
@ Version  : 1.0
"""

import numpy as np
from .graphic import getProjectionMatrix


def ndc2Pix(v, S):
    # 2025/05/07：注意3dgs.py中不管用的是哪个版本的mvp投影矩阵，x,y均映射
    # 到[-1,1]范围，返回的像素坐标是opengl风格，即左下角为图像原点且x右y上！
    return ((v + 1.0) * S - 1.0) * 0.5


def in_frustum(p_orig, viewmatrix):
    # bring point to screen space
    p_view = transformPoint4x3(p_orig, viewmatrix)

    if np.abs(p_view[2]) <= 0.2:    # 丢弃距离cam“太近”的点，自行添加abs()，否则这里强制要求cam是左手系（沿着+z方向观察）！
        return None
    return p_view


def transformPoint4x4(p, matrix):
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
            matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
        ]
    )
    return transformed


def transformPoint4x3(p, matrix):
    matrix = np.array(matrix).flatten(order="F")    # 按列拉直
    x, y, z = p
    transformed = np.array(                         # 这里按c++风格将矩阵乘法手动展开了
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        ]
    )
    return transformed


# covariance = RS[S^T][R^T]
def computeCov3D(scale, mod, rot):
    # create scaling matrix
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    )

    # normalize quaternion to get valid rotation
    # we use rotation matrix
    R = rot

    # compute 3d world covariance matrix Sigma
    M = np.dot(R, S)        # 注意这里是R@S，相当于对R的列向量进行缩放！M@point才能实现缩放+旋转！
    cov3D = np.dot(M, M.T)

    return cov3D


def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.

    t = transformPoint4x3(mean, viewmatrix)     # world2cam

    # viewmatrix将world坐标转到cam坐标，下面确保cam坐标不偏离视锥太远 --> 疑：为何不直接限定于视锥范围内？！
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    # 参考games101讲的透视正交变换矩阵M_persp2ortho，投影后的点p'对投影前的点求导，即为雅可比矩阵J！
    # M_persp2ortho = [[n, 0, 0, 0]; [0, n, 0, 0]; [0, 0, n+f, -nf]; [0, 0, 1, 0]];
    # p' = M_persp2ortho @ [x;y;z;1] = [nx/z; ny/z; (n+f)-nf/z; 1];
    # 注意此处的n,f是对应于HxW图像平面的焦距，这样才能把3Dcov从cam的视锥空间，“捏”到跟图像大小一致的
    # 正交投影空间！转换后的协方差，取前两行两列，就直接是图像平面的协方差！
    # 而games101的n,f对应的那个正交投影空间，无所谓大小，因为会继续被转到NDC空间，最后视口变换还原。
    J = np.array(                           
        [
            [focal_x / t[2], 0, -(focal_x * t[0]) / (t[2] * t[2])],
            [0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])],
            [0, 0, 0],
        ]
    )
    # J[:2] = -J[:2]               # 自增：对于OpenGL风格的透视投影矩阵（cam为右手系），这里雅可比J要反号 --> 但不影响协方差结果！
    
    W = viewmatrix[:3, :3]         # W将cov3D从world转到cam的视锥空间，继而用J将其转到cam的正交投影的视窗！
    T = np.dot(J, W)               # cov3D的前2行2列，就是图像平面的cov2D协方差的样子！

    cov = np.dot(T, cov3D)      
    cov = np.dot(cov, T.T)

    # Apply low-pass filter
    # Every Gaussia should be at least one pixel wide/high  ==> 存疑：有待加深理解...
    # Discard 3rd row and column
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    return [cov[0, 0], cov[0, 1], cov[1, 1]]


if __name__ == "__main__":
    p = [2, 0, -2]
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    projmatrix = getProjectionMatrix(**proj_param)
    transformed = transformPoint4x4(p, projmatrix)
    print(transformed)
