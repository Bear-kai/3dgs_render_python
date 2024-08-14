#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/17 11:13:25
@ Author   : sunyifan
@ Version  : 1.0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# get (h, w, 3) cavas
def create_canvas(h, w):
    return np.zeros((h, w, 3))


def get_model_matrix(angle):
    angle *= np.pi / 180
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


# from world to camera --> eye_pose是cam在world下的坐标，所以平移量需要添加负号，才对应world2cam的变换！
def get_view_matrix(eye_pose):
    return np.array(
        [
            [1, 0, 0, -eye_pose[0]],
            [0, 1, 0, -eye_pose[1]],
            [0, 0, 1, -eye_pose[2]],
            [0, 0, 0, 1],
        ]
    )


# get projection, including perspective and orthographic
def get_proj_matrix(fov, aspect, near, far):
    t2a = np.tan(fov / 2.0)     # gluPerspective风格的透视变换，根据fov确定焦距f：t2a = 1/f
    return np.array(            # 注意有0<near<far，透视变换矩阵将右手系的cam space，转到左手系的NDC space!
        [
            [1 / (aspect * t2a), 0, 0, 0],
            [0, 1 / t2a, 0, 0],
            [0, 0, (near + far) / (near - far), 2 * near * far / (near - far)],
            [0, 0, -1, 0],
        ]
    )


def get_viewport_matrix(h, w):  # 将左手系的NDC space转到img plane，所以这里的图像原点在左下角-->对应OpenGL风格的图像坐标系
    return np.array(
        [[w / 2, 0, 0, w / 2], [0, h / 2, 0, h / 2], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


if __name__ == "__main__":
    H, W = 700, 700
    frame = create_canvas(H, W)
    angle = 0
    eye = [0, 0, 5]     # cam在world下的坐标！

    pts = [[2, 0, -2], [0, 2, -2], [-2, 0, -2]]                     # 项目原始例子
    # pts = [[4, 0, -2], [0, 4, -2], [-4, 0, -2]]                   # x,y方向增大物体，观察成像变化
    # pts = [[6, 0, -2], [0, 6, -2], [-6, 0, -2]]                   # x,y方向继续增大物体，观察成像变化
    # pts = [[6, 0, -6], [0, 6, -6], [-6, 0, -6]]                   # 增大物体使成像变大，增大物体深度使成像变小

    # pts = [[6, 0, -51+5], [0, 6, -51+5], [-6, 0, -51+5]]          # far映射到NDC的+1，深度超过far会被剔除掉，非要映射则是[非常接近但大于1]
    # pts = [[6, 0, -0.1+5], [0, 6, -0.1+5], [-6, 0, -0.1+5]]       # near映射到NDC的-1，深度小于near会被剔除掉，非要映射则是[明显小于-1]
    # pts = [[6, 0, -0.05+5], [0, 6, -0.05+5], [-6, 0, -0.05+5]]    

    viewport = get_viewport_matrix(H, W)

    # get mvp matrix
    mvp = get_model_matrix(angle)
    mvp = np.dot(get_view_matrix(eye), mvp)
    mvp = np.dot(get_proj_matrix(45, 1, 0.1, 50), mvp)  # 4x4

    # loop points
    pts_2d = []
    for i, p in enumerate(pts):
        # mvp transformation
        p = np.array(p + [1])  # 3x1 -> 4x1
        p = np.dot(mvp, p)

        # 自增：clipping （但这里不是真的丢掉此点）
        if abs(p[0]) > abs(p[3]) or abs(p[1]) > abs(p[3]) or abs(p[2]) > abs(p[3]):
            print("outside range: ", pts[i], ' --mvp--> ', p)

        # perspective devision
        p /= p[3]

        # viewport
        p = np.dot(viewport, p)[:2]
        pts_2d.append([int(p[0]), H-int(p[1])])     # H-y, 将OpenGL风格的图像坐标系，转为OpenCV风格，否则后面cv2显示的三角形是倒立的！

    vis = 1
    if vis:
        # visualize 3d
        fig = plt.figure()
        pts = np.array(pts)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        # https://matplotlib.org/stable/plot_types/3D/trisurf3d_simple.html
        ax = fig.add_subplot(111, projection = '3d')   # ax = Axes3D(fig)
        ax.scatter(x, y, z, s=80, marker="^", c="g")   # "^" is triangle_up , 7 is caretdown
        ax.scatter([eye[0]], [eye[1]], [eye[2]], s=180, marker=7, c="r")
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, alpha=0.5)
        plt.show()

        # visualize 2d
        c = (255, 255, 255)
        for i in range(3):
            for j in range(i + 1, 3):
                cv2.line(frame, pts_2d[i], pts_2d[j], c, 2)

        cv2.imshow("screen", frame)
        cv2.waitKey(0)
