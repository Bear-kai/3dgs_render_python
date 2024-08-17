#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/20 17:20:00
@ Author   : sunyifan
@ Version  : 1.0
"""

import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from math import sqrt, ceil
import matplotlib.pyplot as plt
import cv2

from render_python import computeColorFromSH
from render_python import computeCov2D, computeCov3D
from render_python import transformPoint4x4, in_frustum, transformPoint4x3
from render_python import getWorld2View2, getProjectionMatrix, ndc2Pix, in_frustum, \
                          getProjectionMatrix_games101, getProjectionMatrix_opengl
from matplotlib_vis import vis_ellipsoid, generate_random_rotation_matrix


class Rasterizer:
    def __init__(self) -> None:
        pass

    def forward(
        self,
        P,  # int, num of guassians
        D,  # int, degree of spherical harmonics
        M,  # int, num of sh base function
        background,  # color of background, default black
        width,  # int, width of output image
        height,  # int, height of output image
        means3D,  # ()center position of 3d gaussian
        shs,  # spherical harmonics coefficient
        colors_precomp,
        opacities,  # opacities
        scales,  # scale of 3d gaussians
        scale_modifier,  # default 1
        rotations,  # rotation of 3d gaussians
        cov3d_precomp,
        viewmatrix,  # matrix for view transformation
        projmatrix,  # *(4, 4), matrix for transformation, aka mvp
        cam_pos,  # position of camera
        tan_fovx,  # float, tan value of fovx   --> 对应像平面在[-1,1]范围的焦距fx的倒数
        tan_fovy,  # float, tan value of fovy   
        prefiltered,
    ) -> None:

        focal_y = height / (2 * tan_fovy)  # focal of y axis --> 这里转为像平面在[-H/2,H/2]范围的焦距
        focal_x = width / (2 * tan_fovx)

        # run preprocessing per-Gaussians
        # transformation, bounding, conversion of SHs to RGB
        logger.info("Starting preprocess per 3d gaussian...")
        preprocessed = self.preprocess(
            P,
            D,
            M,
            means3D,
            scales,
            scale_modifier,
            rotations,
            opacities,
            shs,
            viewmatrix,
            projmatrix,
            cam_pos,
            width,
            height,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        )

        # produce [depth] key and corresponding guassian indices
        # sort indices by depth
        depths = preprocessed["depths"]
        point_list = np.argsort(depths)

        # render
        logger.info("Starting render...")
        out_color = self.render(
            point_list,
            width,
            height,
            preprocessed["points_xy_image"],
            preprocessed["rgbs"],
            preprocessed["conic_opacity"],
            background,
        )

        # 以下新增返回参数
        extra_return = dict(
            R_2orth = preprocessed["R_2orth"],
            angle2D = preprocessed["angle2D"],
            axes2D_length = preprocessed["axes2D_length"],
            xy_orth = preprocessed["xy_orth"],
            uv_img = preprocessed["points_xy_image"]
        )

        return out_color, extra_return

    def preprocess(
        self,
        P,
        D,
        M,
        orig_points,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        viewmatrix,
        projmatrix,
        cam_pos,
        W,
        H,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
    ):

        rgbs = []             # rgb colors of gaussians
        cov3Ds = []           # covariance of 3d gaussians
        depths = []           # depth of 3d gaussians after view&proj transformation
        radii = []            # radius of 2d gaussians
        conic_opacity = []    # covariance inverse of 2d gaussian and opacity
        points_xy_image = []  # mean of 2d guassians

        R_2orth = []
        angle2D = []
        axes2D_length = []
        xy_orth = []

        for idx in range(P):
            # make sure point in frustum
            p_orig = orig_points[idx]
            p_view = in_frustum(p_orig, viewmatrix)
            if p_view is None:
                continue
            depths.append(p_view[2])

            # transform point, from world to ndc
            # Notice, projmatrix already processed as mvp matrix
            p_hom = transformPoint4x4(p_orig, projmatrix)
            p_w = 1 / (p_hom[3] + 0.0000001)
            p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]

            # compute 3d covarance by scaling and rotation parameters
            scale = scales[idx]
            rotation = rotations[idx]
            cov3D = computeCov3D(scale, scale_modifier, rotation)   # in world space
            cov3Ds.append(cov3D)

            # compute 2D screen-space covariance matrix
            # based on splatting, -> JW Sigma W^T J^T
            cov = computeCov2D(
                p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
            )

            # invert covarance(EWA splatting)
            det = cov[0] * cov[2] - cov[1] * cov[1]
            if det == 0:
                depths.pop()
                cov3Ds.pop()
                continue
            det_inv = 1 / det
            conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]
            conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]])

            # compute radius, by finding eigenvalues of 2d covariance --> 令det(cov-lambda*eye)=0, 再用二次求根公式得出两个根即为lambda
            # transfrom point from NDC to Pixel
            mid = 0.5 * (cov[0] + cov[2])
            lambda1 = mid + sqrt(max(0.1, mid * mid - det))
            lambda2 = mid - sqrt(max(0.1, mid * mid - det))
            my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
            point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)]

            radii.append(my_radius)
            points_xy_image.append(point_image)

            # convert spherical harmonics coefficients to RGB color
            sh = shs[idx]
            result = computeColorFromSH(D, p_orig, cam_pos, sh)
            rgbs.append(result)

            # 自增：计算正交投影空间下3D协方差对应的"假旋转"R_2orth，假设轴长S不变；
            nx, ny = 2/tan_fovx, 2/tan_fovy                          # 计算像平面在[-2,2]范围的焦距
            t = transformPoint4x3(p_orig, viewmatrix)                # world2cam
            Jacobi = np.array(                                       # 该雅可比，对应于将cam视锥空间转换到(正负2范围的)正交投影空间的变换
                [                                                    # 将J[2,2]改为实际偏导数 nf/(z*z)，注意3dgs.py中有设置znear=0.01,zfar=100，
                    [nx / t[2], 0, -(nx * t[0]) / (t[2] * t[2])],    # 这里姑且认为远平面不变，即 f=zfar，再令 n=(focal_x + focal_y)/2
                    [0, ny / t[2], -(ny * t[1]) / (t[2] * t[2])],
                    [0, 0, 50*(nx+ny) / (t[2] * t[2])],              # [0, 0, 0],    
                ]
            )
            Jacobi[2,2] /= 5                                         # 手动调整，方便可视化，渲染仅关注z的相对大小-->光栅化时深度排序
            R_2orth.append( Jacobi @ viewmatrix[:3, :3] @ rotation ) # 由于Jacobi非正交阵，故R_2orth非旋转矩阵，即正交投影空间中，原椭球已被"变形"了！
            # 自增：计算投影到图像平面上2D协方差对应的旋转+轴长
            cov2D = np.array([[cov[0], cov[1]], [cov[1], cov[2]]])
            eigval, eigvec = np.linalg.eig(cov2D)
            angle2D.append( np.arctan2(eigvec[1,0], eigvec[0,0]) )
            axes2D_length.append( (int(round(np.sqrt(5.991*eigval[0]))),   
                                   int(round(np.sqrt(5.991*eigval[1])))) )  # 绘制95%置信区间（自由度为2的卡方分布95%置信度小于5.991）
            xy_orth.append(2*np.array([p_proj[0], p_proj[1], -1.]))         # 将椭球中心从world投影到(正负2范围的)正交投影空间，这里统一设置深度值为-2

        return dict(             # 本demo中未用到cov3Ds和radii
            rgbs=rgbs,
            cov3Ds=cov3Ds,
            depths=depths,
            radii=radii,
            conic_opacity=conic_opacity,
            points_xy_image=points_xy_image,

            # 以下新增返回参数
            R_2orth = R_2orth,
            angle2D = angle2D,
            axes2D_length = axes2D_length,
            xy_orth = xy_orth
        )

    def render(
        self, point_list, W, H, points_xy_image, features, conic_opacity, bg_color
    ):

        out_color = np.zeros((H, W, 3))
        pbar = tqdm(range(H * W))

        # loop pixel
        for i in range(H):
            for j in range(W):
                pbar.update(1)
                pixf = [i, j]
                C = np.zeros(3)  # [0, 0, 0]
                T = 1            # corner case: no 3D gaussian point, then get final color as bg_color directly

                # loop gaussian
                for idx in point_list:

                    # init helper variables, transmirrance --> 透明度
                    T = 1

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    xy = points_xy_image[idx]  # center of 2d gaussian
                    d = [
                        xy[0] - pixf[0],
                        xy[1] - pixf[1],
                    ]  # distance from center of pixel
                    con_o = conic_opacity[idx]
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )                                            # 二维高斯概率密度函数推导：https://www.cnblogs.com/kailugaji/p/15542845.html
                    if power > 0:                                # power势必非负，这里起到assert作用
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    alpha = min(0.99, con_o[3] * np.exp(power))  # con_o[3]对应该高斯的opacity, exp(power)是该2D高斯对该像素的影响程度！
                    if alpha < 1 / 255:                          # 该高斯越不透明，距离该像素越近(即影响越大)，则alpha越大，否则此像素将跳过该高斯！
                        continue
                    test_T = T * (1 - alpha)                     # 更新透明度
                    if test_T < 0.0001:
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    color = features[idx]
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * T           # 类似NeRF中的体积渲染，sum累加{"前面点的透明度 x 当前的不透明度 x 当前颜色"}

                    T = test_T

                # get final color
                for ch in range(3):                              # 对于透明度高的地方，用背景颜色填充
                    out_color[j, i, ch] = C[ch] + T * bg_color[ch]

        return out_color


if __name__ == "__main__":
    np.random.seed(666)

    # set guassian
    pts = np.array([[2, 0, -2], [2, 2, -2], [-3, 0, -2]])         # nx3, in world coord  
    n = len(pts)
    shs = np.random.random((n, 16, 3))
    opacities = np.ones(n)
    scales = np.array([[1,1.5,1],[1.5,1,1],[1,0.5,0.5]])          # 0.5*np.ones((n, 3))   #  
    rotations = np.array([np.eye(3)] * n)                         # nx3x3
    rotations = np.stack([generate_random_rotation_matrix(),
                          generate_random_rotation_matrix(),
                          generate_random_rotation_matrix(),])

    # 新增椭球的可视化  ellipsoid in world space
    vis_ellipsoid(radius_arr=scales, R_arr=rotations, t_arr=pts)
    
    # set params
    H, W = 400, 400
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    cam_pos = np.array([0, 0, 5])                               # in world

    # 1. cam左手系（物体在cam的+z方向），透视投影矩阵基于左手系推导
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])            # cam2world
    viewmatrix = getWorld2View2(R=R, t=cam_pos)                 # world2cam
    # projmatrix = getProjectionMatrix(**proj_param)            # 推导假设：cam为左手系, near --> 0, far --> 1
    projmatrix = getProjectionMatrix_games101(**proj_param)     # 推导假设：cam为左手系, near --> 1, far --> -1

    # 2. cam右手系（物体在cam的-z方向），透视投影矩阵基于右手系推导
    # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])             # cam2world  
    # viewmatrix = getWorld2View2(R=R, t=cam_pos)                 # world2cam    
    # projmatrix = getProjectionMatrix_opengl(**proj_param)       # 推导假设：cam为右手系, near --> -1, far --> 1 
     
    # 3. cam左手系（物体在cam的+z方向），透视投影矩阵基于右手系推导  --> 错误匹配时的成像，与正常成像"中心对称"！
    # R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])            
    # viewmatrix = getWorld2View2(R=R, t=cam_pos)                 # world2cam
    # projmatrix = getProjectionMatrix_opengl(**proj_param)       # 推导假设：cam为右手系, near --> -1, far --> 1 

    # compute mvp transformation   
    projmatrix = np.dot(projmatrix, viewmatrix)                 # mvp, world2NDC 
    tanfovx = math.tan(proj_param["fovX"] * 0.5)
    tanfovy = math.tan(proj_param["fovY"] * 0.5)

    # render
    rasterizer = Rasterizer()
    out_color, extra_return = rasterizer.forward(
        P=len(pts),
        D=3,
        M=16,
        background=np.array([0, 0, 0]),
        width=W,
        height=H,
        means3D=pts,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        scale_modifier=1,
        rotations=rotations,
        cov3d_precomp=None,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        cam_pos=cam_pos,
        tan_fovx=tanfovx,
        tan_fovy=tanfovy,
        prefiltered=None,
    )

    # plt.imshow(out_color)
    # plt.show()

    # 新增椭球和椭圆的可视化
    vis_ellipsoid(radius_arr=scales, 
                  R_arr=extra_return["R_2orth"], 
                  t_arr=extra_return["xy_orth"])  # ellipsoid in 对应图像尺寸的正交投影空间
    for uv, ax_length, ax_angle in zip(extra_return["uv_img"], 
                                       extra_return["axes2D_length"], 
                                       extra_return["angle2D"]):
        uv_int = (int(uv[0]),int(uv[1]))
        out_color = cv2.ellipse(out_color, uv_int, ax_length, 180*ax_angle/np.pi, 0, 360, (255,0,0), 2) 


    cv2.imshow("screen", np.flipud(out_color))    # flipud将OpenGL风格的图像坐标系，转为OpenCV风格
    cv2.waitKey(0)
