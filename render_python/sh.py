#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

"""
@ Description:
@ Date     : 2024/05/22 14:19:32
@ Author   : sunyifan
@ Version  : 1.0
"""

import numpy as np

#【存疑】 这个正负号是如何确定的，为何wiki的表格中没有负号？！
# https://zh.wikipedia.org/wiki/%E7%90%83%E8%B0%90%E5%87%BD%E6%95%B0
SH_C0 = 0.28209479177387814     # math.sqrt((1/4/math.pi))
SH_C1 = 0.4886025119029199      # math.sqrt((3/4/math.pi)), m=-1,0,1分别对应-,+,-
SH_C2 = [
    1.0925484305920792,         # m=-2,  math.sqrt((15/math.pi))/2
    -1.0925484305920792,        # m=-1, -math.sqrt((15/math.pi))/2
    0.31539156525252005,        # m=0,   math.sqrt((5/math.pi))/4
    -1.0925484305920792,        # m=1,  -math.sqrt((15/math.pi))/2
    0.5462742152960396,         # m=2,   math.sqrt((15/math.pi))/4
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def computeColorFromSH(deg, pos, campos, sh):
    # The implementation is loosely based on code for
    # "Differentiable Point-Based Radiance Fields for
    # Efficient View Synthesis" by Zhang et al. (2022)
    # [ref1]: https://github.com/sjtuzq/point-radiance/tree/main/modules/sh.py
    # [ref2]: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu

    dir = pos - campos                  # in world
    dir = dir / np.linalg.norm(dir)     # 归一化方向，使颜色跟观察方向相关！

    result = SH_C0 * sh[0]

    if deg > 0:
        x, y, z = dir
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]

        if deg > 1:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            result = (
                result
                + SH_C2[0] * xy * sh[4]
                + SH_C2[1] * yz * sh[5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6]
                + SH_C2[3] * xz * sh[7]
                + SH_C2[4] * (xx - yy) * sh[8]
            )

            if deg > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3.0 * xx - yy) * sh[9]
                    + SH_C3[1] * xy * z * sh[10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13]
                    + SH_C3[5] * z * (xx - yy) * sh[14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]
                )
    result += 0.5                               # 原出处[ref1]没有+0.5，3DGS[ref2]中加了！
    return np.clip(result, a_min=0, a_max=1)    # 颜色值确保在[0,1]区间！


if __name__ == "__main__":
    deg = 3
    pos = np.array([2, 0, -2])
    campos = np.array([0, 0, 5])
    sh = np.random.random((16, 3))
    color = computeColorFromSH(deg, pos, campos, sh)
    print(color)