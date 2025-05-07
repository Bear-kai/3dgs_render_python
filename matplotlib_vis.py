import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def generate_random_rotation_matrix():
    # 随机生成一个四元数，四元数的每个分量都在[-1, 1]范围内
    q = np.random.rand(4) - 0.5
    # 归一化四元数以确保它是单位四元数
    q /= np.linalg.norm(q)
    # 从四元数创建旋转对象
    rotation = Rotation.from_quat(q)
    # 将旋转对象转换为3x3旋转矩阵
    rotation_matrix = rotation.as_matrix()
    
    return rotation_matrix


def get_ellipsoid(radius, R = np.eye(3), t = np.zeros((3,1))):
    rx, ry, rz = radius
    npts=100
    
    phi = np.linspace(0, 2.*np.pi, npts)
    theta =  np.linspace(0, np.pi, npts)
    phi, theta = np.meshgrid(phi, theta)

    # 椭球标准方程：x^2/a^2 + y^2/b^2 + z^2/c^2 = 1，其中a,b,c分别是x,y,z的半轴长度，
    # 也即rx,ry,rz，可知下面x,y,z满足椭球方程！当rx=ry=rz时，下面就是球的极坐标表达！
    x = rx * np.sin(theta) * np.cos(phi)
    y = ry * np.sin(theta) * np.sin(phi)
    z = rz * np.cos(theta)

    xyz = np.stack([x,y,z], axis=0)    # 3xnxn
    xyz = xyz.reshape(3,-1)
    xyz = R @ xyz + t
    xyz = xyz.reshape(3, npts, npts)
    x, y, z = xyz[0], xyz[1], xyz[2]

    return (x, y, z)


def vis_ellipsoid(radius_arr, R_arr, t_arr, stride_scale=3):
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(-6,6), ylim=(-6,6), zlim=(-6,6), 
            xlabel='X', ylabel='Y', zlabel='Z', 
            aspect="equal")
    
    n = len(radius_arr)
    for i in range(n):
        radius, R, t = radius_arr[i], R_arr[i], t_arr[i].reshape(-1,1)
        x, y, z = get_ellipsoid(radius, R=R, t=t)

        # 使用u和v的正余弦函数创建渐变效果，并创建一个颜色映射
        color_values = (np.sin(x) * np.cos(y) + 1.0)/2.0
        color_map = plt.cm.coolwarm  # plt.cm.get_cmap('viridis')  # 
        colors = color_map(color_values / color_values.max())
        
        ax.plot_surface(x, y, z, rstride=2*stride_scale, cstride=2*stride_scale, 
                        facecolors=colors, # edgecolors='k'   # color='c'  # 
                        linewidth=0.5, antialiased=True, alpha=0.8)
        ax.contourf(x, y, z, zdir='z', offset=-6, cmap='coolwarm')

    plt.title('Ellipsoid')
    plt.show()
    return 0


def example():
    radius_arr = np.array([[1.5, 1.0, 0.5],[0.5, 1.0, 0.5],[0.5, 1.0, 1.5]])
    R_arr = np.stack([generate_random_rotation_matrix(),
                      generate_random_rotation_matrix(),
                      generate_random_rotation_matrix(),])
    t_arr = np.array([[2, 0, -2],[2, 2, -2],[-3, 0, -2]])
    vis_ellipsoid(radius_arr, R_arr, t_arr)


if __name__ == '__main__':

    example()
