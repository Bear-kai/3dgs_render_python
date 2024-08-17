from vpython import *

# ref:  https://vpython.cn/
# orig: https://www.glowscript.org/docs/VPythonDocs/index.html
# install: pip install vpython -i https://pypi.tuna.tsinghua.edu.cn/simple



# 创建一个场景，相当于一个可视化的窗口
scene = canvas(title='Ellipsoid Visualization')

# 定义椭球体的参数
# pos 设置椭球体的中心位置
# length, height, width 分别设置椭球体在x,y,z坐标轴方向上的大小
ellipsoid_pos = [vec(2, 0, -2), vec(2, 2, -2), vec(-3, 0, -2)]
ellipsoid_length = [1,2,3]
ellipsoid_height = [1,2,3]
ellipsoid_width  = [1,2,3]
ellipsoid_opacity = [0.8, 0.6, 0.4]

for pos,x,y,z,opa in zip(ellipsoid_pos,ellipsoid_length, 
                        ellipsoid_length, ellipsoid_width, ellipsoid_opacity):
    # 创建椭球体对象
    e = ellipsoid(pos=pos, length=x, height=y, width=z)  # , color=color.cyan
    # 设置椭球体的透明度，1.0表示完全不透明，0.0表示完全透明
    e.opacity = opa

# 运行一个循环以保持可视化窗口的开启; 检查是否有用户交互，例如关闭窗口等
while True:
    rate(10)            # 控制循环的速率
    k = keysdown()      # 一个包含按下键的列表
    if 'q' in k:
        break
print('Done with loop')
