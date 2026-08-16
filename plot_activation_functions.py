import numpy as np
import cv2

# 定义Sigmoid函数和其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 创建画布 (更大的尺寸)
width, height = 1500, 900
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# 坐标范围
x_min, x_max = -10, 10
y_min, y_max = -0.3, 1.3

# 生成x值
num_points = 1000
x = np.linspace(x_min, x_max, num_points)

# 计算y值
y_sigmoid = sigmoid(x)
y_derivative = sigmoid_derivative(x)

# 坐标转换函数 (增大边距，使曲线图变小)
def x_to_pixel(x_val):
    margin = 150  # 增大边距
    return int(margin + (x_val - x_min) / (x_max - x_min) * (width - 2 * margin))

def y_to_pixel(y_val):
    margin = 120  # 增大边距
    return int(height - margin - (y_val - y_min) / (y_max - y_min) * (height - 2 * margin))

# 绘制坐标轴
axis_color = (100, 100, 100)
cv2.line(image, (x_to_pixel(x_min), y_to_pixel(0)), (x_to_pixel(x_max), y_to_pixel(0)), axis_color, 3)  # 加粗坐标轴
cv2.line(image, (x_to_pixel(0), y_to_pixel(y_min)), (x_to_pixel(0), y_to_pixel(y_max)), axis_color, 3)

# 绘制网格线
grid_color = (220, 220, 220)
for i in range(-10, 11, 2):
    x_pixel = x_to_pixel(i)
    cv2.line(image, (x_pixel, y_to_pixel(y_min)), (x_pixel, y_to_pixel(y_max)), grid_color, 1)

for i in range(0, 14, 2):
    y_val = i / 10.0
    y_pixel = y_to_pixel(y_val)
    cv2.line(image, (x_to_pixel(x_min), y_pixel), (x_to_pixel(x_max), y_pixel), grid_color, 1)

# 绘制Sigmoid函数 (蓝色)
sigmoid_color = (255, 0, 0)
for i in range(len(x) - 1):
    x1, x2 = x[i], x[i+1]
    y1, y2 = y_sigmoid[i], y_sigmoid[i+1]
    p1 = (x_to_pixel(x1), y_to_pixel(y1))
    p2 = (x_to_pixel(x2), y_to_pixel(y2))
    cv2.line(image, p1, p2, sigmoid_color, 4)

# 绘制Sigmoid导数 (橙色)
derivative_color = (255, 165, 0)
for i in range(len(x) - 1):
    x1, x2 = x[i], x[i+1]
    y1, y2 = y_derivative[i], y_derivative[i+1]
    p1 = (x_to_pixel(x1), y_to_pixel(y1))
    p2 = (x_to_pixel(x2), y_to_pixel(y2))
    cv2.line(image, p1, p2, derivative_color, 4)

# 添加标题 (更大的字体)
title_font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'Sigmoid Function and its Derivative', (80, 80),
            title_font, 2.0, (0, 0, 0), 4)  # 增大字体

# 添加图例 (更大的字体和方块)
legend_y = 140
cv2.rectangle(image, (1100, legend_y-20), (1160, legend_y+10), sigmoid_color, -1)
cv2.putText(image, 'Sigmoid', (1170, legend_y+5),
            title_font, 1.2, (0, 0, 0), 3)  # 增大字体

cv2.rectangle(image, (1100, legend_y+40), (1160, legend_y+70), derivative_color, -1)
cv2.putText(image, "Sigmoid's Derivative", (1170, legend_y+65),
            title_font, 1.2, (0, 0, 0), 3)  # 增大字体

# 添加坐标轴标签 (更大的字体)
label_font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'x', (width - 60, y_to_pixel(0) - 20),
            label_font, 1.5, (0, 0, 0), 3)  # 增大字体
cv2.putText(image, 'y', (x_to_pixel(0) + 15, 60),
            label_font, 1.5, (0, 0, 0), 3)  # 增大字体

# 添加刻度值 (更大的字体)
tick_font = cv2.FONT_HERSHEY_SIMPLEX
tick_size = 1.0  # 增大字体

# x轴刻度
for i in range(-10, 11, 5):
    if i != 0:
        x_pixel = x_to_pixel(i)
        cv2.putText(image, str(i), (x_pixel - 15, y_to_pixel(0) + 40),
                    tick_font, tick_size, (80, 80, 80), 3)  # 增大字体

# y轴刻度
for i in range(0, 12, 2):
    y_val = i / 10.0
    y_pixel = y_to_pixel(y_val)
    cv2.putText(image, f'{y_val:.1f}', (x_to_pixel(0) + 15, y_pixel + 8),
                tick_font, tick_size, (80, 80, 80), 3)  # 增大字体

# 添加原点标签 (更大的字体)
cv2.putText(image, 'O', (x_to_pixel(0) + 15, y_to_pixel(0) + 40),
            tick_font, tick_size, (80, 80, 80), 3)  # 增大字体

# 保存图像
output_path = 'd:/Code/python/CycleGAN-Style-Transfer/sigmoid_derivative_plot_v2.png'
cv2.imwrite(output_path, image)
print(f"图像已保存为 {output_path}")
