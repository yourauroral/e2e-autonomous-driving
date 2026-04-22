'''
基于深度学习的端到端无人驾驶系统开发
专业班级：    姓名：    学号：
完成工作：
1、扩充数据
2、模拟真实情况，让模型实现“见多识广”
3、归一化数据
'''

# 1、导入第三方库
import cv2                  # 用于图像处理（读取、变换等）
import numpy as np         # 用于数值计算和随机操作

# 2、初始化变量
image_height, image_width, image_channels = 66, 200, 3   # 模型输入图像尺寸
center = './test/center.jpg'   # 中间摄像头图像路径
left = './test/left.jpg'       # 左摄像头图像路径
right = './test/right.jpg'     # 右摄像头图像路径
steering_angle = 0.0           # 初始方向盘角度

# 3、随机选择图像（数据扩充：模拟不同摄像头）
def image_choose(center, left, right, steering_angle):
    choice = np.random.choice(3)   # 随机选择 0/1/2

    # 根据选择决定使用哪个摄像头
    if choice == 0:
        image_name = center
        bias = 0.0                 # 中间摄像头不需要偏移
    if choice == 1:
        image_name = left
        bias = 0.2                 # 左摄像头需要向右修正方向
    if choice == 2:
        image_name = right
        bias = -0.2                # 右摄像头需要向左修正方向

    image = cv2.imread(image_name)   # 读取图像
    steering_angle = steering_angle + bias  # 修正方向角

    return image, steering_angle

# 4、随机翻转图像（数据增强：镜像）
def image_flip(image, steering_angle):
    if np.random.rand() < 0.5:   # 50% 概率进行翻转
        image = cv2.flip(image, 1)   # 水平翻转
        steering_angle = -steering_angle  # 方向角取反

    return image, steering_angle

# 5、随机平移图像（数据增强：模拟车偏移）
def image_translate(image, steering_angle):
    range_X, range_Y = 100, 10   # 平移范围（左右100像素，上下10像素）

    # 随机生成平移距离
    tran_X = int(range_X * (np.random.rand() - 0.5))
    tran_Y = int(range_Y * (np.random.rand() - 0.5))

    # 构造平移矩阵
    tran_m = np.float32([[1, 0, tran_X], [0, 1, tran_Y]])

    # 应用仿射变换（平移）
    image = cv2.warpAffine(image, tran_m, (image.shape[1], image.shape[0]))

    # 根据水平位移修正方向盘角度（模拟车辆纠偏）
    steering_angle = steering_angle + tran_X * 0.002

    return image, steering_angle

# 6、图像归一化处理（裁剪 + 缩放 + 颜色空间转换）
def image_normalized(image):
    image = image[60:-25, :, :]   # 裁剪掉天空和车头（保留道路）
    image = cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)  # 缩放到模型输入大小
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # 转换为YUV颜色空间（更适合训练）

    return image

# 7、完整预处理流程（数据增强 pipeline）
def image_preprocessing(center, left, right, steering_angle):
    image, steering_angle = image_choose(center, left, right, steering_angle)  # 随机选图
    image, steering_angle = image_flip(image, steering_angle)                  # 随机翻转
    image, steering_angle = image_translate(image, steering_angle)             # 随机平移

    return image, steering_angle

if __name__ == '__main__':
    # 执行数据增强
    image, steering_angle = image_preprocessing(center, left, right, steering_angle)

    # 执行归一化
    image = image_normalized(image)

    # 输出方向角
    print(steering_angle)

    # 显示处理后的图像
    cv2.imshow('image_data', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()