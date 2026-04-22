# 导入Keras相关模块（用于构建神经网络）
from keras.models import Sequential                # 顺序模型（按层堆叠）
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

# 导入预处理阶段定义的图像尺寸
from preprocessing import image_height, image_width, image_channels 

# 定义输入尺寸（高度，宽度，通道数）
input_size = (image_height, image_width, image_channels)


# =========================
# 模型1（基础CNN结构）
# =========================
def build_model1():
  model = Sequential()

  # 数据归一化（像素从[0,255]映射到[-1,1]）
  model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=input_size))

  # 第一层卷积（提取低级特征）
  model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))

  # 池化层（降采样，减少计算量）
  model.add(MaxPooling2D())

  # Dropout（防止过拟合）
  model.add(Dropout(0.25))

  # 第二层卷积（提取更复杂特征）
  model.add(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu'))

  # 池化层
  model.add(MaxPooling2D())

  # 展平（从二维特征图变成一维向量）
  model.add(Flatten())

  # 全连接层
  model.add(Dense(32))

  # Dropout
  model.add(Dropout(0.20))

  # 全连接层
  model.add(Dense(16))

  # 输出层（预测方向盘角度）
  model.add(Dense(1))

  # 输出模型结构
  model.summary()
  return model


# =========================
# 模型2（NVIDIA自动驾驶模型风格）
# =========================
def build_model2():
  model = Sequential()

  # 输入归一化
  model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=input_size))

  # 多层卷积（逐步提取特征）
  model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
  model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'))
  model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'))
  model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='elu'))
  model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='elu'))

  # 展平
  model.add(Flatten())

  # 全连接层（逐步回归方向角）
  model.add(Dense(100, activation='elu'))
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation='elu'))

  # 输出层
  model.add(Dense(1))

  model.summary()
  return model


# =========================
# 模型3（深层CNN + 强正则化）
# =========================
def build_model3():
  model = Sequential()

  # 输入归一化
  model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=input_size))

  # 第一组卷积（浅层特征）
  model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
  model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
  model.add(MaxPooling2D((2, 2), padding='same'))
  model.add(Dropout(0.5))   # 强正则化

  # 第二组卷积（中层特征）
  model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
  model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
  model.add(MaxPooling2D((2, 2), padding='same'))
  model.add(Dropout(0.5))

  # 第三组卷积（高层特征）
  model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
  model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu'))
  model.add(MaxPooling2D((2, 2), padding='same'))
  model.add(Dropout(0.5))

  # 展平
  model.add(Flatten())

  # 全连接层
  model.add(Dense(512, activation='elu'))
  model.add(Dense(64, activation='elu'))
  model.add(Dense(16, activation='elu'))

  # 输出层（回归方向角）
  model.add(Dense(1))

  model.summary()
  return model


# =========================
# 主函数（测试三个模型）
# =========================
if __name__ == '__main__':
  model = build_model1()   # 构建模型1
  model = build_model2()   # 构建模型2
  model = build_model3()   # 构建模型3