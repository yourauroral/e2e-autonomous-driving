'''
基于深度学习的Udacity无人驾驶系统
班级：        学号：       姓名：
'''
#完成工作：训练卷积神经网络模型

#1、导入第三方库
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from preprocessing import image_height, image_width, image_channels
from preprocessing import image_preprocessing, image_normalized
from build_model import build_model1, build_model2, build_model3

#2、初始化变量
data_path = 'data_mountain/' # data_mountain/
test_ration = 0.1
batch_size = 100
batch_num = 200
epoch = 300
#3、导入数据
def load_data(data_path):
    data_csv = pd.read_csv(data_path+'driving_log.csv',
                            header=None,
                            usecols=[0, 1, 2, 3])
    data_csv.columns = ['center', 'left', 'right', 'steering']
    X = data_csv[['center', 'left', 'right']].values
    Y = data_csv['steering'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ration, random_state=0)
    return X_train, X_test, Y_train, Y_test

#4、批量生成数据（喂料机）
def batch_generator(data_path, batch_size, X_data, Y_data, train_flag):
    image_container = np.empty([batch_size, image_height, image_width, image_channels])
    steer_container = np.empty(batch_size)
    while True:
        ii = 0
        for index in np.random.permutation(X_data.shape[0]):
            center, left, right = [data_path + p.strip() for p in X_data[index]]
            steering_angle = Y_data[index]
            if train_flag and np.random.rand()<0.4:
                image, steering_angle = image_preprocessing(center, left, right, steering_angle)
            else:
                image = cv2.imread(center)
            image_container[ii]=image_normalized(image)
            steer_container[ii]=steering_angle
            ii += 1
            if ii == batch_size:
                break
        yield image_container, steer_container

#5、训练模型
X_train, X_test, Y_train, Y_test = load_data(data_path)
model = build_model2()
checkpoint = ModelCheckpoint(
    'ccc_mountain_model2_{epoch:03d}.h5', 
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)
stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=500,
    verbose=1,
    mode='auto'
)
tensor_board = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=1,
    write_images=0
)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
model.fit(
    batch_generator(data_path, batch_size, X_train, Y_train,True),
    steps_per_epoch=batch_num,
    epochs=epoch,
    verbose=1,
    validation_data=batch_generator(data_path, batch_size, X_test, Y_test, False),
    validation_steps=1,
    max_queue_size=1,
    callbacks=[checkpoint,stopping,tensor_board]
)

#6、保存模型
model.save('ccc_mountain_model2.h5')