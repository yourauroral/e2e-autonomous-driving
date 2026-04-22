'''
基于深度学习的端到端无人驾驶系统开发
专业班级：    姓名：    学号：
完成工作：创建网络连接，完成自动驾驶
'''

#1、导入第三方库
#（1）互联网相关类
import socketio
import eventlet.wsgi
from flask import Flask
#（2）图像处理类
import base64,cv2
from io import BytesIO
from PIL import Image
import numpy as np

#2、初始化变量
max_speed=15
steering_angle=-0.02
throttle=0.3

def send_control(steering_angle,throttle):
    sio.emit('steer',data={
             'steering_angle':steering_angle.__str__(),
             'throttle':throttle.__str__()
    })

#3、创建网络连接
sio = socketio.Server()
app = Flask(__name__)
app = socketio.WSGIApp(sio,app)

#4、传递参数，控制汽车行驶
@sio.on('connect')
def on_connect(sid,environ):
    print('与模拟器连接成功')

@sio.on('telemetry')
def on_telemetry(sid,data):
    if data:
        # print(data)
        speed=float(data['speed'])
        # print(speed)

        image =Image.open(BytesIO(base64.b64decode(data['image'])))
        image =np.array(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imshow('Image from Udacity Simulator',image)
        cv2.waitKey(1)
        print(image)

        throttle=1.0-steering_angle**2-(speed/max_speed)**2
        send_control(steering_angle,throttle)

    else:
        sio.emit('manual',data={})

@sio.on('disconnect')
def on_disconnect(sid):
    print('与模拟器断开连接')

#5、自动运行
eventlet.wsgi.server(eventlet.listen(('',4567)),app)