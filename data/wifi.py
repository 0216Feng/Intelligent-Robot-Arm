#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 注意: 此程序在RDK板端端运行
# Attention: This program runs on RDK board.

import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API
from time import time
import argparse
import logging
import socket
import threading
import os
from time import time
from flask import Flask, Response, render_template_string

# 日志模块配置
# logging configs
logging.basicConfig(
    level=logging.INFO,  # 将日志级别调整为INFO，减少Debug信息
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

# 创建Flask应用
app = Flask(__name__)

# 全局变量，用于存储实时检测帧
global_frame = None

def main():
    H = 17.0    # 摄像头高度 (cm)
    theta = np.deg2rad(31.41)  # 标定后的倾斜角(弧度)
    h = 1.0    # 物体平面低于底座的高度 (cm)
    object_heights = load_object_heights() # 加载物体高度信息
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/home/sunrise/Desktop/data/bin_dir/yolo11n_detect_bayese_640x640_nv12/yolo11n_detect_bayese_640x640_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.""") 
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID.')
    parser.add_argument('--port', type=int, default=8080, help='Web server port.')
    parser.add_argument('--classes-num', type=int, default=24, help='Classes Num to Detect.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--ranging', action='store_true', help='Enable distance ranging for all objects')
    parser.add_argument('--focal-length', type=float, default=2465, help='Camera focal length in pixels')
    opt = parser.parse_args()
    logger.info(opt)

    # 获取本机IP地址
    ip_address = get_ip_address()
    logger.info(f"本机IP地址: {ip_address}")
    logger.info(f"打开浏览器访问: http://{ip_address}:{opt.port} 查看实时检测结果")

    # 实例化模型
    model = YOLO11_Detect(opt.model_path, opt.conf_thres, opt.iou_thres)
    
    # 启动Web服务器线程
    server_thread = threading.Thread(target=start_server, args=(opt.port,))
    server_thread.daemon = True
    server_thread.start()
    
    # 创建测距对象（如果启用）
    ranging = None
    if opt.ranging:
        ranging = MonocularRanging(H, theta, h, f=opt.focal_length, img_size=(1280, 720))
        logger.info(f"已启用距离测量功能 - 相机高度: {H}cm, 倾斜角: {np.degrees(theta):.1f}°")
    
    # 启动摄像头
    capture_and_detect(model, opt.camera_id, ranging, object_heights)

def get_ip_address():
    """获取板子的IP地址"""
    try:
        # 优先获取无线网络接口的IP地址
        interfaces = ['wlan0', 'eth0', 'usb0']  # 常见网络接口名称
        
        for ifname in interfaces:
            try:
                # 尝试获取特定接口的IP地址
                import netifaces
                addresses = netifaces.ifaddresses(ifname)
                if netifaces.AF_INET in addresses:
                    return addresses[netifaces.AF_INET][0]['addr']
            except (ImportError, ValueError):
                # 如果netifaces不可用或接口不存在，跳过
                pass
                
        # 回退到通用方法
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"获取IP地址失败: {e}")
        return "127.0.0.1"  # 如果获取失败，返回本地回环地址

def capture_and_detect(model, camera_id, ranging=None, object_heights=None):
    """从摄像头捕获视频并进行实时目标检测"""
    global global_frame
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"无法打开摄像头 {camera_id}")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 帧率统计变量
    frame_count = 0
    fps = 0
    fps_timer = time()
    
    # 初始化物体跟踪器
    tracker = ObjectTracker(stability_threshold=10, distance_threshold=5.0, position_threshold=20)
    
    # 如果没有提供物体高度信息，创建一个默认的全零列表
    if object_heights is None:
        object_heights = [0] * len(coco_names)
    
    logger.info("成功打开摄像头，开始实时检测")
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                logger.error("无法获取摄像头画面")
                break
            
            # 计时器开始
            process_start_time = time()
                
            # 准备输入数据
            input_tensor = model.bgr2nv12(frame)
            
            # 推理
            outputs = model.c2numpy(model.forward(input_tensor))
            
            # 后处理
            try:
                ids, scores, bboxes = model.postProcess(outputs)
                
                # 渲染检测结果
                detection_results = []
                
                for class_id, score, bbox in zip(ids, scores, bboxes):
                    # 获取该类别物体的高度
                    object_height = object_heights[class_id] if class_id < len(object_heights) else 0
                    
                    # 绘制检测框并获取中心点和距离信息
                    center_x, center_y, distance = draw_detection(
                        frame, bbox, score, class_id, ranging, object_height)
                    
                    # 保存检测结果，用于跟踪和稳定性分析
                    detection_results.append((center_x, center_y, distance, class_id, score))
                
                # 更新跟踪器并获取稳定的物体
                stable_objects = tracker.update(detection_results)
                
                # 处理稳定的物体
                for avg_distance, class_id in stable_objects:
                    # 获取垃圾分类类别
                    category = get_waste_category(class_id)
                    logger.info(f"检测到稳定物体: 类别={coco_names[class_id]}({class_id}), 距离={avg_distance:.1f}cm, 垃圾分类={category}")
                    
                    # 发送指令到串口
                    send_to_serial(avg_distance, category)
                
                # 显示检测统计信息
                if detection_results:
                    # 计算平均距离
                    distances = [d[2] for d in detection_results if d[2] is not None]
                    if distances:
                        avg_distance = sum(distances) / len(distances)
                        cv2.putText(frame, f"Avg Dist: {avg_distance:.1f}cm", (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 显示检测到的物体数量
                    cv2.putText(frame, f"Objects: {len(detection_results)}", (10, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            except Exception as e:
                logger.error(f"检测处理出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 计算并更新FPS
            frame_count += 1
            current_time = time()
            
            # 每秒更新一次FPS值
            if current_time - fps_timer >= 1.0:
                fps = frame_count / (current_time - fps_timer)
                frame_count = 0
                fps_timer = current_time
            
            # 计算处理单帧的时间
            process_time = time() - process_start_time
                
            # 在画面上显示性能信息
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Process Time: {process_time*1000:.1f} ms", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 更新全局帧
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])  
            global_frame = buffer.tobytes()
            
    except KeyboardInterrupt:
        logger.info("检测已停止")
    finally:
        cap.release()

def start_server(port):
    """启动Flask Web服务器"""
    app.run(host='0.0.0.0', port=port, threaded=True)

@app.route('/')
def index():
    """网页主页"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RDK Realtime Object Detection Streaming</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                text-align: center;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
            }
            .video-container {
                margin: 20px auto;
                max-width: 800px;
            }
            img {
                max-width: 100%;
                border: 2px solid #333;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>RDK Realtime Object Detection Streaming</h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="视频流">
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    """提供视频流"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """生成实时视频帧"""
    global global_frame
    while True:
        if global_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n')
        else:
            # 如果没有可用帧，生成一个空白帧
            blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(blank_frame, "等待摄像头画面...", (150, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

class BaseModel:
    def __init__(
        self,
        model_file: str
        ) -> None:
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_file)
            logger.debug("Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)))
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(model_file))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("-> input tensors")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("-> output tensors")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        self.model_input_height, self.model_input_weight = self.quantize_model[0].inputs[0].properties.shape[2:4]

    def resizer(self, img: np.ndarray)->np.ndarray:
        img_h, img_w = img.shape[0:2]
        self.y_scale, self.x_scale = img_h/self.model_input_height, img_w/self.model_input_weight
        return cv2.resize(img, (self.model_input_weight, self.model_input_height), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存
    
    def preprocess(self, img: np.ndarray)->np.array:
        begin_time = time()
        input_tensor = self.resizer(img)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.uint8)  # NCHW
        return input_tensor

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        begin_time = time()
        bgr_img = self.resizer(bgr_img)
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def forward(self, input_tensor: np.array) -> list[dnn.pyDNNTensor]:
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        return quantize_outputs

    def c2numpy(self, outputs) -> list[np.array]:
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        return outputs

class YOLO11_Detect(BaseModel):
    def __init__(self, 
                model_file: str, 
                conf: float, 
                iou: float
                ):
        super().__init__(model_file)
        # 将反量化系数准备好, 只需要准备一次
        self.s_bboxes_scale = self.quantize_model[0].outputs[0].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[1].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[2].properties.scale_data[np.newaxis, :]

        # DFL求期望的系数, 只需要生成一次
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]

        # anchors, 只需要生成一次
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)

        # 输入图像大小, 一些阈值, 提前计算好
        self.input_image_size = 640
        self.conf = conf
        self.iou = iou
        self.conf_inverse = -np.log(1/conf - 1)
        logger.info("iou threshol = %.2f, conf threshol = %.2f"%(iou, conf))
    
    def postProcess(self, outputs: list[np.ndarray]) -> tuple[list]:
        begin_time = time()
        try:
            # reshape
            s_bboxes = outputs[0].reshape(-1, 64)
            m_bboxes = outputs[1].reshape(-1, 64)
            l_bboxes = outputs[2].reshape(-1, 64)
            s_clses = outputs[3].reshape(-1, 24)
            m_clses = outputs[4].reshape(-1, 24)
            l_clses = outputs[5].reshape(-1, 24)

            # classify: 利用numpy向量化操作完成阈值筛选(优化版 2.0)
            s_max_scores = np.max(s_clses, axis=1)
            s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
            s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
            s_scores = s_max_scores[s_valid_indices]

            m_max_scores = np.max(m_clses, axis=1)
            m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
            m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
            m_scores = m_max_scores[m_valid_indices]

            l_max_scores = np.max(l_clses, axis=1)
            l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)  # 得到大于阈值分数的索引，此时为小数字
            l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
            l_scores = l_max_scores[l_valid_indices]

            # 3个Classify分类分支：Sigmoid计算
            s_scores = 1 / (1 + np.exp(-s_scores))
            m_scores = 1 / (1 + np.exp(-m_scores))
            l_scores = 1 / (1 + np.exp(-l_scores))

            # 3个Bounding Box分支：筛选
            s_bboxes_float32 = s_bboxes[s_valid_indices,:]
            m_bboxes_float32 = m_bboxes[m_valid_indices,:]
            l_bboxes_float32 = l_bboxes[l_valid_indices,:]

            # 3个Bounding Box分支：dist2bbox (ltrb2xyxy)
            s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            s_anchor_indices = self.s_anchor[s_valid_indices, :]
            s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
            s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
            s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8

            m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            m_anchor_indices = self.m_anchor[m_valid_indices, :]
            m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
            m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
            m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16

            l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            l_anchor_indices = self.l_anchor[l_valid_indices,:]
            l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
            l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
            l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32

            # 大中小特征层阈值筛选结果拼接
            dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
            scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
            ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

            # 处理可能的空检测情况
            if len(dbboxes) == 0:
                return [], [], []

            # nms
            indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf, self.iou)

            # 还原到原始的img尺度
            bboxes = dbboxes[indices] * np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])
            bboxes = bboxes.astype(np.int32)

            return ids[indices], scores[indices], bboxes
        except Exception as e:
            logger.error(f"后处理出错: {e}")
            return [], [], []


coco_names = [
    'Disposable Fast Food Box', 'Book Paper', 'Plastic Utensils', 'Plastic Toys', 'Dry Battery', 'Express Paper Bag', 'Plug Wire', 'Can', 'Peel and Pulp', 'Stuffed Toy', 'Defiled Plastic', 'Contaminated paper', 'Toilet care products', 'Cigarette butts', 'Carton box', 'Tea residue', 'Cai Bang Cai Ye', 'Egg Shell', 'Sauce Bottle', 'Ointment', 'Expired Medicine', 'Metal Food Cans', 'edible oil drums', 'drink bottles'
    ]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img: np.array, 
                   bbox: tuple[int, int, int, int],
                   score: float, 
                   class_id: int,
                   ranging=None,
                   object_height=0) -> tuple:
    """
    绘制检测框并计算距离（如果提供了测距对象）

    Parameters:
        img (np.array): 输入图像
        bbox (tuple[int, int, int, int]): 边界框坐标 (x1, y1, x2, y2)
        score (float): 检测分数
        class_id (int): 类别ID
        ranging (MonocularRanging, optional): 单目测距对象

    Returns:
        tuple: (center_x, center_y, distance) - 检测框中心坐标和距离
    """
    x1, y1, x2, y2 = bbox
    # 检查边界框坐标是否超出图像范围
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    # 计算中心点坐标
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 绘制中心点
    cv2.circle(img, (int(center_x), int(center_y)), 3, (0, 255, 255), -1)
    
    # 测距计算
    distance = None
    if ranging is not None:
        distance = ranging.calculate_distance(center_y, object_height)
    
    color = rdk_colors[class_id % 20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 安全检查，确保class_id在有效范围内
    if 0 <= class_id < len(coco_names):
        class_name = coco_names[class_id]
    else:
        class_name = f"Unknown-{class_id}"
    
    # 添加距离信息到标签中
    if distance is not None:
        label = f"{class_name}: {score:.2f} [{distance:.1f}cm]"
    else:
        label = f"{class_name}: {score:.2f}"
        
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return (center_x, center_y, distance)

class MonocularRanging:
    def __init__(self, H, theta, h=0, f=800, img_size=(640,480)):
        self.H = H
        self.theta = theta  # 倾斜角（弧度）
        self.h = h
        self.f = f
        self.img_center_y = img_size[1] / 2  # 仅需垂直中心

    def pixel_to_alpha(self, y_pixel):
        """根据垂直像素坐标计算角度偏移"""
        v = y_pixel - self.img_center_y  # 计算垂直偏移
        alpha = np.arctan(v / self.f)
        return alpha

    def calculate_distance(self, y_pixel, object_height=0):
        alpha = self.pixel_to_alpha(y_pixel)
        # 根据物体高度调整有效高度
        H_eff = self.H + self.h - object_height/2
        D = H_eff / np.tan(self.theta + alpha) - 7.0 #减去相机到基座前端的距离
        if object_height > 0:
            # 如果有物体高度信息，进行距离修正
            D = D + 4.0 # 增加偏移量
        return D
    
class ObjectTracker:
    def __init__(self, stability_threshold=10, distance_threshold=5.0, position_threshold=20):
        """
        初始化物体跟踪器
        
        参数:
            stability_threshold: 判定物体稳定所需的连续帧数
            distance_threshold: 判定距离稳定的阈值(cm)
            position_threshold: 判定位置稳定的阈值(像素)
        """
        self.tracked_objects = {}  # 跟踪的物体 {id: {positions:[], distances:[], class_id:, stable_count:, sent:}}
        self.stability_threshold = stability_threshold
        self.distance_threshold = distance_threshold
        self.position_threshold = position_threshold
        
    def update(self, detections):
        """
        更新跟踪的物体
        
        参数:
            detections: 列表，每个元素是 (center_x, center_y, distance, class_id, score)
        
        返回:
            stable_objects: 刚刚变得稳定的物体列表，每个元素是 (avg_distance, class_id)
        """
        global last_command_time
        stable_objects = []
        
        # 检查是否可以发送新指令
        current_time = time()
        can_send_new_command = (current_time - last_command_time >= 20)
        
        # 如果不能发送新指令，就不需要检测新的稳定物体
        if not can_send_new_command:
            return []
        
        # 标记所有当前物体为未更新
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['updated'] = False
            
        # 为每个检测匹配或创建跟踪对象
        for center_x, center_y, distance, class_id, score in detections:
            if distance is None:
                continue
                
            # 尝试匹配到现有物体
            matched = False
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data['class_id'] == class_id and not obj_data['updated']:
                    # 检查位置距离是否接近
                    last_x, last_y = obj_data['positions'][-1]
                    pos_distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                    
                    if pos_distance < self.position_threshold:
                        # 匹配成功，更新数据
                        obj_data['positions'].append((center_x, center_y))
                        obj_data['distances'].append(distance)
                        obj_data['updated'] = True
                        
                        # 保持最近的10个数据点
                        if len(obj_data['positions']) > 10:
                            obj_data['positions'].pop(0)
                            obj_data['distances'].pop(0)
                        
                        # 判断物体是否稳定
                        if not obj_data['sent']:
                            # 计算距离方差
                            if len(obj_data['distances']) >= self.stability_threshold:
                                recent_distances = obj_data['distances'][-self.stability_threshold:]
                                distance_std = np.std(recent_distances)
                                
                                recent_positions = obj_data['positions'][-self.stability_threshold:]
                                x_std = np.std([p[0] for p in recent_positions])
                                y_std = np.std([p[1] for p in recent_positions])
                                
                                # 如果位置和距离都稳定
                                if distance_std < self.distance_threshold and x_std < self.position_threshold/2 and y_std < self.position_threshold/2:
                                    obj_data['stable_count'] += 1
                                else:
                                    obj_data['stable_count'] = 0
                                    
                                # 如果连续几帧都稳定，则认为物体稳定
                                if obj_data['stable_count'] >= 3:
                                    avg_distance = np.mean(recent_distances)
                                    stable_objects.append((avg_distance, class_id))
                                    obj_data['sent'] = True
                        
                        matched = True
                        break
            
            # 如果没有匹配到现有物体，创建新物体
            if not matched:
                new_id = len(self.tracked_objects) + 1
                self.tracked_objects[new_id] = {
                    'positions': [(center_x, center_y)],
                    'distances': [distance],
                    'class_id': class_id,
                    'stable_count': 0,
                    'updated': True,
                    'sent': False
                }
        
        # 移除长时间未更新的物体
        ids_to_remove = []
        for obj_id, obj_data in self.tracked_objects.items():
            if not obj_data['updated']:
                ids_to_remove.append(obj_id)
                
        for obj_id in ids_to_remove:
            del self.tracked_objects[obj_id]
            
        return stable_objects

def load_object_heights():
    """从label.txt文件中加载物体高度信息"""
    object_heights = [0] * len(coco_names)  # 默认高度为0
    try:
        with open('label.txt', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                # 尝试提取行末尾的数字（物体高度）
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # 最后一个部分可能是高度数字
                        height = float(parts[-1])
                        if i < len(object_heights):
                            object_heights[i] = height
                    except ValueError:
                        # 如果不是数字，则忽略
                        pass
    except Exception as e:
        logger.error(f"读取label.txt文件时出错: {e}")
    
    logger.info(f"已加载物体高度信息: {object_heights}")
    return object_heights

def get_waste_category(class_id):
    """
    根据类别ID获取垃圾分类类别
    
    参数:
        class_id: 检测的类别ID
        
    返回:
        category: 垃圾分类类别 (0-可回收, 1-厨余垃圾, 2-有害垃圾, 3-其他垃圾)
    """
    # 根据label.txt映射类别ID到垃圾分类类别
    # 这里需要根据您实际的label.txt内容修改
    recyclable = [1, 2, 3, 5, 6, 7, 9, 12, 14, 18, 21, 22, 23]  # 可回收物
    kitchen_waste = [8, 15, 16, 17]  # 厨余垃圾
    hazardous = [4, 19, 20]  # 有害垃圾
    
    if class_id in recyclable:
        return 0
    elif class_id in kitchen_waste:
        return 1
    elif class_id in hazardous:
        return 2
    else:
        return 3  # 其他垃圾

# 在主函数之前添加一个全局变量来跟踪最后一次发送指令的时间
last_command_time = 0

def send_to_serial(distance, category):
    """
    使用control.py发送指令到串口，并保证指令间隔至少20秒
    
    参数:
        distance: 距离，单位cm
        category: 垃圾分类类别 (0-3)
    """
    global last_command_time
    current_time = time()
    
    # 检查是否满足发送时间间隔要求
    if current_time - last_command_time < 20:
        time_to_wait = 20 - (current_time - last_command_time)
        logger.info(f"上次发送指令距现在只有 {current_time - last_command_time:.1f} 秒，"
                   f"需等待 {time_to_wait:.1f} 秒")
        return
    
    # 距离限制：大于20cm不发送
    if distance > 20:
        logger.info(f"距离 {distance:.1f}cm 超过20cm限制，不发送指令")
        return
    
    command = f"%D{distance:.1f}C{category}*"
    logger.info(f"发送指令: {command}")
    
    # 构建调用命令
    import subprocess
        
    # 发送命令
    result = subprocess.run(['python', 'control.py', '--data', command, '--keep-open'], 
                           capture_output=True, text=True)
    
    # 更新最后发送时间
    last_command_time = time()
    
    logger.info(f"发送结果: {result.stdout}")
    if result.stderr:
        logger.error(f"发送错误: {result.stderr}")

if __name__ == "__main__":
    main()