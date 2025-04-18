# prepare_calibration_data.py
import os
import cv2
import numpy as np

src_root = 'C:/Yolo/yolo11/ultralytics-main/datasets/feet.v3i.yolov11/train/images'
cal_img_num = 100  # 想要的图像个数
dst_root = 'C:/RdkX5_TOOL/DATA/calibration_data_rgb_f32_640'


## 1. 从原始图像文件夹中获取100个图像作为校准数据
num_count = 0
img_names = []
for src_name in sorted(os.listdir(src_root)):
    if num_count > cal_img_num:
        break
    img_names.append(src_name)
    num_count += 1

# 检查目标文件夹是否存在，如果不存在就创建
if not os.path.exists(dst_root):
    os.system('mkdir {0}'.format(dst_root))

## 2 为每个图像转换
# 参考了OE中/open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/01_common/python/data/下的相关代码
# 转换为float32格式的RGB图像，保存为.rgbchw格式

## 2.1 定义图像缩放函数，返回为np.float32
# 图像缩放为目标尺寸(W, H)
# 值得注意的是，缩放时候，长宽等比例缩放，空白的区域填充颜色为pad_value, 默认127
def imequalresize(img, target_size, pad_value=127.):
    target_w, target_h = target_size
    image_h, image_w = img.shape[:2]
    img_channel = 3 if len(img.shape) > 2 else 1

    # 确定缩放尺度，确定最终目标尺寸
    scale = min(target_w * 1.0 / image_w, target_h * 1.0 / image_h)
    new_h, new_w = int(scale * image_h), int(scale * image_w)

    resize_image = cv2.resize(img, (new_w, new_h))

    # 准备待返回图像
    pad_image = np.full(shape=[target_h, target_w, img_channel], fill_value=pad_value)

    # 将图像resize_image放置在pad_image的中间
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    pad_image[dh:new_h + dh, dw:new_w + dw, :] = resize_image

    return pad_image

## 2.2 开始转换
for each_imgname in img_names:
    img_path = os.path.join(src_root, each_imgname)

    img = cv2.imread(img_path)  # BGR, HWC
    if img is None:
        print(f"Error reading image: {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB, HWC
    img = imequalresize(img, (640, 640))
    img = np.transpose(img, (2, 0, 1))  # RGB, CHW

    # 检查图像形状
    if img.shape != (3, 640, 640):
        print(f"Unexpected image shape: {img.shape}")

    # 保存为 float32
    dst_path = os.path.join(dst_root, each_imgname + '.rgbchw')
    print("write:%s" % dst_path)
    img.astype(np.float32).tofile(dst_path)  # 保存为 float32

print('finish')
