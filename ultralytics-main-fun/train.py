import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'./ultralytics/cfg/models/11/yolo11s.yaml')  # 模型配置文件)
    model.load('yolo11s.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'../../trainval/trash.yaml',  # 数据集路径
                imgsz=640,
                epochs=100,
                batch=16,
                workers=4,
                device='',
                optimizer='SGD',
                close_mosaic=50,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )