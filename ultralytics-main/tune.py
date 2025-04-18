from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.tune(data='/kaggle/input/huawei2020-trash/trash.yaml', epochs=10, iterations=300, optimizer='AdamW', plots=False, save=False, val=False, project='/kaggle/working/tune',
                name='exp',)