from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.tune(data='../dataset/trainval/trash.yaml', epochs=10, iterations=20, optimizer='SGD', plots=False, save=False, val=False)