from ultralytics import YOLO
YOLO('best.pt').export(imgsz=640, format='onnx', simplify=True, opset=11)