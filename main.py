from ultralytics import YOLO

model = YOLO("/Users/rdong/Documents/Github/circuit-scan/yolo11_50epoch.pt")

model.predict(source = "image.png", show=True, save=True, conf=0.1)