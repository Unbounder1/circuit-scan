from ultralytics import YOLO
import numpy as np
import json
import cv2

model = YOLO("/Users/rdong/Documents/Github/circuit-scan/models/Train_26.pt")

model.predict(source = "image2.png", show=True, save=True, conf=0.5)