from ultralytics import YOLO
import numpy as np
import json
import cv2

model = YOLO("models/obb/train5_obb_e900.pt")

model.predict(source = "image5.png", save=True, conf=0.1)