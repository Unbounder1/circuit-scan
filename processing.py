from ultralytics import YOLO
import numpy as np
import json
import cv2

model = YOLO("old_models/Best_OBB.pt")

result = model("image3.png")
model.predict(source = "image3.png", save=True, conf=0.1)