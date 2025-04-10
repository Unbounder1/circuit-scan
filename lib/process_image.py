from ultralytics import YOLO
import cv2
import json
from scipy.spatial import KDTree
import math
import os
import base64

# export YOLO="/Users/rdong/Documents/Github/circuit-scan/models/Train_25.pt"
# export YOLO_OBB="/Users/rdong/Documents/Github/circuit-scan/models/obb/train5_obb_e445.pt"

model = YOLO("./models/yolo.pt")
model_obb = YOLO("./models/yolo_obb.pt")

def process_image(image, threshold=0.5): 
    """
    Processes images using default specified yolo model
    Params:
    image -> cv2 read version of the image (processing already done)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)

    annotated_image = results[0].plot(
    conf=True,         
    line_width=2
    )

    # Encode image to PNG format (returns success flag and buffer)
    _, buffer = cv2.imencode(".png", annotated_image)

    img_base64 = base64.b64encode(buffer).decode("utf-8")

    #non obb result
    bounding_boxes = []
    for result in results: 
        boxes = result.boxes  # Bounding boxes object

        for box in boxes:
            if (box.conf[0].item() < threshold):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordinates (top-left and bottom-right)
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID

            
            # Append to list
            bounding_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": confidence,
                "class_id": class_id
            })

    # ----- debug ------
    # Save bounding boxes to a file 
    with open("output.json", "w") as f:
        json.dump(bounding_boxes, f, indent=4)

    # Print bounding boxes 
    # print(json.dumps(bounding_boxes, indent=

    return bounding_boxes, img_base64

def resize_image(image, max_size=1000):
    """
    Resize an image while maintaining aspect ratio such that the longest side is max_size.
    
    :param image: Input image (numpy array).
    :param max_size: Maximum allowed dimension (width or height).
    :return: Resized image.
    """
    h, w = image.shape[:2]

    # Compute scaling factor while maintaining aspect ratio
    scale = max_size / max(h, w)
    
    # Compute new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    print(f"Width = {new_w}, Height = {new_h}")
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized

def normalize_image(bounding_boxes, standard_x=100, standard_y=100):
    scales = []

    for box in bounding_boxes:
        if box["class_id"] not in [1, 2]:
            x1, x2 = box["x1"], box["x2"]
            y1, y2 = box["y1"], box["y2"]

            width = x2 - x1
            height = y2 - y1

            if width == 0 or height == 0:
                continue  # skip invalid box

            scalarx = standard_x / width
            scalary = standard_y / height
            scale = (scalarx + scalary) / 2
            scales.append(scale)

    if not scales:
        print("No resistor to normalize image")
        return -1

    return sum(scales) / len(scales)

def process_bounding_box(bounding_boxes, image):
    for box in bounding_boxes:
        if box["class_id"] != 1: # ignore junctions for bounding box
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            image[y1:y1 + h, x1:x1 + w] = (0, 0, 0)
    return image

def compute_theta(x1, y1, x2, y2):
    """Computes the rotation angle θ (in radians) of the bounding box."""
    theta = math.atan2(y2 - y1, x2 - x1)
    return theta

def associate_rotation(image, bounding_boxes, kdtree, threshold = 0.1):
    """
    Processes images using default specified yolo model
    :param bounding_boxes: Bounding box input 
    :param kdtree: bounding box kdtree
    :return: bounding_boxes with ["theta"] attribute
    """
    #obb output
    obb_results = model_obb(image)
    for result in obb_results: 
        
        if result.obb is None:
            print("Warning: No OBB detections found!")
            continue

        obb_coords = result.obb.xyxyxyxy  # Polygon format (4 points per box)
        class_ids = result.obb.cls.int()  # Class ID for each box
        confs = result.obb.conf  # Confidence scores
        names = [result.names[cls.item()] for cls in class_ids]  # Class names

        for i in range(len(obb_coords)):
            if confs[i].item() < threshold:
                continue

            # Extract polygon points
            obb_points = obb_coords[i] 

            x1, y1 = obb_points[0]
            x2, y2 = obb_points[1]
            x3, y3 = obb_points[2]
            x4, y4 = obb_points[3]
            x_c = (x1 + x2 + x3 + x4) / 4
            y_c = (y1 + y2 + y3 + y4) / 4

            _, idx = kdtree.query((x_c, y_c), k=1)
            
            if (bounding_boxes[idx]["x1"] - 1 <= x_c <= bounding_boxes[idx]["x2"]) and (bounding_boxes[idx]["y1"] - 1 <= y_c <= bounding_boxes[idx]["y2"]):
                bounding_boxes[idx]["theta"] = compute_theta(x1,y1,x2,y2)
            else:
                bounding_boxes[idx]["theta"] = 0

    for box in bounding_boxes:
        try:
            print(f"The angle of the thing is {box['theta']}")
        except:
            continue

    
    return bounding_boxes
    
if __name__ == "__main__":
    image = resize_image(cv2.imread('image.png'))# resized image
    bounding_boxes = process_image(image)

    # show bounding boxes
    for box in bounding_boxes:
        if box["class_id"] != -1: # ignore junctions
            x1 = box["x1"]
            x2 = box["x2"]
            y1 = box["y1"]
            y2 = box["y2"]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            h, w = int(h), int(w)
            x1, y1 = int(x1), int(y1)
            image[y1:y1 + h, x1:x1 + w] = (0, 0, 0) # set black

    cv2.imshow('Image with White Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


