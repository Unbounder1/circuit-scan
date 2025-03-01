from ultralytics import YOLO
import cv2
import json

model = YOLO("/Users/rdong/Documents/Github/circuit-scan/models/Train_25.pt")

def process_image(image, threshold=0.5): 
    """
    Processes images using default specified yolo model
    Params:
    image -> cv2 read version of the image (processing already done)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)

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
    print(json.dumps(bounding_boxes, indent=4))

    return bounding_boxes


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
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized

def normalize_image(bounding_boxes, standard_x=80, standard_y=40):
    for box in bounding_boxes:
        if box["class_id"] == 10: # If its a resistor
            x1 = box["x1"]
            x2 = box["x2"]
            y1 = box["y1"]
            y2 = box["y2"]
            scalarx = standard_x/(x2-x1)
            scalary = standard_y/(y2-y1)
            scale = (scalarx+scalary)/2 # Return average of ratio of difference between resistor bounding & ideal
            return scale
    print("No resistor to normalize image")
    return -1
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

    bounding_boxes = process_image(image)

    cv2.imshow('Image with White Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


