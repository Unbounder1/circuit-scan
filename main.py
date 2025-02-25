from ultralytics import YOLO
import json
import cv2

model = YOLO("/Users/rdong/Documents/Github/circuit-scan/models/Train_25.pt")

#model.predict(source = "image.png", show=True, save=True, conf=0.1)

noComponents = cv2.imread('image.png')
results = model(["image.png"])

bounding_boxes = []
for result in results:  # Iterate through results
    boxes = result.boxes  # Bounding boxes object

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordinates (top-left and bottom-right)
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID

        if class_id != 2 and class_id != 1: # ignore junctions
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            h, w = int(h), int(w)
            x1, y1 = int(x1), int(y1)

            noComponents[y1:y1 + h, x1:x1 + w] = (255, 255, 255) # set white
        
        # Append to list
        bounding_boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": confidence,
            "class_id": class_id
        })

# Save bounding boxes to a file
with open("output.json", "w") as f:
    json.dump(bounding_boxes, f, indent=4)

# Print bounding boxes
print(json.dumps(bounding_boxes, indent=4))

cv2.imshow('Image with White Boxes', noComponents)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('image_with_white_boxes.jpg', image)