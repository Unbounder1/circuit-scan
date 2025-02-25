import cv2
import numpy as np
from ultralytics import YOLO
import json

if __name__ == "__main__":
    import process_image as p
    # Load image
    image = cv2.imread('image.png')
    if image is None:
        print("Error: Could not load image.")
        exit(1)

    # Process image to get bounding boxes
    bounding_boxes = p.process_image(image)

    # Remove objects other than junctions
    for box in bounding_boxes:
        if box["class_id"] != 2:  # ignore junctions
            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            image[y1:y1 + h, x1:x1 + w] = (0, 0, 0)  # set to black

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours using a copy of the edges
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw the contours (1-pixel wide)
    # Using a single channel image where 0 represents black and 255 white
    contour_img = np.zeros_like(gray)

    # Draw each contour as a 1-pixel wide line
    for cnt in contours:
        cv2.polylines(contour_img, [cnt], isClosed=True, color=255, thickness=1)

    # Display the result for DFS path extraction
    cv2.imshow('1-Pixel Contours', contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()