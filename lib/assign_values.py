import cv2
import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt

classes = [
        "__background__", "text", "junction", "crossover", "terminal", "gnd", "vss",
        "voltage.dc", "voltage.ac", "voltage.battery",
        "resistor", "resistor.adjustable", "resistor.photo",
        "capacitor.unpolarized", "capacitor.polarized", "capacitor.adjustable",
        "inductor", "inductor.ferrite", "inductor.coupled", "transformer",
        "diode", "diode.light_emitting", "diode.thyrector", "diode.zener",
        "diac", "triac", "thyristor", "varistor",
        "transistor.bjt", "transistor.fet", "transistor.photo",
        "operational_amplifier", "operational_amplifier.schmitt_trigger", "optocoupler",
        "integrated_circuit", "integrated_circuit.ne555", "integrated_circuit.voltage_regulator",
        "xor", "and", "or", "not", "nand", "nor",
        "probe", "probe.current", "probe.voltage",
        "switch", "relay", "socket", "fuse",
        "speaker", "motor", "lamp", "microphone", "antenna", "crystal",
        "mechanical", "magnetic", "optical", "block", "explanatory", "unknown"
    ]

def create_kdtree_from_boxes(boxes):
        """
        Create a KDTree from bounding boxes for fast pixel lookup.
        :param boxes: List or array of (x1, y1, x2, y2) bounding boxes.
        :return: KDTree object and list of bounding boxes.
        """
        box_centers = []
        for box in boxes:

            box_centers.append([(box["x1"] + box["x2"]) / 2,  # Get center of first box
              (box["y1"] + box["y2"]) / 2])
        return KDTree(box_centers), boxes

def find_nearest_text

if __name__ == "__main__":
    import process_image as p
    
    # Load image
    image = cv2.imread('image3.png')
    image = p.resize_image(image)
    if image is None:
        print("Error: Could not load image.")
        exit(1)

    # Process image to get bounding boxes
    bounding_boxes = p.process_image(image)

    # Remove objects other than junctions (class_id = 1)
    for box in bounding_boxes:
        if box["class_id"] != 1:  # Ignore non-junctions
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            image[y1:y1 + h, x1:x1 + w] = (0, 0, 0)  # Set to black

    # Convert the graph's binary image to RGB for colored text
    rgb_image = cv2.cvtColor(graph.image, cv2.COLOR_GRAY2BGR)

    # Draw indexes on the RGB image
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        
        # Calculate correct center of each bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Draw index number on the RGB image (red text)
        cv2.putText(rgb_image, str(i) + ": " + str(classes[box["class_id"]]), (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display both OpenCV image and Matplotlib graph visualization
    cv2.imshow('1-Pixel Contours with Indexes', rgb_image)
    cv2.waitKey(1)  # Prevent window from closing immediately

    print(graph)  # Display adjacency list in the console

    # cv2.waitKey(0)  # Wait for user keypress before closing windows
    # cv2.destroyAllWindows()