import cv2
import numpy as np
from scipy.spatial import KDTree
import re
import pytesseract

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

label_match = {
    # "component name": "regex of label"
    "resistor" : "R*" 
}

value_match = {
    # "component name" : "regex of value"
    "resistor" : "{1-9}*" 
}

def process_values(image, text_box):
    x1, x2, y1, y2 =  int(text_box["x1"]), int(text_box["x2"]), int(text_box["y1"]), int(text_box["y2"])
    cropped_image = image[y1-5:y2+5, x1-5:x2+5]

    print( pytesseract.image_to_string(cropped_image))

    return pytesseract.image_to_string(cropped_image)

    


def create_kdtree_from_boxes(boxes):
    """
    Create a KDTree from bounding boxes for fast lookup.
    
    :param boxes: List of bounding boxes as dictionaries (x1, y1, x2, y2, class_id, text).
    :param class_filter: Class ID to filter for KDTree construction (e.g., text elements).
    :return: KDTree object and filtered boxes.
    """
    box_centers = []
    filtered_boxes = []

    for box in boxes:
        if box["class_id"] == 1:  # Filter for text ONLY
            center_x = (box["x1"] + box["x2"]) / 2
            center_y = (box["y1"] + box["y2"]) / 2
            box_centers.append((center_x, center_y))
            filtered_boxes.append(box) 

    if not box_centers:
        return None, [] 

    return KDTree(np.array(box_centers)), filtered_boxes


def find_nearest_text(box, kdtree, text_boxes, image, search_size=3, search_radius=300):
    """
    Find the nearest text labels using KDTree lookup.

    :param box: Dictionary with bounding box coordinates.
    :param kdtree: KDTree object of text box centers.
    :param text_boxes: List of text bounding boxes (must align with KDTree indices).
    :param search_size: Number of nearby text boxes to check.
    :return: (output_label, output_value)
    """
    if kdtree is None or len(text_boxes) == 0:
        return "", ""  # No valid KDTree or text boxes

    x = (box["x1"] + box["x2"]) / 2
    y = (box["y1"] + box["y2"]) / 2

    # Get nearest text box indices
    distances, indices = kdtree.query((x, y), k=min(search_size, len(text_boxes)))

    print(distances)

    # Ensure indices is iterable (if only one result, convert to list)
    if isinstance(indices, np.int64):
        indices = [indices]

    output_label = ""
    output_value = ""
    i = 0
    for idx in indices:
        if (distances[i] > search_radius):
            break
        text_content = process_values(image, text_boxes[idx])

        # Define regex patterns for labels and values
        value_match = re.compile(r"^\d+(\.\d+)?\s?[a-zA-Z.]+$")
        label_match = re.compile(r"^[a-zA-Z]+[0-9]*$")

        if value_match.match(text_content):
            output_value = text_content
        if label_match.match(text_content):
            output_label = text_content
        i += 1

    return output_label, output_value

if __name__ == "__main__":
    import process_image as p
    import node_connections as n
    
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

    scale = p.normalize_image(bounding_boxes)
    graph = n.node_graph(bounding_boxes, image, scalar=scale)

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