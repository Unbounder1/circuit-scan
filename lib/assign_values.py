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
    # Reference designators: R1, R27, etc.
    "resistor"      : r"^R\d+$",
    # Inductor designators: L1, L10, etc.
    "inductor"      : r"^L\d+$",
    # Voltage sources: V1, VCC, VDD, VSS, etc.
    "voltage"       : r"^V[A-Za-z0-9_]*$",
}

value_match = {
    # Resistor values: 10Ω, 4.7kΩ, 100R, 2.2MΩ, etc.
    "resistor"      : r"^[0-9]*\.?[0-9]+(?:[kKmM]?Ω|[kKmM]?R)$",
    # Inductor values: 10uH, 2.2 mH, 100H, etc.
    "inductor"      : r"^[0-9]*\.?[0-9]+(?:uH|μH|mH|H)$",
    # Voltage values: 3.3V, 5V, 12V, etc.
    "voltage"       : r"^[0-9]*\.?[0-9]+V$",
}

def safe_crop(image, x1, y1, x2, y2, x_padding=5, y_padding=5):
    height, width = image.shape[:2]

    # Expand the box with padding, and clamp to image bounds
    x1 = max(0, x1 - x_padding)
    y1 = max(0, y1 - y_padding)
    x2 = min(width, x2 + x_padding)
    y2 = min(height, y2 + y_padding)


    return image[y1:y2, x1:x2]

def fix_misread_numbers(value):
    """
    Corrects common OCR misreadings:
    - Replaces 'O' (uppercase letter 'O') with '0' if it appears before a decimal.
    - Ensures correct format for numerical values.

    :param value: str, raw extracted text
    :return: str, cleaned value
    """

    value = re.sub(r"\bO\.", "0.", value)  

    return value


def create_kdtree_from_boxes(boxes, image):
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
            x1, x2, y1, y2 =  int(box["x1"]), int(box["x2"]), int(box["y1"]), int(box["y2"])
            center_x = (box["x1"] + box["x2"]) / 2
            center_y = (box["y1"] + box["y2"]) / 2

            cropped_image = safe_crop(image, x1, y1, x2, y2, x_padding=5, y_padding=5)

            box_centers.append((center_x, center_y))
            box["text"] = fix_misread_numbers(pytesseract.image_to_string(cropped_image))
            filtered_boxes.append(box) 

    if not box_centers:
        return None, [] 

    return KDTree(np.array(box_centers)), filtered_boxes


def find_nearest_text(box, kdtree, text_boxes, search_size=2, search_radius=300):
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

    #print(classes[box["class_id"]])
    #print(f"Nearby Indices: {text_boxes[indices[0]]["text"]},  {text_boxes[indices[1]]["text"]},  {text_boxes[indices[2]]["text"]}")

    # Ensure indices is iterable (if only one result, convert to list)
    if isinstance(indices, np.int64):
        indices = [indices]

    output_label = ""
    output_value = ""
    i = 0
    for idx in indices:
        if (distances[i] > search_radius):
            break
        text_content = text_boxes[idx]["text"]

        # Define regex patterns for labels and values
        value_match = re.compile(r"^\d+(\.\d+)?\s?[a-zA-Z.]+$")
        label_match = re.compile(r"^[a-zA-Z]+[0-9]*$")

        # Greedy based algorithm 
        if value_match.match(text_content) and output_value == "":
            output_value = text_content
            print(output_value)
        if label_match.match(text_content) and output_label == "":
            output_label = text_content
            print(output_value)

        i += 1

    return output_label, output_value

def assign_values(bounding_boxes, text_kdtree, text_boxes, search_size=2, search_radius=300):
    for box in bounding_boxes:
        if box["class_id"] != 2 and box["class_id"] != 1:
            output_label, output_value = find_nearest_text(box, text_kdtree, text_boxes, search_size, search_radius)
            box["label"] = output_label
            box["value"] = output_value
    return bounding_boxes

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