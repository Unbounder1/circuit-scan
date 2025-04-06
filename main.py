from flask import Flask, request, jsonify
import cv2
cv2.setNumThreads(1)
import numpy as np
import base64

import lib.process_image as p
import lib.node_connections as n
import lib.assign_values as a
import lib.convert_ltspice as c

def get_config_value(config, key, default):
    return config.get(key, default)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/process_image", methods=['POST'])
def process_image():
    '''
    :return: 
            ltspice_file_str, 
            graph_plot(base64 image of adj list), 
            ml_out(base64 image of bounding boxes with labels)
    '''
    input = request.get_json()

    if "input" not in input:
        return jsonify({"error": "No image provided"}), 400
    
    process_image_threshold = get_config_value(input, 'process_image_threshold', 0.5)  # object detection accuracy threshold
    resize_image_max = get_config_value(input, 'resize_image_max', 1000)               # resize image size
    normalize_x = get_config_value(input, 'normalize_x', 80)                           # default goal resistor size-x
    normalize_y = get_config_value(input, 'normalize_y', 80)                           # default goal resistor size-y
    binary_threshold_min = get_config_value(input, 'binary_threshold_min', 160)          # min threshold when converting to binary
    binary_threshold_max = get_config_value(input, 'binary_threshold_max', 255)          # maximum threshold to convert to white
    kdtree_bounding_threshold = get_config_value(input, 'kdtree_bounding_threshold', 1)    # threshold for bounding boxes when calculating the kdtree
    grid_size = get_config_value(input, 'grid_size', 128)                              # default ltspice snapping grid size
    text_search_size = get_config_value(input, 'text_search_size', 2)                    # number of text options to search per component
    text_search_radius = get_config_value(input, 'text_search_radius', 300)              # maximum radius of search for text in components

    # Decode image from base64
    image_bytes = base64.b64decode(input["input"]) 
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Use configuration variables instead of hard-coded values
    p.resize_image(image, max_size=resize_image_max)
    bounding_boxes = p.process_image(image, threshold=process_image_threshold)
    bounded_image = p.process_bounding_box(bounding_boxes, image)
    scale = p.normalize_image(bounding_boxes, standard_x=normalize_x, standard_y=normalize_y)
    graph = n.node_graph(bounding_boxes, bounded_image, scalar=scale, threshold_min=binary_threshold_min, threshold_max=binary_threshold_max)
    
    text_kdtree, text_boxes = a.create_kdtree_from_boxes(bounding_boxes, graph.image)
    bounding_boxes = a.assign_values(bounding_boxes, text_kdtree, text_boxes, search_size=text_search_size, search_radius=text_search_radius, )
    ltspice_file_str = c.graph_to_ltspice(graph.adjacency_list, bounding_boxes, grid_size=grid_size)

    # Draw bounding boxes with index and confidence percentage
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.putText(image, f"{i}: {int(box['confidence']*100)}%", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Encode the image with bounding boxes to PNG and then to base64
    _, buffer = cv2.imencode(".png", image)
    ml_plot = base64.b64encode(buffer).decode("utf-8")
    graph_plot = str(graph)

    return {
        "ltspice": ltspice_file_str,
        "mlplot": ml_plot,
        "graph": graph_plot
    }
