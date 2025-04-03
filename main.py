from flask import Flask, request, jsonify
import cv2
cv2.setNumThreads(1)
import numpy as np
import base64

import lib.process_image as p
import lib.node_connections as n
import lib.assign_values as a
import lib.convert_ltspice as c


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


    image_bytes = base64.b64decode(input["input"]) # into bytes

    image_array = np.frombuffer(image_bytes, np.uint8) # into array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # into cv2

    p.resize_image(image, max_size=1000)

    bounding_boxes = p.process_image(image, threshold=0.5) 
    bounded_image = p.process_bounding_box(bounding_boxes, image)
    scale = p.normalize_image(bounding_boxes, standard_x=100, standard_y=100)
    graph = n.node_graph(bounding_boxes, bounded_image, scalar=scale, threshold_min=160, threshold_max=255)
    text_kdtree, text_boxes = a.create_kdtree_from_boxes(bounding_boxes, graph.image)
    bounding_boxes = a.assign_values(bounding_boxes, text_kdtree, text_boxes)
    ltspice_file_str = c.graph_to_ltspice(graph.adjacency_list, bounding_boxes, image)

    # Draw bounding boxes
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Draw index number on the RGB image (red text)
        cv2.putText(image, str(i) + ": " + str(int(box["confidence"]*100)) + "%", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    _, buffer = cv2.imencode(".png", image)  # Encode image to PNG format
    ml_plot = base64.b64encode(buffer).decode("utf-8")

    graph_plot = str(graph)

    return {
        "ltspice": ltspice_file_str,
        "mlplot": ml_plot,
        "graph": graph_plot
    }

