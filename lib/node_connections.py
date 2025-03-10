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

class node_graph:
    '''
    boxes: Bounding boxes in a numpy array -> NO TEXT BOXES
    image: Processed binary image in cv2 format
    kdtree: Created kdtree from input boxes
    adjacency_list: List of nodes with array of 
    scalar: multiplication factor

    DFS METHOD:
        For each node, find all bounding boxes connected directly 
        Add to adjacency list (dictionary) keys -> node; value -> set of connections
    '''
    def __init__(self, bounding_boxes, image, scalar=1, threshold_min=160, threshold_max=255):
        self.scalar = scalar
        self.kdtree, self.boxes = self.create_kdtree_from_boxes(bounding_boxes)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, self.image = cv2.threshold(gray_image, threshold_min, threshold_max, cv2.THRESH_BINARY)  # Ensure binary image --SET TO 250 FOR NO CUTOFF-- (make variable later)

        # Set up using DFS: ------
        self.adjacency_list = {}

        self.create_graph_from_image()

    def __str__(self):
        self.visualize_adjacency_list(self.adjacency_list)
        return "Visualized adjacency list in matlab"
    
    def create_graph_from_image(self):
        for i in range(0, len(self.boxes)): # for all boxes
            if self.boxes[i]["class_id"] == 1:
                continue

            visited = np.zeros(self.image.shape[:2], dtype=bool) # visited 2d array
            stack = [] # to visit stack
            
            self.adjacency_list[i] = set() 
            box = self.boxes[i]

            box_center = [(box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2]  # (x, y)

            stack.append((int(box_center[1]), int(box_center[0])))  # Push as (row, col)
            visited[int(box_center[1]), int(box_center[0])] = True  # Mark as visited
            
            while stack:
                #print (len(stack))
                cur_pixel = stack.pop()  # cur_pixel is (row, col)
                
                searchRadius = range(-1, 2) # Search radius (pixels) rn its -1 to 1
                directions = [(dy, dx) for dy in searchRadius for dx in searchRadius if (dy, dx) != (0, 0)] # all within 2 pixels
                
                for dy, dx in directions:
                    new_row, new_col = cur_pixel[0] + dy, cur_pixel[1] + dx
                    #print(self.image[new_row, new_col])
                    
                    # Check bounds: rows vs. columns
                    if 0 <= new_row < self.image.shape[0] and 0 <= new_col < self.image.shape[1]:
                        if visited[new_row, new_col]:
                            continue
                        
                        if self.image[new_row, new_col] == 0:
                            visited[new_row, new_col] = True
                            # note is_pixel_in_boxes_kdtree expects (x, y)
                            isContained, idx = self.is_pixel_in_boxes_kdtree(new_col, new_row)
                            if isContained and idx != i and self.boxes[idx]["class_id"] != 1: # is contained, id isnt itself, id isnt text
                                distance = self.get_direction_and_distance(box, self.boxes[idx])
                                self.adjacency_list[i].add((idx, distance))
                                continue
                            stack.append((new_row, new_col))

    def get_direction_and_distance(self, box1, box2):
        # Calculate centers of both boxes.
        c1 = ((box1["x1"] + box1["x2"]) / 2, (box1["y1"] + box1["y2"]) / 2)
        c2 = ((box2["x1"] + box2["x2"]) / 2, (box2["y1"] + box2["y2"]) / 2)
        
        # Compute differences between centers.
        dx_center = c2[0] - c1[0]
        dy_center = c2[1] - c1[1]

        scalar = self.scalar  # SCALE OUTPUT 
        
        # Compute widths and heights of the boxes.
        box1_width = box1["x2"] - box1["x1"]
        box2_width = box2["x2"] - box2["x1"]
        box1_height = box1["y2"] - box1["y1"]
        box2_height = box2["y2"] - box2["y1"]
        
        # Decide dominant axis.
        if abs(dx_center) >= abs(dy_center):
            # Horizontal is dominant.
            if dx_center >= 0:
                # box2 is mostly to the right of box1.
                # Compute horizontal gap: distance from box1's right edge to box2's left edge.
                gap = box2["x1"] - box1["x2"] if box1["x2"] < box2["x1"] else 0
                # Add half-widths of both boxes to shift from edge-to-edge to center-to-center.
                total_distance = (box1_width / 2) + gap + (box2_width / 2)
                return (scalar * total_distance, 0)
            else:
                # box2 is mostly to the left of box1.
                gap = box1["x1"] - box2["x2"] if box2["x2"] < box1["x1"] else 0
                total_distance = (box2_width / 2) + gap + (box1_width / 2)
                return (scalar * -total_distance, 0)
        else:
            # Vertical is dominant.
            if dy_center >= 0:
                # box2 is mostly below box1.
                gap = box2["y1"] - box1["y2"] if box1["y2"] < box2["y1"] else 0
                total_distance = (box1_height / 2) + gap + (box2_height / 2)
                return (0, scalar * total_distance)
            else:
                # box2 is mostly above box1.
                gap = box1["y1"] - box2["y2"] if box2["y2"] < box1["y1"] else 0
                total_distance = (box2_height / 2) + gap + (box1_height / 2)
                return (0, scalar * -total_distance)
        
    def create_kdtree_from_boxes(self, boxes):
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

    def is_pixel_in_boxes_kdtree(self, x, y, threshold=1): # threshold = amt of error for bounding boxes in px
        """
        Check if a pixel (x, y) is within a bounding box using KDTree for fast lookup.
        
        :param x, y: Coordinates of the pixel.
        :param kdtree: Prebuilt KDTree from bounding box centers.
        :param boxes: Original bounding boxes array (x1, y1, x2, y2).
        :param threshold: Distance threshold for fast lookup.
        :return: True if pixel is inside any bounding box, else False.
        """
        _, idx = self.kdtree.query((x, y), k=1)  # Find nearest bounding box center
        x1, y1, x2, y2 = self.boxes[idx]["x1"], self.boxes[idx]["y1"], self.boxes[idx]["x2"], self.boxes[idx]["y2"]  # Retrieve closest box
        return (x1 - 1 <= x <= x2 + threshold) and (y1 - 1 <= y <= y2 + threshold), idx # Final check, bounding +1 pixel
    
    def visualize_adjacency_list(self, adj_list):
        """
        Visualizes the adjacency list as a graph using networkx.
        Each edge stores a distance vector (dx, dy). For example:
            (50, 0) means the neighbor is 50 pixels to the right.
            (0, 30) means the neighbor is 30 pixels above.
        
        This version ensures that every node is assigned a position,
        even in a disconnected graph.
        """
        positions = {}
        
        # For every node in the adjacency list, if it hasn't been assigned a position,
        # perform a BFS starting from that node.
        for start in adj_list.keys():
            if start in positions:
                continue
            positions[start] = (0, 0)  # assign a default position for the new component
            queue = [start]
            while queue:
                node = queue.pop(0)
                current_pos = positions[node]
                for neighbor in adj_list[node]:
                    neighbor_id, distance = neighbor  # distance is a tuple (dx, dy)
                    if neighbor_id not in positions:
                        # Note: subtracting distance[1] from current y to flip the y-axis if needed.
                        new_pos = (current_pos[0] + distance[0], current_pos[1] - distance[1])
                        positions[neighbor_id] = new_pos
                        queue.append(neighbor_id)
        
        # Build the graph.
        G = nx.Graph()
        for node in adj_list.keys():
            G.add_node(node)
        for node, neighbors in adj_list.items():
            for neighbor in neighbors:
                neighbor_id, distance = neighbor
                G.add_edge(node, neighbor_id, weight=distance)
        
        # Draw the graph using our computed positions.
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, positions, node_color='lightblue', node_size=700)
        nx.draw_networkx_edges(G, positions, edge_color='black')
        nx.draw_networkx_labels(G, positions, font_size=10)
        
        plt.title("Adjacency List With Positions")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    import process_image as p
    
    # Load image
    image = cv2.imread('image4.png')
    image = p.resize_image(image)
    if image is None:
        print("Error: Could not load image.")
        exit(1)

    # Process image to get bounding boxes
    bounding_boxes = p.process_image(image, threshold=0.6)

    # Remove objects other than junctions (class_id = 1)
    for box in bounding_boxes:
        if box["class_id"] != 1:  # Ignore non-junctions
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            image[y1:y1 + h, x1:x1 + w] = (0, 0, 0)  # Set to black

    scale = p.normalize_image(bounding_boxes)

    # Create graph from processed image
    graph = node_graph(bounding_boxes, image, scalar=scale)

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