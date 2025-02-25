import cv2
import numpy as np
from scipy.spatial import KDTree
import networkx as nx
import matplotlib.pyplot as plt

class node_graph:
    '''
    boxes: Bounding boxes in a numpy array -> NO TEXT BOXES
    image: Processed binary image in cv2 format
    kdtree: Created kdtree from input boxes
    adjacency_list: List of nodes with array of 

    DFS METHOD:
        For each node, find all bounding boxes connected directly 
        Add to adjacency list (dictionary) keys -> node; value -> set of connections
    '''
    def __init__(self, bounding_boxes, image):
        self.kdtree, self.boxes = self.create_kdtree_from_boxes(bounding_boxes)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, self.image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)  # Ensure binary image --SET TO 250 FOR NO CUTOFF-- (make variable later)

        # Set up using DFS: ------
        self.adjacency_list = {}

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
                                self.adjacency_list[i].add(idx)
                                continue
                            stack.append((new_row, new_col))

    def __str__(self):
        self.visualize_adjacency_list(self.adjacency_list)
        return "Visualized adjacency list in matlab"
        
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

    def is_pixel_in_boxes_kdtree(self, x, y, threshold=50):
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
        return (x1 - 1 <= x <= x2 + 1) and (y1 - 1 <= y <= y2 + 1), idx # Final check, bounding +1 pixel
    
    def visualize_adjacency_list(self, adj_list):
        """
        Visualizes the adjacency list as a graph using networkx.

        :param adj_list: Dictionary where keys are node indices and values are sets of connected nodes.
        """
        G = nx.Graph()  # Create an empty graph

        # Add nodes and edges from the adjacency list
        for node, neighbors in adj_list.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)  # Add connection between nodes

        # Draw the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)  # Spring layout for better visualization
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)

        plt.title("Adjacency List Visualization")
        plt.show()

if __name__ == "__main__":
    import process_image as p
    
    # Load image
    image = cv2.imread('image.png')
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

    # Create Graph from processed image
    graph = node_graph(bounding_boxes, image)

    # Convert the graph's binary image to RGB for colored text
    rgb_image = cv2.cvtColor(graph.image, cv2.COLOR_GRAY2BGR)

    # Draw indexes on the RGB image
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        
        # Calculate correct center of each bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Draw index number on the RGB image (red text)
        cv2.putText(rgb_image, str(i), (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display both OpenCV image and Matplotlib graph visualization
    cv2.imshow('1-Pixel Contours with Indexes', rgb_image)
    cv2.waitKey(1)  # Prevent window from closing immediately

    print(graph)  # Display adjacency list in the console

    # cv2.waitKey(0)  # Wait for user keypress before closing windows
    # cv2.destroyAllWindows()