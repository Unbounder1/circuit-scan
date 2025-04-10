# Circuit Scan

Converts images/drawings to LTSpice diagrams 

## TODO
- Add contrast slider

## Function Usage

### Defaults

```c++
process_image_threshold=0.5 //object detection accuracy threshold
resize_image_max=1000 // resize image size
normalize_x=80 // default goal resistor size-x
normalize_y=80 // default goal resistor size-y
binary_threshold_min=160 // min threshold when converting to binary
binary_threshold_max=255 // maximum threshold to convert to white
kdtree_bounding_threshold=1 // default threshold for the bounding boxes when calculating the kdtree
grid_size=128 // default ltspice snapping grid size
text_search_size=2 // number of text options to search per component
text_search_radius=300 // the maximum radius of search for text in components
```

### Pseudocode Usage

```python
    import process_image as p
    import node_connections as n
    import assign_values as a
    import convert_ltspice as c
    import cv2

    # Load image
    image = cv2.imread('image')

    # process image
    image = p.resize_image(image) # making image resize to 1000 pixels

    # Get bounding boxes from YOLO model
    bounding_boxes = p.process_image(image, threshold=0.5) 

    # Apply bounding box to new image
    bounded_image = p.process_bounding_box(bounding_boxes, image)

    # Get image normalization scale (based on component size)
    scale = p.normalize_image(bounding_boxes)

    # Create adjacency list graph
    graph = n.node_graph(bounding_boxes, bounded_image, scalar=scale)

    # Create the rotation parameter for every component
    bounding_boxes = p.associate_rotation(image, bounding_boxes,graph.kdtree)

    # Debug Adjacency list creation && image
    cv2.imshow('1-Pixel Contours with Indexes', graph.image) # check binary thresholds
    print(graph) # show adjacency list with matplot

    # Get kdtree for text and also a seperate dictionary for just text 
    text_kdtree, text_boxes = a.create_kdtree_from_boxes(bounding_boxes, graph.image)

    # Add ["label"] and ["value"] to bounding_boxes
    bounding_boxes = a.assign_values(bounding_boxes, text_kdtree, text_boxes)
    
    # Take data and convert to ltspice 
    ltspice_file_str = c.graph_to_ltspice(graph.adjacency_list, bounding_boxes, image)





```
## Methodology

First take in input image:

![Example Input](./public/documentations1.png?raw=true "Input Image")

Apply machine learning OBB model[1] to detect bounding boxes of components, text, and junctions:

![ML output](./public/documentations2.png?raw=true "ML Output")

Create bounding boxes and fill in with black bounding boxes to ensure circuit continuity for next step, and binarize image:

![Bounding Box](./public/documentations3.png?raw=true "Bounding Box")

Using a pixel based DFS with KDTree for determining if pixel in (any) bounding, discover neighbors to each node and construct a x,y weighted adjacency list representing each of the nodes:

![Adjacency List](./public/documentations4.png?raw=true "Adjacency List")

Finally, using a table of symbol mapping and component offsets (components like resistors are positioned based on corners not center, unlike other components)'

![LTspice](./public/documentations5.png?raw=true "LTspice")

## File Structure 
```
├── README.md
├── convert_yolo.py # convert to YOLO format given dataset I used
├── lib
│   ├── convert_ltspice.py # convert LTSPICE
│   ├── find_contours.py # find contours
│   ├── node_connections.py # DFS graph creation process
│   └── process_image.py
├── main.py
├── models
│   ├── ... # all models
├── output.json
├── output.txt
├── processing.py
├── public
│   ├── ... # images
└── test.asc # ltspice output
```

## References

**[1]** H. Zhang, Y. Jiang, Y. Zhu, and T. Mitra, “Did you train it the same way? Replicability study on adversarial training,” arXiv preprint arXiv:2106.11559, Jun. 2021. [Online]. Available: https://arxiv.org/pdf/2106.11559.

## Notes

1. gevent uses monkey patching
	•	gevent replaces standard Python modules (like threading, socket, etc.) with async-friendly versions using greenlets (lightweight coroutines).
	•	This monkey-patching is great for I/O-bound apps (like APIs waiting on HTTP/database), but breaks things that depend on native threads or strict low-level behavior.

 2. torch, OpenCV, pytesseract use native threads or C extensions
	•	Libraries like torch, opencv, and tesseract are implemented in C/C++ and expect true POSIX threads and predictable memory handling.
	•	When monkey-patching threads (like threading.Lock) with gevent, those native libraries can break — leading to segfaults (SIGSEGV) or deadlocks.