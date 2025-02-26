# Circuit Scan

Converts images/drawings to LTSpice diagrams 

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