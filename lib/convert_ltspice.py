def get_rotation_precise(box):
    """
    Returns the LTspice rotation string based OBB object detection ---- implement later
    :param image: OpenCV input array
    :param box: x1,x2,y1,y2 coordinates for bounding box
    :param kdtree: KDTree for bounding boxes of OBB output

    """
    if ("theta" in box):
        theta = box["theta"]
    else:
        theta = 0
    
    theta += 0.78

    theta /= 1.57 

    if (int(theta) == 0):
        return "R0"
    elif (int(theta) == 1.57):
        return "R90"
    elif (int(theta) == 3.14):
        return "R180"
    elif (int(theta) == 4.71):
        return "R270"
    else:
        return "R0"
def get_rotation(dx, dy):
    """
    Returns the LTspice rotation string based on the offset direction.
    (dy = 0)  => R0   (facing left/right)
    (dx = 0)  => R90  (facing up/down)
    """
    if dx == 0:
        return "R0"
    elif dy == 0:
        return "R90"
    else:
        return "R1"
    
def compute_positions_bfs(adj_list):
    """
    Computes absolute positions for each node using BFS.
    Also determines a rotation for each node based on the edge that discovered it.
    
    Returns:
      positions   -> dict {node_idx: (x, y)}
      orientations -> dict {node_idx: "R0"/"R90"/"R180"/"R270"}
    """
    positions = {}
    orientations = {}
    visited = set()

    for node in adj_list.keys():
        if node not in visited:
            positions[node] = (1000, 1000)
            queue = [node]
            visited.add(node)

            while queue:
                current = queue.pop(0)
                cx, cy = positions[current]
                for neighbor_idx, offset in adj_list.get(current, []):
                    dx, dy = offset
                    new_x = cx + dx
                    new_y = cy + dy
                    new_pos = (int(round(new_x)), int(round(new_y)))
                    if neighbor_idx not in visited:
                        positions[neighbor_idx] = new_pos

                        orientations[neighbor_idx] = get_rotation(dx, dy)
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)
                    else:
                        pass

    if positions:
        min_x = min(pos[0] for pos in positions.values())
        min_y = min(pos[1] for pos in positions.values())
        for node in positions:
            x, y = positions[node]
            positions[node] = (x - min_x, y - min_y)

    return positions, orientations

def snap_to_grid(pos, grid_size=256):
    x, y = pos
    return (round(x / grid_size) * grid_size, round(y / grid_size) * grid_size)

def graph_to_ltspice(adj_list, boxes, grid_size=128):
    """
    Converts a graph to an LTspice .asc file string by processing one node at a time.
    
    For each node:
      - Draws wires to its neighbors using Manhattan routing (horizontal then vertical).
      - Places the component symbol (if applicable) at the node with a fixed offset.
      
    This ensures that all wiring is strictly horizontal and vertical.
    """
    # Define LTspice classes and symbol mappings.
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
    class_to_symbol = {
        "resistor": "Res",
        "voltage.dc": "voltage",
        "voltage.ac": "voltage",
        "voltage.battery": "voltage",
        "gnd": "FLAG",
        "capacitor.unpolarized": "Cap",
        "capacitor.polarized": "Cap",
        "inductor": "Ind",
        # ... add more mappings as needed ...
    }
    # Define pin offsets if the LTspice symbol's electrical pin is not at (0,0).
    symbol_pin_offsets = {
        "Res": {"R0": (16, 16), "R90": (-48, 16)},
        "voltage": {"R0": (0, 0), "R90": (0, 0), "R180": (0, 0), "R270": (0, 0)},
        "Cap": {"R0": (16, 32), "R90": (16, 16)},
        "Ind": {"R0": (16, 16), "R90": (-48, 16)},
        # ... add more components as needed ...
    }
    rotation_components = {
        "voltage.dc", "voltage.ac", "voltage.battery",
        "diode", "diode.light_emitting", "diode.thyrector", "diode.zener",
    }
    
    # Get node positions and orientations via BFS.
    positions, orientations = compute_positions_bfs(adj_list)
    # Snap node positions to a grid (this makes them exact multiples of grid_size).
    positions = {node: snap_to_grid(pos, grid_size=grid_size) for node, pos in positions.items()}
    
    ltspice_lines = []
    ltspice_lines.append("Version 4")
    ltspice_lines.append("SHEET 1 2000 2000")
    
    drawn_edges = set()
    
    # Process nodes one by one (sorted for deterministic output).
    for node in sorted(positions.keys()):
        x, y = positions[node]
        
        # For each neighbor from this node, draw the connecting wires.
        for neighbor_idx, _ in adj_list.get(node, []):
            edge = tuple(sorted((node, neighbor_idx)))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            nx, ny = positions[neighbor_idx]
            # Always use Manhattan routing: horizontal then vertical.
            ltspice_lines.append(f"WIRE {x} {y} {nx} {y}")
            ltspice_lines.append(f"WIRE {nx} {y} {nx} {ny}")
        
        # Place the component symbol if this node has an associated box.
        if node < len(boxes):
            box = boxes[node]
            class_name = classes[box["class_id"]]
            # Skip non-component nodes.
            if class_name not in ["text", "junction"] and class_name in ["GND", "VSS"]:
                ltspice_lines.append(f"FLAG {x} {y} 0")
            if class_name not in ["text", "junction"]:
                symbol = class_to_symbol.get(class_name, "Unknown")

                # Get rotation
                if (class_name in rotation_components):
                    rotation = get_rotation_precise(box)
                else:
                    rotation = orientations.get(node, "R0")

                pin_offset = symbol_pin_offsets.get(symbol, {}).get(rotation, (0, 0))

                # Adjust the position by subtracting the pin offset.
                adjusted_x = x - pin_offset[0]
                adjusted_y = y - pin_offset[1]
                ltspice_lines.append(f"SYMBOL {symbol} {adjusted_x} {adjusted_y} {rotation}")

                # Set the instance name based on the symbol type.
                if box["label"] != "":
                    inst_name = box["label"]
                elif symbol == "Res":
                    inst_name = f"R{node}"
                elif symbol == "voltage":
                    inst_name = f"V{node}"
                elif symbol == "FLAG":
                    inst_name = "GND"
                else:
                    inst_name = f"{symbol}{node}"
                ltspice_lines.append(f"SYMATTR InstName {inst_name}")

                if box["value"] != "":
                    ltspice_lines.append(f"SYMATTR Value {box["value"]}")
    
    return "\n".join(ltspice_lines)

if __name__ == "__main__":
    import process_image as p
    import node_connections as n
    import cv2
    import assign_values as a

    # Load image
    image = cv2.imread('image3.png')
    image = p.resize_image(image)
    if image is None:
        print("Error: Could not load image.")
        exit(1)

    # Process image to get bounding boxes
    bounding_boxes = p.process_image(image, threshold=0.6)

    bfs_image = p.process_bounding_box(bounding_boxes, image)

    scale = p.normalize_image(bounding_boxes)

    # Create graph from processed image
    graph = n.node_graph(bounding_boxes, bfs_image, scalar=scale)

    bounding_boxes = p.associate_rotation(image, bounding_boxes,graph.kdtree)

    text_kdtree, text_boxes = a.create_kdtree_from_boxes(bounding_boxes, graph.image)

    bounding_boxes = a.assign_values(bounding_boxes, text_kdtree, text_boxes)

    # Generate LTspice schematic text
    ltspice_file_str = graph_to_ltspice(graph.adjacency_list, bounding_boxes, image)

    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        
        # Calculate correct center of each bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Draw index number on the RGB image (red text)
        cv2.putText(image, str(i) + ": ", (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    cv2.imshow('1-Pixel Contours with Indexes', image)

    print(graph)

    with open("test.asc", "w") as f:
        f.write(ltspice_file_str)
    # print(ltspice_file_str)