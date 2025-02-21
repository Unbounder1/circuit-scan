import os
import json
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# Paths
root_base_path = 'C:\\Users\\Ryan\\Downloads\\cghd'
yolo_labels_path = os.path.join(root_base_path, 'yolo_labels')
compiled_images_path = os.path.join(root_base_path, 'compiled_images')
os.makedirs(yolo_labels_path, exist_ok=True)
os.makedirs(compiled_images_path, exist_ok=True)

# Load classes
with open(os.path.join(root_base_path, 'classes.json')) as f:
    classes = json.load(f)


# Function to convert VOC bounding boxes to YOLO format
def convert_voc_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# Iterate through all drafter folders
for drafter_folder in os.listdir(root_base_path):
    if not drafter_folder.startswith('drafter_'):
        continue

    annotations_path = os.path.join(root_base_path, drafter_folder, 'annotations')
    images_path = os.path.join(root_base_path, drafter_folder, 'images')

    # Iterate through annotation files
    for annotation_file in os.listdir(annotations_path):
        if not annotation_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(annotations_path, annotation_file))
        root = tree.getroot()

        image_file = root.find('filename').text
        image_path = os.path.join(images_path, image_file)

        # Copy image to compiled images folder
        shutil.copy(image_path, compiled_images_path)

        # Get image dimensions
        img = Image.open(image_path)
        w, h = img.size

        # Prepare YOLO label file
        yolo_label_file = os.path.splitext(annotation_file)[0] + '.txt'
        yolo_label_path = os.path.join(yolo_labels_path, yolo_label_file)

        with open(yolo_label_path, 'w') as yolo_file:
            for obj in root.findall('object'):
                label = obj.find('name').text
                class_id = classes[label]

                xml_box = obj.find('bndbox')
                box = (
                    float(xml_box.find('xmin').text),
                    float(xml_box.find('xmax').text),
                    float(xml_box.find('ymin').text),
                    float(xml_box.find('ymax').text)
                )

                yolo_box = convert_voc_to_yolo((w, h), box)

                yolo_file.write(f"{class_id} {' '.join([str(a) for a in yolo_box])}\n")

print("Conversion to YOLO format and image compilation completed!")