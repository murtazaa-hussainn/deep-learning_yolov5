import os
import xml.etree.ElementTree as ET
from PIL import Image

class_id_to_name_mapping = {
    "0": "AIRPLANE",
    "1": "BIRD",
    "2": "DRONE",
    "3": "HELICOPTER"
}

def create_pascal_voc_xml(filename, objects, img_size, folder='annotations'):
    # Create the file structure
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = filename

    # Add objects (bounding boxes) to the annotation
    for obj in objects:
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(img_size[0])
        ET.SubElement(size, 'height').text = str(img_size[1])
        ET.SubElement(size, 'depth').text = str(img_size[2])  # Assuming RGB images
        obj_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(obj_elem, 'name').text = class_id_to_name_mapping[str(obj['class_id'])]  # Adjust with actual class name or ID if possible
        bndbox = ET.SubElement(obj_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(obj['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(obj['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(obj['ymax'])

    # Create a new XML file with the results
    mydata = ET.tostring(annotation)
    myfile = open(os.path.join(folder, filename.replace('png', 'xml')), "wb")
    myfile.write(mydata)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_path = os.path.join(script_dir, 'images')
    labels_path = os.path.join(script_dir, 'labels')
    xml_annotations_path = os.path.join(script_dir, 'xml_annotations')
    # images_path = 'images/'
    # labels_path = 'labels/'
    # annotations_path = 'annotations/'

    # Ensure the annotations folder exists
    if not os.path.exists(xml_annotations_path):
        os.makedirs(xml_annotations_path)

    for image_file in os.listdir(images_path):
        # Assuming label file name is the same as the image file name
        label_file = image_file.replace('png', 'txt')

        # Open image to get its size
        img_path = os.path.join(images_path, image_file)
        with Image.open(img_path) as img:
            img_size = img.size + (3,)  # Width, Height, Depth
        
        # Read label file and convert to required object format
        objects = []
        print(os.path.join(labels_path, label_file))
        with open(os.path.join(labels_path, label_file), 'r') as file:
            for line in file.readlines():
                class_id, xmin, ymin, xlen, ylen = line.strip().split()
                xmin = float(xmin)
                ymin = float(ymin)
                xmax = float(xmin) + float(xlen)
                ymax = float(ymin) + float(ylen)
                objects.append({'class_id': class_id, 'xmin': xmin, 'ymin': ymin , 'xmax': xmax, 'ymax': ymax})

        create_pascal_voc_xml(image_file, objects, img_size, xml_annotations_path)


if __name__ == '__main__':
    main()