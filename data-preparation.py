import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import json
import shutil

# Load the configuration file
with open('config.json') as json_data_file:
    config = json.load(json_data_file)


# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = float(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {
    "AIRPLANE": 0,
    "BIRD": 1,
    "DRONE": 2,
    "HELICOPTER": 3
}

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict, annotations_path):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join(annotations_path, info_dict["filename"].replace("png", "txt"))
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))


#Utility function to move images 
def move_files_to_folder(list_of_files, script_dir, destination_folder):
    if not os.path.exists(os.path.join(script_dir, destination_folder)):
        os.makedirs(os.path.join(script_dir, destination_folder))
    for f in list_of_files:
        try:
            shutil.move(f, os.path.join(script_dir, destination_folder))
        except:
            print(f)
            assert False


def create_partitions(script_dir):
    # Read images and annotations
    images = [os.path.join(script_dir, 'images', x) for x in os.listdir(os.path.join(script_dir, 'images'))]
    annotations = [os.path.join(script_dir, 'annotations', x) for x in os.listdir(os.path.join(script_dir, 'annotations')) if x[-3:] == "txt"]

    images.sort()
    annotations.sort()

    # Split the dataset into train-valid-test splits 
    train_images, test_images, train_annotations, test_annotations = train_test_split(images, annotations, test_size = config["partition"]["test"], random_state = 1)
    if (config["partition"]["val"] > 0):
        val_images, test_images, val_annotations, test_annotations = train_test_split(test_images, test_annotations, test_size = config["partition"]["val"], random_state = 1)

    # Move the splits into their folders
    move_files_to_folder(train_images, script_dir, 'images/train')
    move_files_to_folder(train_annotations, script_dir, 'annotations/train/')
    move_files_to_folder(test_images, script_dir, 'images/test/')
    move_files_to_folder(test_annotations, script_dir, 'annotations/test/')
    if (config["partition"]["val"] > 0):
        move_files_to_folder(val_images, script_dir, 'images/val/')
        move_files_to_folder(val_annotations, script_dir, 'annotations/val/')
    return 'Partitions Created'


def main():
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    xml_annotations_path = os.path.join(script_dir, 'xml_annotations')
    annotations_path = os.path.join(script_dir, 'annotations')

    # Ensure the annotations folder exists
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    for file in os.listdir(xml_annotations_path):
        # Assuming label file name is the same as the image file name
        info_dict = extract_info_from_xml(os.path.join(xml_annotations_path,file))
        convert_to_yolov5(info_dict, annotations_path)

    create_partitions(script_dir)


if __name__ == '__main__':
    main()