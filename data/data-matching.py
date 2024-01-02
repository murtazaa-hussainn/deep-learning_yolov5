import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(script_dir, 'labels')
    images_path = os.path.join(script_dir, 'images')

    labels_files = [f.replace('.txt', '') for f in os.listdir(labels_path)]
    images_files = [f.replace('.png', '') for f in os.listdir(images_path)]

    print('Files with Labels present but Images absent:')
    for file in labels_files:
        if file not in images_files:
            print(file)
            os.remove(os.path.join(labels_path, file + '.txt'))
    
    print('Files with Images present but Labels absent:')
    for file in images_files:
        if file not in labels_files:
            print(file)
            os.remove(os.path.join(images_path, file + '.png'))


if __name__ == '__main__':
    main()