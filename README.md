Steps to perform to run this repo:
1. Data is stored in data folder (images and labels)
2. Run the command pip install -r requirements.txt to install all the required libraries
3. Run the python file: data/data-matching.py to check if all the labels are available against all the images
4. Once the above is completed, run the python file: data/label-annotation.py to convert the labels in pascal VOC format
5. Once the above completes its run, run the python file: data-preparation.py, it will divide the dataset into training and testing data
6. Once the above one is completed, the data is now in order and you can customise the config.json file and run train.py to train the model
7. Once the model is trained used the best.pt weights to test the data using val.py script
8. You can also detect the object using images and videos as source using detect.py file
