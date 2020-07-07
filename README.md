# Face-Recognition-with-Matlab


The aim of the project is to crate a Face Recognition System via through deep learning methods that will help in recognizing the face of users based on the image dataset 
that is provided to it. The system will first be trained with a set a images which will be used as the default classifier for comparison.

The classifiers were implemented with the following methods **HOG Features, ECOC Classifier etc, Viola Jones Algorithm**


## Load the Image information 
The system beigns by first collecting images of different people and stores them onto the database of the system. Each individual on the database will have 
atleast 7 different unique images of themselves. 
The `imageSet` which is an in-built MATLAB function will be used for grouping the images. 

## Detection and Extraction of the Image
Face detection is an important step in face recognition that corresponds the localization of the face in a given image. Once the face is detected it is cropped for recognition or a rectangle box will be overlayered on the identified persons image
![GitHub Logo](/images/all.jpg)

## Extraction using the HOG Feature 
The HOGs are a feature descriptor that been successfully used for object and pedestrian detection, represented as a single value vector as opposed to a set of feature vectors where each represents a region of the image. The results of the image shown illustrates the output result when the HOG feature extraction
method was used.
                  ![GitHub Logo](/images/pp2.jpg)
