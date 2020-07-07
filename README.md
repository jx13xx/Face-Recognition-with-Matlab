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

```MATLAB
person = 21;
[hogFeature, visualization] = ...
    extractHOGFeatures(read(trainingSet(person),1));

figure;
subplot(1,2,1); imshow(read(trainingSet(person),1)); title('Input Face');
subplot(1,2,2); plot(visualization); title('HoG Feature');

```
                  
<p align="center">
  <img width="460" height="400" src="/images/pp2.jpg">
</p>

## Classifier Training
 The `ECOC (Error CorrectingOutput codes)` basically converts multi-class classification problem into a binary classification problem with the help of various coding
schemes accompanied by a linear like Support Vector Machine (SVM).
In the Face recognition system the ECOC classifier is trained with the extracted features from the HOG.

```MATLAB
faceClassifier =(fitcecoc(training_Features,trainingLabel));
```
## Results from the Classifier 
Once the classifier is successfully created we can then test the trained classifier by providing a query image. The query image can be any image that is included in the face dataset. The classifier compares the query image with all the trained images and displays the output picture of the identified person.

<p align="center">
  <img width="460" height="600" src="/images/final_result.jpg">
</p>

