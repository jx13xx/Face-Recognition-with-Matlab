% Takes image information from the Folder "Faces Data" groups all the
% images into one single object 
Face_set_Database = imageSet('FacesData','recursive')


%% Display Montage of First Face
figure;
montage(Face_set_Database(7).ImageLocation)
title('Images of Single Face of Person(17)')

%% Display Query Image and Database Side-Side
personToQuery = 7;
galleryImage = read(Face_set_Database(personToQuery),1);
figure;
for i =1:size(Face_set_Database,2)
    
    imageList(i) = Face_set_Database(i).ImageLocation(5);
end
subplot(1,2,1);imshow(galleryImage);
subplot(1,2,2);montage(imageList);
diff = zeros(1,9);

%% We split the Facedatabase into two sets the Training Sets and Test Set
[trainingSet, testSet] = partition(Face_set_Database,[0.8 0.2]);

%% Extract and display Histogram of Oriented Gradient Features for a Single Face HOG FEATURE EXTRACTION 
person = 21;
[hogFeature, visualization] = ...
    extractHOGFeatures(read(trainingSet(person),1));

figure;
subplot(1,2,1); imshow(read(trainingSet(person),1)); title('Input Face');
subplot(1,2,2); plot(visualization); title('HoG Feature');

%% To extract the HOG features from the Trainig set images ALL THE IMAGES 
% After extarction we put them in them for training in the classifier
% and update the feature count and label names
tic
training_Features = zeros(size(trainingSet,2)*trainingSet(1).Count,18144);
featureCount =1;
for i= 1:size(trainingSet,2)
    for j = 1:trainingSet(i).Count
        training_Features(featureCount,:) = extractHOGFeatures(read(trainingSet(i),j));
        trainingLabel{featureCount} = trainingSet(i).Description;
        featureCount = featureCount +1;
    end
    personIndex{i} = trainingSet(i).Description;
end

toc
%% Create 111 class Classifier using fitceoc using the EOC Classifier 

tic

 
 faceClassifier =(fitcecoc(training_Features,trainingLabel));


toc



%% Test Images from Test Set

person =33;
queryImage = read(testSet(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);
%Map back to training set to find identity 
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);


subplot(1,2,1); imshow(queryImage); title('Query Face');
subplot(1,2,2); imshow(read(trainingSet(integerIndex),1)); title('Matched Class')

%% Test First 5 people from Test Set
% same code but used for testing 1:5 peopel 
figure;
figureNum = 1;
for person=1:5
    for j = 1:testSet(person).Count
        queryImage = read(testSet(person),j);
        queryFeatures = extractHOGFeatures(queryImage);
        personLabel = predict(faceClassifier,queryFeatures);
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        integerIndex = find(booleanIndex);
        subplot(2,5,figureNum);imshow(imresize(queryImage,3));title('Query Face');
        subplot(2,5,figureNum+1);imshow(imresize(read(trainingSet(integerIndex),1),3));title('Matched Class');
        figureNum = figureNum+2;
        
    end
    figure;
    figureNum = 1;
end


































