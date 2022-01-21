# ethnicity-recognition

## Objectives
1. Use Tensorflow to build a model that successfully identifies ethnicity at 90% confidence
2. Train the model to identify key attributes it can use in classifying individuals by race
3. Understand how to correctly preprocess and handle data to use in machine learning model.
4. Identify parameters that increase the recognition of racial features to apply to more effective models
5. Gain a better understanding why ML models misidentify people of color at such a high rate 

## Dataset

We used a Kaggle database called UTKFace which has over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. The data was preprocessed by first resizing each of the images into 200 by 200 pixels then identify the class of each image based on the embedded identifier for race in the file name. We broke the 20000 images into 6000 each for training, validation, and testing. In preprocessing the data we created different lists for each of the validation, test, and training data. We created a Face class which contained all the attributes associated with the “face” image and added that to a list of “faces”.

## Machine Learning Methodology
Maxpooling and a 2 layered neural network to train the model. Each face was flattened before being run through the model.

## Conclusion
We were able build a machine learning model that can successfully identify ethnicity at 93% accuracy. While improvements to the model could have been made with more time, we are happy with the results of our project. In future renditions of this project we plan to build our own dataset that we can train it to look for specific parameters that make racial classification easier and that can be implemented in others’ ML models to increase the accuracy of facial recognition across the industry. Although it was a relatively simple concept, the process of learning how to properly handle data and understand the math/structure behind models was a good experience. We also have plans to train model on solely people of color, which will allow for increased accuracy in broader models when implemented.

## Motivation 
- Lohr , Steve. “Facial Recognition Is Accurate, If You’Re a White Guy.” The New York Times , 9 Feb. 2018, https://www.nytimes.com/ 2018/02/09/technology/facial-recognition-race-artificial- intelligence.html.

- Simonite, Tom. “The Best Algorithms Still Struggle to Recognize Black Faces.” Wired, Conde Nast, 22 July 2019, https:// www.wired.com/story/best-algorithms-struggle-recognize-black- faces-equally/.

## Acknowledgements
We thank Duke Research Computing for the use of the Duke Compute Cluster for high-throughput computation. The Duke Data Commons storage is supported by the National Institutes of Health (1S10OD018164-01).
