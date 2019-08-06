# **Traffic Sign Recognition** 

## Writeup

### This file describes the work in the Project "Traffic Sign Recognition"

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./traffic_signs/road_works_.jpg "Road Works Sign"
[image2]: ./traffic_signs/roundabout_.jpg "Roundabout Sign"
[image3]: ./traffic_signs/speed_lim_80_.jpg "Speed Limit 80 Sign"
[image4]: ./traffic_signs/traffic_signal_.jpg "Traffic Signal Sign"
[image5]: ./traffic_signs/yield_.jpg "Yield Sign"
[image6]: ./examples/example_sign.png "Example Sign"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  



### Data Set Summary & Exploration

#### A basic summary of the data set

I used the python standard library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43



### Design and Test a Model Architecture

#### Preprocessing of the image data

One example of the traffic sign data set is
![alt text][image6] 

Within the preprocessing step I normalized the data to a range of -1 ... 1 out of 0 ... 255 by subtracting the mean (128) and dividing by half of the maximum value (128). This step was applied to the train, validation and test data set. 

Grayscaling was also tested but turned out to be not as performant as having three color layers.  


#### The model architecture is the well-known LeNet designed by Yann LeCun

My model of the LeNet consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16  				|
| Convolution 5x5	    | 1x1 stride,  outputs 10x10x16					|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten				| output 400									|
| Fully connected		| output 120   									|
| RELU					|												|
| Fully connected		| output 84   									|
| RELU					|												|
| Fully connected		| output 43   									|
|						|												|


#### Training the model 

The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used various functions of the TensorFlow library. First, I calculated the cross entropy of the output logits of the LeNet. For the optimization, the adam optimizer was used. 

The following hyperparameters were chosen:
EPOCHS = 50
BATCH_SIZE = 128
LEARN_RATE = 0.0025


#### The approach taken for finding a solution and getting the validation set accuracy to be at least 0.93

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.939
* test set accuracy of 0.928


As written above, the LeNet was chosen for the Traffic Sign Recignition. The model works extremely well with the traning data. However, with the validation set and the test set, the accuracy is below training set. 
This implicated that we have the effect of overfitting. The dropout method was also tested to prevent from overfitting. Yet, it turned out the the overall performance goes down rather than up with dropout. 

Finding methods to prevent the model from overfitting would be the next steps to improve the performace. 


### Test a Model on New Images

#### Five German traffic signs found on the web

Here are five German traffic signs that I found on the web. For each image, the quality is discussed in the following and the points that might cause problems in the recognition are outlined. 

![alt text][image1]
Image 1: A problem with this image could be, that the top part of the sign is cut off. So we do not see the whole sign in the image.


![alt text][image2]
Image 2: Critical could be that the sign is not round in the image but something like oval. Further, the background does not differ a lot from the sign itself.


![alt text][image3]
Image 3: This image looks quite clear and good to recognize. However, the very light background may cause problems and the model has to recognize especially the 8 which is quite simial to the numbers 0,3,5,6 and 9


![alt text][image4]
Image 4: In the traffic signal image, the background of the image is quite diffuse, which might disrupt the recognition. 


![alt text][image5]
Image 5: Just like image 2, the yield image is distorted a lot. Apart from that, the background of the sign is diffuse and consists of different colors at diffentent areas of the image. 


#### The model's predictions on these new traffic signs
and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).



Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work     		| Keep left   									| 
| Roundabout mandatory 	| Roundabout mandatory							|
| Speed limit (80km/h)	| Speed limit (60km/h)							|
| Traffic signals		| Traffic signals				 				|
| Yield     			| Yield             							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This result lies below the expected performance of the test set with about 93 %.

However, the only faulty recognized image was the "Speed limit (80km/h)" Sign which was classified as "Speed limit (60km/h)". So the digit 8 was classified as a 6 which is quite similar. However, this should not happen in a real-life traffic sign recognition for an self-driving vehicle. 

#### How certain is the model when predicting on each of the five new images by looking at the softmax probabilities for each prediction

For the first image, the model is about 45 % sure that this is a Keep Left sign (probability of 0.45), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			| Keep Left   									| 
| .39     				| Speed limit (50km/h)							|
| .34					| No vehicles   								|
| .33     				| Turn right ahead								|
| .29					| Road work      								|



For the second image the model is 33 % sure, that it is a roundabout mandytory sign (probability of 0.33), and the image does contain a roundabount mandytory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .33         			| Roundabout mandatory							| 
| .22     				| Go straight or right							|
| .09					| Speed limit (80km/h)  						|
| .08     				| Speed limit (60km/h)							|
| .08					| Keep right        							|



For the third image the model is 66 % sure, that it is a Speed limit (60km/h) sign (probability of 0.66), and the image does contain a Speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .66         			| Speed limit (60km/h)							| 
| .57     				| Speed limit (80km/h)							|
| .35					| Speed limit (30km/h)							|
| .17     				| Speed limit (50km/h)							|
| -.1					| Go straight or right							|



For the fourth image the model is 60 % sure, that it is a Traffic Signals sign (probability of 0.60), and the image does contain a Traffic Signals sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Traffic Signals   							| 
| .29     				| Vehicles over 3.5 metric tons prohibited		|
| .21					| Bicycles crossing   							|
| .12     				| Priority road                      			|
| -.02					| No passing for vehicles over 3.5 metric tons	|



For the fifth image the model is 40 % sure, that it is a Yield sign (probability of 0.4), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .40         			| Yield             							| 
| .21     				| End of speed limit (80km/h)       			|
| .25					| Priority road     							|
| .08     				| Speed limit (60km/h)               			|
| .07					| End of no passing  							|

