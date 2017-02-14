#**Behavioral Cloning** 

##Writeup Report

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
By using only training data collected around the first (left) track, the trained model eventually managed to drive the car around both tracks in the standard simulator (but failed to drive around the more challenging track in the beta simulator).

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

A convolution neural network with depths between 24 and 64 is employed in this project.  
The model includes RELU layers (for convolutional layers) and tanh (for fully-connected layers) to introduce nonlinearity.
The network consists of a total of 13 layers (excluding the final output layer), including 5 convolutional layers, 3 pooling layers and 3 fully connected layers. 

| Layer # | Layer Type   | parameter   |  activation |
| ------------- |-------------| ------|------------------------|
| 1      | cropping      | 50 rows from top, 20 from bottom | NA |
| 2      | normalization|   [-1, 1] after normalization| NA |
| 3      | convolutional      |    5×5 kernel, 24 filters | RELU |
| 4      | maxpooling      |    2×2  | NA |
| 5      | convolutional      |    5×5 kernel, 36 filters | RELU |
| 6      | maxpooling      |    2×2  | NA |
| 7      | convolutional      |    5×5 kernel, 48 filters | RELU |
| 8      | maxpooling      |    2×2  | NA |
| 9      | convolutional      |    3×3 kernel, 64 filters | RELU |
| 10      | convolutional      |    3×3 kernel, 64 filters | RELU |
| 11      | fully-connected      |  100 hidden units | tanh |
| 12      | fully-connected      |  50 hidden units | tanh |
| 13      | fully-connected      |  10 hidden units | tanh |

The first layer is a cropping layer, which crops the not-so-important parts of the image (the sky and the hood of the car); 
The second layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process, the data is normalized in the model using a Keras lambda layer; 
The convolutional layers were designed to perform feature extraction, and the fully-connected layers are meant for regression that maps the high level features of the images to the steering angle command. 


####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. With the current model, there is no overfitting issues (in fact the training loss almost always a little larger than the validation loss...) even without dropout layers or other regularization method. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. For the issue of recovering the car from being off-center, I used the multiple camera method, i.e., using all three sets of images from the left, the center and the right cameras, and "correcting" the steering angles corresponding to the left and the right images. 
For details about how I created the training data, please see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
