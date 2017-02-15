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

The model was trained and validated on different data sets to ensure that the model was not overfitting. Although during training the training loss is almost always a little larger than the validation loss even without dropout layers or other regularization method, seemingly indicating there is no overfitting issue, the model trained without the dropout layers perform much more erratic than the model with some dropout layers. I added 4 dropout layers, after layer 6, layer 10, layer 11 and 12, respectively. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (both the first and the second tracks).

####3. Model parameter tuning

The model used an adam optimizer (rmsprop was also tried and worked), so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. For the issue of recovering the car from being off-center, I used the multiple camera method, i.e., using all three sets of images from the left, the center and the right cameras, and "correcting" the steering angles corresponding to the left and the right images. 
For details about how I created the training data, please see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The model architecture usesd in this project is heavily based on the [Nvidia's CNN for self-driving cars] (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for obvious reasons: we are dealing with an almost-exactly-the-same problem as they and their network worked. Also their network (convolutional layers + fully-connected layers) makes sense for obvious reasons: we have raw camera images as our input, the features of which convolutional neural networks really excel to extract, that prompts the use of convolutional layers; then from the extracted high-level image features to the final output, i.e., the steering angle, it makes sense to use a fully-connected NN to do the mapping due to the regression nature of the probelm. 

Thus the design of the network itself becomes the easy part of this project: Nvidia's CNN for self-driving cars is employed with some slight modification. I did try to use less layers or add more layers, experiment with different parameters for each layer, but none of these efforts resulted in improved performance, and often resulted in a desfunction model. So I finally decided to stick with the parameters used in Nvidia's CNN.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. As I mentioned earlier, I never had any overfitting issue that manifested in the larger validation loss than the traning loss, rather, it's almost always the opposite (that is also kind of weird...). This was true when I trained the model with 6000+ samples, 11000+ samples, and 34000+ samples. That said, by adding some dropout layers, the performance of the model could be much better (w/o dropout layers the motion of the vehicle is rather erratic). However, an interesting note: when I first started (using less layers than the final architecture), I added dropout for each layer. I was stuck for a long time since the model output of the steering angle is always very small (never exceeded 0.1), even leaving me wonder if it's because of my model or the simulator is not working right. Finally I took out all the dropout layers and the trained model started to output comparable steering angle to the training output. 

I first trained my model using only the sample data set provided in the course resources. But that never resulted in a workable version of the CNN model. Which brings me to my conclusion after finishing this project: that the data used for the model training and the training process is the real key to obtain the final network. I'll detail the training data/process in the next section. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

For details of the final model architecture (model.py lines 78-138) including the layers used and layer sizes, please refer to the above table in the previous section.


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded about three or four laps on track one using center lane driving. Since I noticed that in the sample data given in the project resources the car was driven mostly clockwise, so I only did counter-clockwise driving. I have to say I didn't really master the "driving" of the car even with a XBox controller, so in the training data I eventually fed to the model optimizer, the car was sometimes driving kind of close to one side of the road and at one point even off the track (but I then drive it back on the track). But, that's the best I can do... in a limited time. Also I was kind of curiours how good a performance can I get with this kind of training data. 

So I got another 6000+ data samples (there were 8000+ data samples in the original sample data folder). Now the real key, or trick, of training a CNN to drive a car is how to address the issue of correcting the car when driven off the center as the project tip pointed out. Two methods were recommended, one is to use multiple camera images, the other is record only when the car is driven from the side to the center of the lane. Here I chose the first one (model.py line 40-55) since 1) it's more systematic and it's easier to record the data, you only have to record when you do center lane driving, with the second method, you have to record those times when you only drive from one side to the center and you have to do that many times; 2) with the second method you have to track the proportion of your side-to-center data and keep wondering when it'll be enough to do this, it won't be a problem if you use method 1; 3) with the multiple camera image method, you automatically get triple data samples. After choosing this method, the only problem left is to tune the "correction" steering angle paramter for left or right camera image. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. Adam optimizere was employed to train the model. (RMSProp was also tried)

