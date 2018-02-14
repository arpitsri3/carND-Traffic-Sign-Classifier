# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the writeup. Here's a link to the HTML file - https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.html
You're reading it! and here is a link to my [project code](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

The len and shape functipons were called to get the data size:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32.3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution in training dataset.

![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/stats_images/stat1.png)

Min vs Max
Label with least Data -  0  has  180  values.
Label with most Data -  2  has  2010  values.

A bar chart showing the data distribution in validation dataset.

![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/stats_images/stat2.png)


Min vs Max
Label with least Data -  0  has  30  values.
Label with most Data -  1  has  240  values.


A bar chart showing the data distribution in Test dataset.

![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/stats_images/stat3.png)

Min vs Max
Label with least Data -  0  has  60  values.
Label with most Data -  2  has  750  values.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the lesser the depth the more easily the network will be able to train itself. Also in the project1 (on finding lane lines) we have observed that grayscale was effective in cutting out the noise. 

Here is an example of a traffic signs images before and after grayscaling.

![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/stats_images/stat_color.png)

![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/stats_images/stat_gray.png)


As a last step, I normalized the image data because basically it is suggested in the checkpoints for the project, is fairly straightforward and effective, and also because data normalization is normal distribution is statistically relevant for our helping in training the network easily.

Initially I planned to augment data and had implemented equalize_histogram, crop and sharpen methods (Visible as part of markup in py notebook, but can be visualized in the HTML file), but I ran into issues during training as I have done the project on my local instead of AWS (there's some issue with credits), so I decided to drop the augmentation for now and focus on implementation of architecture to get on with the project.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried implementing the LeneT LAB Solutionarchitecture initially. It worked kind of okay at first but the accuracy maxed out at 90-91%. After going through the paper on the baseline model for the problem (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), mentioned in the project, I made some changes as shown below . My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| (32x32x1)     | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 5x5		| (1x1 stride, valid padding, outputs 1x1x400	|
| RELU					|												|
| FLATTEN				| 5x5x16 and 1x1x400 Layers       				|
| Concatenate			| 400+400=800 									|
| Dropout 				| keep_prob=0.5 								|
| Fully connected		| 800->43      									|
|						|												|
|						|												|
 
1. An extra conolution layer made a noticeable difference in accuracy. Output dimension when converted to 1x1 gave a per pixel data.
2. Flattening and concatenating pooling and subsequent convolution layer gave good results when applying dropout on the data.
3. Finally a fully connected layer at the end for 43 class output . 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For training purpose:

Type of optimizer - AdamOptimizer
Batch Size - 128
Epochs - 30
Learning rate - 0.0005

Initially I used the exact same implementation as for the LeNet lab , but later some changes such as supplying 'keep_prob' via feed_dict, increasing the number of epochs to 30 and decreasing the learning rate to 0.0005 was how i could get the best result on my machine. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 93.3%
* validation set accuracy of 94.3% 
* test set accuracy of 93.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture i tried was LeNetT LAB architecture. I chose it as it was the recommended architecture and again due to the machine constraint i couldn't actually consider working on a complicated architecture. 

* What were some problems with the initial architecture?
The main problem with the initial architecture was that after around 90-91% the accuracy maxed out and even after trying till upto 30 epochs there wasn't much difference (say till max 92%)

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Adjusting the architecture was a little bit hit and trial and little bit looking into the classroom for helpful points. For example- adding the extra convolution layer to get a 1x1x400 output came into mind because of the lesson -11- 11.6, 11.9 and 11.28 where it is mentioned that try to decrease the dimensionality while progressing towards the classifier, along with advantages og 1*1 dimension.
Similarly, concatenating the two layers before dropout was as per the intuitions from the point mentioned in Lesson 10.18 and 10.19.

* Which parameters were tuned? How were they adjusted and why?
The main parameter which i modified was increasing the number of epochs to 30 and decreasing the learning rate to 0.0005. This is simply because when i tried with a learning rate of 0.001 the model reached a 92% accuracy quickly but got stuck at that point. Again, on decreasing the learning rate i adjusted the epochs.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
As evident from the bar_graphs for all the datasets , its obvious that certain labels have a significantly larger footprint than others. Thus, we might even say that there's an inherent bias in the dataset. This may be tackled , as discussed in the classroom, by applying dropout (so as to randomly filter data ) which may be helpful in reducing the bias and doing pooling. CNNs are recommended here as this is an image dataset on which the network needs to be trained.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet Architecture
* Why did you believe it would be relevant to the traffic sign application?
In the leNet lab it was already used on the MNIST dataset which is a relevant observation in favor of the architecure.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The model was modified somewhat to achieve the required accuracy of 93%. The model is working okay I'd say, but not well, and there is  room for improvement, which I hope to work on personally once I have access to AWS instances.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/test-images/germany-traffic-sign-1.jpg)
![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/test-images/germany-traffic-sign-2.jpg)
![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/test-images/germany-traffic-sign-3.jpg)
![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/test-images/germany-traffic-sign-4.jpg)
![alt text](https://github.com/arpitsri3/carND-Traffic-Sign-Classifier/blob/master/test-images/germany-traffic-sign-5.jpg)

The images are difficult to classify because :
1. They have noise in the form of watermarks.
2. They have slightly different posture than the ones in the original dataset.
3. They have some signs which are portrayed differently, such as , the 'towards right' sign which is towards right but pointing slightly up.
4. They have different colors but that may not matter as the images are converted to grayscale before being fed into the network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction						| 
|:---------------------:|:-------------------------------------:| 
| Caution        		| 20 km/h   							| 
| Right     			| Right 								|
| 30 km/h 				| Priority Road							|
| 50 km/h	      		| 20 km/h					 			|
| Pedestrians			| Caution      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares unfavorably to the accuracy on the test set of training dataset. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The reason for the above might be because of the fact that each of these images are watermarked and that is a scenario not encountered by the network in the training set. 
Only in the second image the network is certain with a probability of 0.84 that the sign is infact a 'Towards right' sign. Not unsurprisingly , the image has a very faint watermark , relatively to others. Its actually very interesting to try out this dataset with a new and different case of watermark. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


