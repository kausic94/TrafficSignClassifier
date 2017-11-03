<h1><center>Traffic Sign Classifier</h1></center>

I firstly converted the 3 channel color image into a single channel image. I converted it into a luminance image. TO do that I multiplied by the coefficients suggested by the ITU-R recommendation. Luminance image is supposedly the way our human visual system perceives it. (It works well in our brain, So I'm guessing it should work well in this one too. xD). But, more importantly I did this so that there is less or no weightage on the type of color and the model is derived to look for the correct shapes and patterns and not color. I also applied a normalization step on the training and test data so that we have a uniform data representation.

<center><img src =normalized.png></center>
<center><i> This is an example of the image normalized </i></center>

I used the thumb rule of 80-20 to split my data into the validation and training set. I did generate additional data. There were a lot of images for some classes and while others had very few. Fearing a biased prediction due to this I generated some additional data. This additional data is the deformed version of the same training images. This not only helps to create a uniform distribution of the data but it also helps to increase the robustness of the prediction.

I created various functions such as rotation, translation , shearing, flipping, change in brightness etc and randomly subjected the images in each class to a random number of these operations and appended them with the original data with the right labels. I've made sure I have split the validation and training data from the data only after shuffling so that most of my new images are not pushed into validation set.

<h2> Architecture </h2>

The architecture I have used here is based on the Lenet architecture that was taught in the class. However, I did add my own modifications to it. I took the output of layer1, flattend it and added a fully connected layer to it so that I get a vector that is equal in length to the flattend layer of the second convolutional layer. From there on I doubled all the exsisting lenet parameters to finally get a resonably working model.
Here is a summary of architecture:
Input layer (32x32x1)
convolutional Layer (32x32x1 with filter 5x5 and stride 1 ouput 28x28x6)
ReLu activation
Maxpooling Layer(28x28x6 with filter 2x2 and stride 2 output 14x14x6)
flatten layer (14x14x6 to 1176)
Convolutional Layer (14x14x6 with filter 5x5 and stride 1 output 10x10x16)
ReLu activation
Maxpooling layer (10x10x16 with filter 2x2 and stride 2 output 5x5x16)
flatten layer (5x5x16 to 400 nodes)
fully connected layer of flattend layer 1 (1176 to 400 with drop out and relu activation)
cocatenation of two flattend layers from (1x400 and 1x400 to 1x800)
fully connected layer (800 to 240 with drop out and Relu activation)
fully connected layer (240 to 168 with drop out and Relu activation)
fully connected layer (168 to 84 with drop out and Relu activation)
fully connected layer (84 to 43)

<h2> Summary </h2>
I implemented the network based on the LeNet architecture. I experimented with various learning rates, batch size and epoch values. No matter how much I increased the epoch value, the accuracy plateaus at 93% validation accuracy. In addition to the original LeNet architecture I also added a layer of dropouts to the fully connected layers in the network. In addition to the existing 3 fully connected layers there are 2 more layers with the drop outs.This architecture also takes into account the initial features that is output from the first convolutional layer unlike the LeNet architecture which classifies based only on the second convolution layer. Lenet had shown good promise in the Mnist database for character recognition and hen. It also seemed like an easy to implement architecture with only few hyper parameters to tune.Hence I chose to work on architecture based on this model, for the problem.