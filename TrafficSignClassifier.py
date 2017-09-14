
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import math
def rotation(img):
    h,w = img.shape
    theta=np.linspace(0,360,361)     # rotation
    angle=np.random.randint(0,len(theta))
    angle=theta[angle]
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    return cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    
def translate(img):
    translation=np.linspace(-5,5,11)            # translation
    tx,ty=np.random.randint(0,len(translation)),np.random.randint(0,len(translation))
    tx,ty=translation[tx],translation[ty]
    M = np.float32([[1,0,tx],[0,1,ty],[0,0,1]])
    return cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    
def projection(img):
    c=np.linspace(-.0003,.0003,15)               #projection
    c1,c2=np.random.randint(0,len(c)),np.random.randint(0,len(c))
    c1,c2=c[c1],c[c2]
    M = np.float32([[1,0,0],[0,1,0],[c1,c2,1]])
    return cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    
def shearing(img):
    shear=np.linspace(-np.radians(15),np.radians(15),31)  #shear
    s=np.random.randint(0,len(shear))
    s=shear[s]
    M = np.float32([[1,s,0],[0,1,0],[0,0,1]])
    return cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))

def scaling(img):
    h,w=img.shape
    scales=np.linspace(0.6,1.6,20)
    i=np.random.randint(0,len(scales))
    s=scales[i]
    M = cv2.getRotationMatrix2D((w/2,h/2),0,s)
    return cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))

def flipping(img):
    index=np.random.randint(0,2)
    flips=[np.fliplr(img),np.flipud(img)]
    return flips[index]

def brightness_change(img):
    br=np.random.randint(-75,75)
    img=cv2.add(img,br)
    return img

def deform_img(img):
    op=np.random.randint(1,7)
    operations={1:rotation(img),2:translate(img),3:scaling(img),4:flipping(img),5:shearing(img),6:brightness_change(img)} 
    img=operations[op]
    return img;

def LeNet(x,dropout_probability):
    sigma = 0.1
    mu=0
    
    #Convolutional. Input = 32x32x1. Output = 28x28x6.
    w1=tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    b1=tf.Variable(tf.zeros([6]))
    strides=[1,1,1,1]
    layer1=tf.nn.conv2d(x,w1,strides,'VALID')
    layer1=tf.nn.bias_add(layer1,b1)
    layer1=tf.nn.relu(layer1)
    
    #Pooling. Input = 28x28x6. Output = 14x14x6.
    ksize=[1,2,2,1]
    strides=[1,2,2,1]
    layer1=tf.nn.max_pool(layer1,ksize,strides,padding='VALID')
    #Flattening Layer1 to 1176 vector
    flattend_layer1=flatten(layer1) 
    # Convolutional. Output = 10x10x16.
    w2=tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    b2=tf.Variable(tf.zeros([16]))
    strides=[1,1,1,1]
    layer2=tf.nn.conv2d(layer1,w2,strides,padding='VALID')
    layer2=tf.nn.bias_add(layer2,b2)
    layer2=tf.nn.relu(layer2)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2=tf.nn.max_pool(layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flattend=flatten(layer2)
    #Converting 1176 vector to 400 length vector.
    fc=tf.Variable(tf.truncated_normal(shape=(1176,400),mean=mu,stddev=sigma))
    bc=tf.Variable(tf.zeros([400]))
    flattend_layer1=tf.add(tf.matmul(flattend_layer1,fc),bc)
    flattend_layer1=tf.nn.relu(flattend_layer1)
    flattend_layer1=tf.nn.dropout(flattend_layer1,dropout_probability)
    
    # Fully Connected. Input = 800. Output = 240.
    final_flattend=tf.concat(1,[flattend_layer1,flattend])
    print(final_flattend.get_shape())
    w3=tf.Variable(tf.truncated_normal(shape=(800, 240), mean = mu, stddev = sigma)) # 400,120
    b3=tf.Variable(tf.zeros([240]))#240
    logits=tf.add(tf.matmul(final_flattend,w3),b3)
    logits=tf.nn.relu(logits)
    logits=tf.nn.dropout(logits,dropout_probability)
    
    # Fully Connected. Input = 240. Output = 168.
    w4=tf.Variable(tf.truncated_normal(shape=(240, 168), mean = mu, stddev = sigma)) #120,84
    b4=tf.Variable(tf.zeros([168])) 
    logits=tf.add(tf.matmul(logits,w4),b4)
    logits=tf.nn.relu(logits)
    logits=tf.nn.dropout(logits,dropout_probability)
    #fully Connected.
    fc1=tf.Variable(tf.truncated_normal(shape=(168,84),mean=mu,stddev=sigma))
    bc1=tf.Variable(tf.zeros([84]))
    logits=tf.add(tf.matmul(logits,fc1),bc1)
    logits=tf.nn.relu(logits)
    logits=tf.nn.dropout(logits,dropout_probability)  
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    w5=tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    b5=tf.zeros([n_classes])
    logits=tf.add(tf.matmul(logits,w5),b5)
    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,dropout_probability:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



training_file = "train.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_test = len(X_test)
image_shape = (32,32)
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

plt.figure()
plt.plot(y_train)
plt.figure()
hist,bin_edges=np.histogram(y_train,range(n_classes+1)) # This plots a histogram of all the data 
plt.bar(bin_edges[:-1],hist)
plt.figure()
img=X_train[np.random.randint(0,n_train),:,:,:] #Visualizing random image
plt.imshow(img)
plt.show()

X_train=(X_train-X_train.mean())/(np.max(X_train)-np.min(X_train))
X_test=(X_test-X_test.mean())/(np.max(X_test)-np.min(X_test))
X_test=(X_test-X_test.mean())/(np.max(X_test)-np.min(X_test))
X_train_processed=(X_train[:,:,:,0]*0.0722 + X_train[:,:,:,1]*0.7152 + X_train[:,:,:,2]*0.2126)
X_test_processed=(X_test[:,:,:,0]*0.0722 + X_test[:,:,:,1]*0.7152 + X_test[:,:,:,2]*0.2126)
plt.imshow(X_train_processed[768],cmap='gray')
plt.figure()
plt.imshow(X_train[768],cmap='gray')

#Generating Additional data

new_X_train=list()
new_y_train=list()
validation_data=list()
validation_label=list()
y_train=y_train.tolist()
tempX,tempY=list(),list()
X_train_processed=list(X_train_processed)
import random
max_hist_value=max(hist)
prev=0
for i in range(n_classes):
    no_of_images=hist[i]     # original images
    no_of_new_images=max_hist_value-hist[i]  # finding the number of new images to be added
    tempX=X_train_processed[prev:prev+no_of_images] # getting the images and labels
    tempY=y_train[prev:prev+no_of_images]
    prev+=no_of_images    #Tracking them
   
    for i in range(no_of_new_images): 
        tempX.append(deform_img(tempX[np.random.randint(0,no_of_images)]))  #adding deformed images
        tempY.append(tempY[-1])  # adding corresponding lables
    random.shuffle(tempX)   #shuffling the list of
    new_X_train+=tempX[:int(len(tempX)*0.8)]  # separating into  training data and validation data
    new_y_train+=tempY[:int(len(tempY)*0.8)]
    validation_data+=tempX[int(len(tempX)*0.8):]
    validation_label+=tempY[int(len(tempY)*.8):]
    del tempX[:]  # deleting the buffer
    del tempY[:]
print(len(new_X_train),",",len(validation_data))
print(len(new_y_train),",",len(validation_label))
print (new_X_train[0].shape)

#Visualizing the added data
hist0,bin_edges0=np.histogram(new_y_train,range(n_classes+1)) # This plots a histogram of all the training data 
plt.bar(bin_edges0[:-1],hist0)

plt.figure()
hist1,bin_edges1=np.histogram(validation_label,range(n_classes+1)) # this plots the histogram of all the validation data
plt.bar(bin_edges1[:-1],hist1)

plt.show()

#shuffling the entire dataset as well as the labels and making it into an array of the required shape
X_test_processed=(X_test_processed).reshape(-1,32,32,1)
new_X_train=np.array(new_X_train).reshape(-1,32,32,1)
validation_data=np.array(validation_data).reshape(-1,32,32,1)
validation_label=np.array(validation_label)
new_y_train=np.array(new_y_train)

print(validation_data.shape,validation_label.shape,new_X_train.shape,new_y_train.shape)


assert(len(new_X_train)==len(new_y_train))
assert(len(validation_data)==len(validation_label))
print("Image Shape: {}".format(new_X_train[0].shape))
print("Training Set:   {} samples".format(len(new_X_train)))
print("Validation Set: {} samples".format(len(validation_data)))
EPOCHS = 275
BATCH_SIZE = 1024


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
dropout_probability=tf.placeholder(tf.float32)
rate = 0.001
logits = LeNet(x,dropout_probability)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    
    for i in range(EPOCHS):
        new_X_train, new_y_train = shuffle(new_X_train, new_y_train)
        loss_value=[]
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = new_X_train[offset:end], new_y_train[offset:end]
            _,l=sess.run([training_operation,loss_operation], feed_dict={x: batch_x, y: batch_y ,dropout_probability:0.75})
            loss_value.append(l)
            
        print ("Loss :",np.mean(loss_value))
        validation_accuracy = evaluate(validation_data, validation_label)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './Traffic_Sign_Classification_with_data_augmentation')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    
    test_accuracy = evaluate(X_test_processed, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))