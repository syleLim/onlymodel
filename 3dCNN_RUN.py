import cv2
import numpy
import tensorflow as tf
import os
import copy
import numpy as np

dir_path = './3d_learn_img_64/eval/no/78'

width = 64
height = 64
depth = 30
nLabel = 2

def load_image (path) :

    tmp_img_3d = list()

    for img_name in os.listdir(path) :
        img = cv2.imread(path +'/'+ img_name, cv2.IMREAD_GRAYSCALE)
        tmp_img_3d.append(img.tolist())

    #print(tmp_img_3d)

    tmp_1 = list()
    tmp_2 = list()
    tmp_3 = list()

    for first in range(0, width) : 
        for second in range(0, height) : 
            for third in range(0, depth) : 
                a = copy.deepcopy(tmp_img_3d[third][first][second])
                tmp_1.append(a)

            b = copy.deepcopy(tmp_1)
            tmp_2.append(b)
            tmp_1.clear()
            
        c = copy.deepcopy(tmp_2)
        tmp_3.append(c)
        tmp_2.clear()

    data = np.array(tmp_3, np.float32)

    #resized_image = tf.image.resize_images(images=data, size=(width, height), method=1)

    return data

img = load_image(dir_path)

# set first data
x = tf.placeholder(tf.float32, shape=[None, width*height*depth]) 
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  

# weight initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#bias initalization
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and Pooling
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') 
  # set stride = [1, 1, 1, 1, 1]

# max pooling
def max_pool_2x2(x):  
  return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')
  # set stride = [1, 2, 2, 2, 1]


## 1st Convolution
weight_conv1 = weight_variable([5, 5, 5, 1, 32])
bias_conv1 = bias_variable([32])  

x_image = tf.reshape(x, [-1,width,height,depth,1]) # ,1] .. grayscale / ,3].. rgb
print("input size : ")
print(x_image.get_shape) # (?, 256, 256, 30, 1)

#conv-relu-maxpool
h_conv1 = tf.nn.relu(conv3d(x_image, weight_conv1) + bias_conv1)
h_pool1 = max_pool_2x2(h_conv1)  
print("1st Conv-relu-maxpool end, size :")
print(h_conv1.get_shape) # (?, 256, 256, 30, 32)
print(h_pool1.get_shape) # (?, 128, 128, 30, 32)



## 2nd Convolution
weight_conv2 = weight_variable([5, 5, 5, 32, 64]) 
bias_conv2 = bias_variable([64]) 

#conv-relu-maxpool
h_conv2 = tf.nn.relu(conv3d(h_pool1, weight_conv2) + bias_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool 
print("2nd Conv-relu-maxpool end, size :")
print(h_conv2.get_shape) # (?, 128, 128, 20, 64) 
print(h_pool2.get_shape) # (?, 64, 64, 10, 64)   



# fully-connected layer
weight_fc1 = weight_variable([4*4*2*64, 1024])  # all data set(count) -> 1024
bias_fc1 = bias_variable([1024]) 

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*2*64])  # -> all data set(count)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight_fc1) + bias_fc1)  
print("fully-connected end, size : ")
print(h_pool2_flat.get_shape)  # (?, count)
print(h_fc1.get_shape) # (?, 1024) 



# drop_out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print("drop_out end, siz : ")
print(h_fc1_drop.get_shape)  # -> output: 1024

# Readout Layer
weight_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
bias_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, weight_fc2) + bias_fc2
print('last out_put : ')
print(y_conv.get_shape)  # -> 1024

####
#p = tf.nn.softmax()
####


init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

"""
img_ = img.reshape(1, width*height*depth)
p = tf.nn.softmax(y_conv)

sess = tf.InteractiveSession()
sess.run(init_op)
#saver = tf.train.Saver()
saver.restore(sess, "./cnn_3d_model.ckpt")

print(type(img))


p = sess.run(p, feed_dict = {x: img_, keep_prob : 1.0})

print(p)
"""


"""
img_ = img.reshape(1, width*height*depth)

with tf.Session() as sess:
    sess.run(init_op)

    resized_image = tf.image.resize_images(images=img, size=(width, height), method=1)
    image = sess.run(resized_image)  # (256,256,30)
    #x_data = np.append(x_data, np.asarray(img, dtype='float32')) # (image.data, dtype='float32')

    img_ = image.reshape(1, width*height*depth)

    saver.restore(sess, "./cnn_3d_model.ckpt")

    prediction = sess.run(y_conv, feed_dict = {x: img_, keep_prob : 1.0}) ##??? wrong count
    print(prediction)

"""


with tf.Session() as sess:
    sess.run(init_op)
    resized_image = tf.image.resize_images(images=img, size=(width, height), method=1)

    p = tf.nn.softmax(y_conv)
    
    x_d = sess.run(resized_image)
    saver.restore(sess, "./cnn_3d_model.ckpt")

    x_d = x_d.reshape(1, width*height*depth)

    prediction = tf.argmax(p, 1)
    print(prediction)

    print(prediction.eval(feed_dict={x: x_d, keep_prob: 1.0}, session= sess))



    



