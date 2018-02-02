import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import copy


width = 64
height = 64
depth = 30  # 3                
batch_index = 0
filenames = []

# user selection
data_dir = './3d_learn_img_64'            
num_class = 2                      

## read the file name
def get_filenames(data_set):
    global filenames
    labels = []

    with open(data_dir + '/labels.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list

    for i, label in enumerate(labels):
        list = os.listdir(data_dir  + '/' + data_set + '/' + label)
        for filename in list:
            filenames.append([label + '/' + filename, i])

    random.shuffle(filenames)

## img load / img stack / sess.run -> return data for train
def get_data_jpeg(sess, data_set, batch_size):
    global batch_index, filenames

    if len(filenames) == 0 or data_set == 'eval': 
        filenames.clear()
        get_filenames(data_set) 
    
    max = len(filenames)

    #print(filenames)

    # train only batch size
    begin = batch_index
    end = batch_index + batch_size

    if end >= max:
        end = max
        batch_index = 0

    print('begin : '+str(begin))
    print('end : '+ str(end))

    x_data = np.array([], np.float32)
    y_data = np.zeros((batch_size, num_class)) 
    index = 0

    for i in range(begin, end):
        
        imagePath = data_dir + '/' + data_set + '/' + filenames[i][0]
        
        tmp_img_3d = []

        for img_name in  os.listdir(imagePath):
            img = cv2.imread(imagePath +'/'+img_name, cv2.IMREAD_GRAYSCALE)
            tmp_img_3d.append(img.tolist())

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
        
        print("img_dimention : " + str(data.shape))
        # 256 * 256 * 30 //
        resized_image = tf.image.resize_images(images=data, size=(width, height), method=1)
        
        image = sess.run(resized_image)  
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) 
        y_data[index][filenames[i][1]] = 1  
        index += 1

    batch_index = 0  
    x_data_ = x_data.reshape(batch_size, height * width * depth)

    random.shuffle(filenames)

    print("img_conducting_end")

    return x_data_, y_data, sess