
# coding: utf-8

# In[13]:


# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import glob
from keras.preprocessing import image
import h5py

train_path = os.getcwd()+'/dataset/train'
test_path = os.getcwd()+'/dataset/test'

train_labels = os.listdir(train_path)
test_labels = os.listdir(test_path)

img_size = (64,64)
no_of_train_img = 1500 #750+750
no_of_test_img = 100 #50+50
no_of_channel = 3

train_x = np.zeros(((img_size[0]*img_size[1]*no_of_channel),no_of_train_img))
train_y = np.zeros((1,no_of_train_img))


test_x = np.zeros(((img_size[0]*img_size[1]*no_of_channel),no_of_test_img))
test_y = np.zeros((1,no_of_test_img))


num_label =0
count =0

for i,label in enumerate(train_labels):
    cur_path = train_path+ "/" +label
    for img_path in glob.glob(cur_path+"/*.jpg"):
        img = image.load_img(img_path,target_size=img_size)
        x = image.img_to_array(img)
        x= x.flatten()
        #x = np.expand_dims(x,axis=0)
        train_x[:,count]= x
        train_y[:,count] = num_label
        count+=1
    num_label+=1
 
count =0
num_label =0
for i,label in enumerate(test_labels):
    cur_path = test_path+ "/" +label
    for img_path in glob.glob(cur_path+"/*.jpg"):
        img = image.load_img(img_path,target_size=img_size)
        x = image.img_to_array(img)
        x= x.flatten()
        #x = np.expand_dims(x,axis=0)
        test_x[:,count]= x
        test_y[:,count] = num_label
        count+=1
    num_label+=1
 
    
 #standardization
train_x = train_x/255
test_x = test_x/255

print ("train_labels : " + str(train_labels))
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape : " + str(test_x.shape))
print ("test_y shape : " + str(test_y.shape))

#saving dataset
h5_train = h5py.File("train_x.h5",'w')
h5_train.create_dataset("data_train", data=np.array(train_x))
h5_train.close()

h5_test = h5py.File("test_x.h5",'w')
h5_test.create_dataset("data_test", data=np.array(test_x))
h5_test.close()

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def init_params(dimension):
    w = np.zeros((dimension, 1))
    b = 0
    return w, b

def propogate(w,b,X,Y):
    m= X.shape[1]
    
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1/m)*(np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))))
    dw = (1/m)*(np.dot(X, (A-Y).T))
    db = (1/m)*(np.sum(A-Y))
    #cost = np.squeeze(cost)
    
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w,b,X,Y,epochs,lr):
    costs =[]
    
    for i in range(epochs):
        grads,cost = propogate(w,b,X,Y)
        
        dw = grads['dw']
        db = grads['db']
        
        w = w - (lr*dw)
        b = b - (lr*db)
        
        if i%10== 0:
            print (".",end="", flush=True)
        
    params = {"w": w, "b": b}

    grads  = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_predict = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_predict[0, i] = 0
        else:
            Y_predict[0,i]  = 1

    return Y_predict


def model(X_train, Y_train, X_test, Y_test, epochs, lr):
    w, b = init_params(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr)

    w = params["w"]
    b = params["b"]

    Y_predict_train = predict(w, b, X_train)
    Y_predict_test  = predict(w, b, X_test)
    
    print ("train_accuracy: {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print ("test_accuracy : {} %".format(100-np.mean(np.abs(Y_predict_test  - Y_test)) * 100))

    log_reg_model = {"costs": costs,
                "Y_predict_test": Y_predict_test, "Y_predict_train" : Y_predict_train, "w" : w, "b" : b,"learning_rate" : lr,"epochs": epochs}

    return log_reg_model

epochs = 2000
lr = 0.1


myModel = model(train_x, train_y, test_x, test_y, epochs, lr)
print(myModel['Y_predict_test'])
print(myModel['w'])
print(myModel['b'])



# In[14]:


import urllib.request


# In[15]:


def download_web_image(url):
    name = "image"
    full_name = name+".jpg"
    urllib.request.urlretrieve(url,full_name)
    
    


# In[16]:


download_web_image("https://akm-img-a-in.tosshub.com/indiatoday/images/story/201806/big_dog_0.jpeg?Hj9xcINeQkp9emv9UULlHzWGNSZeIWov")


# In[17]:


from IPython.display import Image
Image(filename='image.jpg') 


# In[18]:


read_image = cv2.imread('image.jpg')


# In[19]:



resized_image = cv2.resize(read_image, (64, 64)) 


# In[20]:


resized_image


# In[21]:


x = image.img_to_array(resized_image)


# In[22]:


x=x.flatten()
x


# In[23]:


temp = sigmoid(np.dot(myModel['w'].T, x) + myModel['b'])
if(temp<=0.5):
    print("airplane")
else:
    print("Bike")

