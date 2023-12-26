import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense
from keras.optimizers import Adam

def getname(filepath):
    return filepath.split('\\')[-1]


def importDataInfo(path):
    columns=['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data=pd.read_csv(os.path.join(path,'driving_log.csv'),names=columns)
    print(data.head())
    print(data['Center'][0])
    print(getname(data['Center'][0]))
    data['Center']=data['Center'].apply(getname)
    print(data.head())
    print('Total images Imported: ',data.shape[0])
    return data


def balancedata(data,dis=True):
    nbins=31
    samplesperbin=1000
    hist, bins=np.histogram(data['Steering'],nbins)
    # print(bins)
    if dis:
        center=(bins[:-1]+bins[1:])*0.5
        # print(center) 
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesperbin,samplesperbin))
        plt.show()


    removeindexlist=[]
    for j in range(nbins):
        bindatalist=[]
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                bindatalist.append(i)
        bindatalist=shuffle(bindatalist)
        bindatalist=bindatalist[samplesperbin:]
        removeindexlist.extend(bindatalist)
 
    print("Removed Images: ",len(removeindexlist))
    data.drop(data.index[removeindexlist],inplace=True)
    print('Remaing Images: ',len(data))

    if dis:
        hist, _ =np.histogram(data['Steering'],nbins)
        plt.bar(center,hist,width=0.06)
        plt.plot((-1,1),(samplesperbin,samplesperbin))
        plt.show()

    return data

def loaddata(path,data):
    imgspath=[]
    steering=[]

    for i in range(len(data)):
        indexeddata=data.iloc[i]
        # print(indexeddata)
        imgspath.append(os.path.join(path,'IMG',indexeddata[0]))
        steering.append(float(indexeddata[3]))        
    imgspath=np.asarray(imgspath)
    steering=np.asarray(steering)
    return imgspath,steering

def augmentimg(imgpath,steering):
    img=mpimg.imread(imgpath)
    #pan
    if np.random.rand() < 0.5:
        pan=iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    #zoom
    if np.random.rand() < 0.5:
        zoom=iaa.Affine(scale=(1,1.2))
        img=zoom.augment_image(img)
    #brightness
    if np.random.rand() < 0.5:
        brightness=iaa.Multiply((0.4,1.2))
        img=brightness.augment_image(img)
    #flip 
    if np.random.rand() < 0.5:
        img=cv2.flip(img,1)
        steering=-steering

    return img,steering

def preprocessing(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img

def batchgen(imgspath,steeringlist,batchsize,flag):
    while True:
        imgbatch=[]
        steeringbatch=[]

        for i in range(batchsize):
            index=random.randint(0,len(imgspath)-1)
            if flag:
                img,steering=augmentimg(imgspath[index],steeringlist[index])
            else:
                img=mpimg.imread(imgspath[index])
                steering=steeringlist[index]
            img=preprocessing(img)
            imgbatch.append(img)
            steeringbatch.append(steering)
        yield(np.asarray(imgbatch),np.asarray(steeringbatch))

def createmodel():
    model=Sequential()

    model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(48,(5,5),(2,2),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),input_shape=(66,200,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),input_shape=(66,200,3),activation='elu'))
    
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))
    model.compile(Adam(learning_rate=0.0001),loss='mean_squared_error')

    return model