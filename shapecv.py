# find . -name '.DS_Store' -type f -delete -->to delete DS_Store file in the folder 
import cv2
import numpy as np
import os
from time import sleep
def empty(a):
	pass
def preprocess(img):
	img=cv2.GaussianBlur(img,(5,5),6) ##blur
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ##gray
	img=cv2.resize(img,(32,32))
	img=cv2.equalizeHist(img)
	img=img/255 #the value fall in range of[0,1]
	return img
count=0
images=[]
classes=[]
x=os.listdir("objects")
for object in x[1:]:
	images_list=os.listdir("objects/"+str(object))
	for image in images_list[1:]:
		read_img=cv2.imread("objects/"+str(object)+"/"+str(image))
		print(image)
		img=preprocess(read_img)
		images.append(img)
		classes.append(count)
	print(count,end=" ")
	count+=1
	
images=np.array(images,dtype=int)
classes=np.array(classes)
from keras.utils import to_categorical
classes = to_categorical(classes)
print("classes",classes)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(images,classes)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Flatten,Conv2D, MaxPooling2D
# from tensorflow.keras.layers.convolutional import Conv2D, MaxPooling2D

def myModel():
    no_Of_Filters=60
    size_of_Filter=(5,5) # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
                         # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter2=(3,3)
    size_of_pool=(2,2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    no_Of_Nodes = 500   # NO. OF NODES IN HIDDEN LAYERS
    model= Sequential()
    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(images[0].shape[0],images[0].shape[1],1),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
 
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.6))
 
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.6)) # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(3,activation='softmax')) # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=myModel()
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=500,batch_size=256)
ypred=model.predict(xtest)
# classIndex = model.predict_classes(img)
print(ypred,ytest)
print(ypred[0].argmax(axis=-1))
print(ypred[1].argmax(axis=-1))
print(ypred[2].argmax(axis=-1))
print(ypred[3].argmax(axis=-1))
print(ypred[4].argmax(axis=-1))
print(ypred[5].argmax(axis=-1))
ypred_cond=[]
from sklearn.metrics import accuracy_score,confusion_matrix
res={0:"car",1:"bike",2:"airoplane"}
for i in ypred:
	ypred_cond.append(i.argmax(axis=-1))
zeroes=np.zeros((len(ypred_cond),3))
for i,j in enumerate(zeroes):
	print(i)
	j[ypred_cond[i]]=1
print(zeroes)
print(accuracy_score(ytest,zeroes))


vid=cv2.VideoCapture(0)
cv2.namedWindow("parameters")
cv2.resizeWindow("parameters",1200,2400)
cv2.createTrackbar("Threshold1","parameters",23,255,empty)
cv2.createTrackbar("Threshold2","parameters",24,255,empty)
cv2.createTrackbar("area","parameters",500,5000,empty)
# imgb=cv2.GaussianBlur(vid,(5,5),6)
def getcontours(img,imgcountour):
	countours,hirarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	for cnt in countours:
		area_thr=cv2.getTrackbarPos("area","parameters")
		area=cv2.contourArea(cnt)
		if(area>area_thr):
			cv2.drawContours(imgcountour,cnt,-1,(255,0,255),7)


# cv2.waitKey(0)
while 1:
	sleep(1)
	suc,cap=vid.read()
	cv2.imshow("original Image", cap)
	img = preprocess(cap)
	cv2.imshow("Processed Image", img)
	img=np.array(img,dtype=int)
	img = img.reshape(1, 32, 32, 1)
	predictions = model.predict(img)
	print(predictions)
	print(res[int(predictions.argmax(axis=-1))])
	# print(res[predictions.argmax(axis=-1)])
	# imgcontour=cap.copy()
	# imgb=cv2.GaussianBlur(cap,(5,5),6)
	# imggray=cv2.cvtColor(imgb,cv2.COLOR_BGR2GRAY)
	# Threshold1=cv2.getTrackbarPos("Threshold1","parameters")
	# Threshold2=cv2.getTrackbarPos("Threshold2","parameters")
	# imgcany=cv2.Canny(imgb,Threshold1,Threshold2)
	# getcontours(imgcany,imgcontour)
	# # cv2.imshow('video',vid)
	# # cv2.imshow('video blur',imgb)
	# cv2.imshow('video edge',imgcany)
	# # cv2.imshow('original',cap)
	# # cv2.imshow("gray image",imggray)
	# cv2.imshow("object",imgcontour)
	if(cv2.waitKey(1) & 0xFF==ord('q')):
		break

