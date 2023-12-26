from util import *
from sklearn.model_selection import train_test_split

##Step1
path="mydata"
data=importDataInfo(path)
##Step2
data=balancedata(data,dis=False)
##Step3
imgspath,steerings=loaddata(path,data)
# print(imgspath[0],steering[0])

##Step4
xtrain,xval,ytrain,yval=train_test_split(imgspath,steerings,test_size=0.2,random_state=5)
print(len(xtrain))
print(len(xval))

##Step5 data augmentation


##step6 data preprocessing


##step8
model=createmodel()
# model.summary()

##step9
history=model.fit(batchgen(xtrain,ytrain,100,1),steps_per_epoch=300,epochs=20,
          validation_data=batchgen(xval,yval,100,0),validation_steps=200)

model.save('model.h5')
print('MODEL SAVED')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('LOSS')
plt.xlabel('Epoch')
plt.show()