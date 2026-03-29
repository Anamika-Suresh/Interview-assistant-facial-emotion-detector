import tensorflow as tf
import pickle
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPooling2D,Dense,Dropout,SpatialDropout2D,Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv('fer2013.csv')
emotion = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
training_data = pd.get_dummies(data,columns = ['emotion'] )

training = training_data[training_data['Usage'] == 'Training']
test = training_data[training_data['Usage'] == 'PublicTest']
t = training['pixels']
X_train = []
for x in t:
  temp = x.split(' ')
  temp = [int(_) for _ in temp]
  temp = np.array(temp)
  temp = temp.reshape(48,48)
  temp  = temp/255 
  X_train.append(temp)
X_test = []
t = test['pixels']
for x in t:
  temp = x.split(' ')
  temp = [int(_) for _ in temp]
  temp = np.array(temp)
  temp = temp.reshape(48,48)
  temp  = temp/255 
  X_test.append(temp)


y_test = training_data[training_data['Usage'] == 'PublicTest'].drop(['pixels','Usage'],axis=1)
y_train = training_data[training_data['Usage'] == 'Training'].drop(['pixels','Usage'],axis=1)
X_train = np.array(list(np.array(X_train))).reshape(-1,48,48,1)
X_test = np.array(list(np.array(X_test))).reshape(-1,48,48,1)

model = tf.keras.models.Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', padding="same", input_shape=(48,48,1),kernel_regularizer=l1(1e-6)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu', padding='same',kernel_regularizer=l1(1e-6)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same',kernel_regularizer=l1(1e-6)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same',kernel_regularizer=l1(1e-6)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SpatialDropout2D(0.5))
model.add(Flatten())

model.add(Dense(256,activation = 'relu',kernel_regularizer=l1(1e-6)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(512,activation = 'relu',kernel_regularizer=l1(1e-6)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(7,activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
early_stop = EarlyStopping(
    monitor='val_loss',      
    patience=5,              
    restore_best_weights=True,  
    min_delta=0.001         
)
history = model.fit(X_train,y_train,epochs=25,callbacks=[early_stop], batch_size=128,validation_split=0.1,shuffle=True)

model.save("model.h5")
print("Model saved as model.h5")
