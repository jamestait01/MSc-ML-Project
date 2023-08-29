import numpy as np
from PIL import Image
import os
import keras 
TUMOR_path = 'Kather_texture_2016_image_tiles_5000/01_TUMOR/'
STROMA_path = 'Kather_texture_2016_image_tiles_5000/02_STROMA/'
COMPLEX_path = 'Kather_texture_2016_image_tiles_5000/03_COMPLEX/'
LYMPHO_path = 'Kather_texture_2016_image_tiles_5000/04_LYMPHO/'
DEBRIS_path = 'Kather_texture_2016_image_tiles_5000/05_DEBRIS/'
MUCOSA_path= 'Kather_texture_2016_image_tiles_5000/06_MUCOSA/'
ADIPOSE_path = 'Kather_texture_2016_image_tiles_5000/07_ADIPOSE/'
EMPTY_path = 'Kather_texture_2016_image_tiles_5000/08_EMPTY/'

#Load the images from each folder path
TUMOR = np.array([np.array(Image.open(TUMOR_path + fname)) for fname in os.listdir(TUMOR_path)])
STROMA = np.array([np.array(Image.open(STROMA_path + fname)) for fname in os.listdir(STROMA_path)])
COMPLEX = np.array([np.array(Image.open(COMPLEX_path + fname)) for fname in os.listdir(COMPLEX_path)])
LYMPHO = np.array([np.array(Image.open(LYMPHO_path + fname)) for fname in os.listdir(LYMPHO_path)])
DEBRIS = np.array([np.array(Image.open(DEBRIS_path + fname)) for fname in os.listdir(DEBRIS_path)])
MUCOSA = np.array([np.array(Image.open(MUCOSA_path + fname)) for fname in os.listdir(MUCOSA_path)])
ADIPOSE = np.array([np.array(Image.open(ADIPOSE_path + fname)) for fname in os.listdir(ADIPOSE_path)])
EMPTY = np.array([np.array(Image.open(EMPTY_path + fname)) for fname in os.listdir(EMPTY_path)])

#Create the labels for each image so we know which image belongs to which class
TUMOR_labels = np.ones((len(TUMOR),1))
STROMA_labels = np.ones((len(STROMA),1))*2
COMPLEX_labels = np.ones((len(COMPLEX),1))*3
LYMPHO_labels = np.ones((len(LYMPHO),1))*4
DEBRIS_labels = np.ones((len(DEBRIS),1))*5
MUCOSA_labels = np.ones((len(MUCOSA),1))*6
ADIPOSE_labels = np.ones((len(ADIPOSE),1))*7
EMPTY_labels = np.ones((len(EMPTY),1))*8

#Combine all the images and labels into one X and y array
X = np.concatenate((TUMOR,STROMA,COMPLEX,LYMPHO,DEBRIS,MUCOSA,ADIPOSE,EMPTY),axis=0)
y = np.concatenate((TUMOR_labels,STROMA_labels,COMPLEX_labels,LYMPHO_labels,DEBRIS_labels,MUCOSA_labels,ADIPOSE_labels,EMPTY_labels),axis=0)

#Shuffle the data (prevents non random assignment to training and testing)
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)

#Split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Normalize the data to be between 0 and 1 for same scale
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#Convert the labels to categorical
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Create the model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(50,50,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#Evaluate the model
model.evaluate(X_test, y_test)

#Save the model
model.save('model.h5')

#Load the model
from keras.models import load_model
model = load_model('model.h5')

#Predict the model
y_pred = model.predict(X_test)

#Print the predicted and actual labels
print(y_pred)
print(y_test)

#Print the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))




