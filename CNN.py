# important imports
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
from keras.datasets import cifar10


  #loading the Cifar10 data into train and test 
  
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


#The data is in integers therefore we need to convert them to float first  
#and normalise the data by diving by 255 as it's in RGB

X_train, X_test = X_train.astype('float32')/255.0, X_test.astype('float32')/255.0

# Change the labels from categorical to one-hot encoding

y_train, y_test = u.to_categorical(y_train, 10), u.to_categorical(y_test, 10)


#Building the CNN Model

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=(momentum=0.5, decay=0.0004,metrics=['accuracy']))

model.summary()

#Run and training the algorithm!
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=512)
#Save the weights to use for later
model.save_weights("cifar10.hdf5")
#Finally print the accuracy of our model!
test_loss,test_acc = model.evaluate(X_test, y_test,verbose=2)


# prediction
#Prints out a number
#  1 - airplane, 2 - automobile, 3 - bird, 4 - cat, 5 - deer, 6 - dog, 7 - frog, 8 - horse, 9 - ship, 10 - truck

y_pred=model.predict(X_test)
