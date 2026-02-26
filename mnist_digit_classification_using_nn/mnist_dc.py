#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix   # FIX 3

#loading the data from keras datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
type(X_train)

# shape of the numpy arrays
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# printing the 10th image
print(X_train[10])
print(X_train[10].shape)

# displaying the image
plt.imshow(X_train[25])
plt.show()

# print the corresponding label
print(Y_train[25])

#image lables
print(Y_train.shape, Y_test.shape)

# unique values in Y_train
print(np.unique(Y_train))

# unique values in Y_test
print(np.unique(Y_test))

# scaling the values
X_train = X_train/255
X_test = X_test/255

# printing the 10th image
print(X_train[10])

#buildinng neural metwork model
model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')   # FIX 2
])

# compiling the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the Neural Network
model.fit(X_train, Y_train, epochs=10)

#accuracy on test data
loss, accuracy = model.evaluate(X_test, Y_test)   # FIX 1
print(accuracy)
print(X_test.shape)

# first data point in X_test
plt.imshow(X_test[0])
plt.show()
print(Y_test[0])

Y_pred = model.predict(X_test)
print(Y_pred.shape)
print(Y_pred[0])

# converting the prediction probabilities to class label
label_for_first_test_image = np.argmax(Y_pred[0])
print(label_for_first_test_image)

# converting the prediction probabilities to class label for all test data points
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

conf_mat = confusion_matrix(Y_test, Y_pred_labels)
print(conf_mat)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

input_image_path = "D:\\AIML\\MNIST Digit classifircation\\MNIST Digit classifircation\\3-digit.PNG"
input_image = cv2.imread(input_image_path)

type(input_image)
print(input_image)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.show()

input_image.shape
grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
grayscale.shape
input_image_resize = cv2.resize(grayscale, (28, 28))
input_image_resize.shape
plt.imshow(input_image_resize, cmap='gray')
plt.show()
input_image_resize = input_image_resize/255
type(input_image_resize)
image_reshaped = np.reshape(input_image_resize, [1,28,28])
input_prediction = model.predict(image_reshaped)
print(input_prediction)
input_pred_label = np.argmax(input_prediction)
print(input_pred_label)

#predictive system for handwritten digit recognition
input_image_path = input('Path of the image to be predicted: ')
input_image = cv2.imread(input_image_path)

plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.show()

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
input_image_resize = cv2.resize(grayscale, (28, 28))
input_image_resize = input_image_resize/255
image_reshaped = np.reshape(input_image_resize, [1,28,28])
input_prediction = model.predict(image_reshaped)
input_pred_label = np.argmax(input_prediction)

print('The Handwritten Digit is recognised as ', input_pred_label)

model.save("mnist_model.h5")