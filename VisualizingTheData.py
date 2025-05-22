!pip install tensorflow
#%%
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train_cnn = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_valid_cnn = X_valid.reshape(-1, 28, 28, 1).astype('float32')
X_test_cnn = X_test.reshape(-1, 28, 28, 1).astype('float32')

y_train_cnn = y_train
y_valid_cnn = y_valid
y_test_cnn = y_test

print("Train images shape:       ",X_train.shape)
print("Train labels shape:       ",y_train.shape)
print("Validation images shape:  ",X_valid.shape)
print("Validation labels shape:  ",y_valid.shape)
print("Test images shape:        ",X_test.shape)
print("Test labels shape:        ",y_test.shape)
print("Train CNN images shape:       ",X_train_cnn.shape)
print("Train CNN labels shape:       ",y_train_cnn.shape)
#%%
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()
#%%
#convert from int to float to prepare for normalization
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

#normalizes data to have mean = 0 and std = 1

mean = np.mean(X_train)
std = np.std(X_train)

X_train = (X_train-mean)/(std + 1e-7)
X_valid = (X_valid-mean)/(std + 1e-7)
X_test = (X_test-mean)/(std + 1e-7)
print("Train images shape:       ",X_train.shape)
# print("Train labels shape:       ",y_train.shape)
#converts digit labels into vector
#needed for softmax and categorical crossentropy

mean = np.mean(X_train_cnn)
std = np.std(X_train_cnn)
X_train_cnn = (X_train_cnn - mean) / (std + 1e-7)
X_valid_cnn = (X_valid_cnn - mean) / (std + 1e-7)
X_test_cnn = (X_test_cnn - mean) / (std + 1e-7)
print("Train images shape:       ",X_train_cnn.shape)

y_train_cnn = to_categorical(y_train_cnn, 10)
y_valid_cnn = to_categorical(y_valid_cnn, 10)
y_test_cnn = to_categorical(y_test_cnn, 10)
print("Train labels shape:       ",y_train_cnn.shape)
print("Validation labels shape:  ",y_valid_cnn.shape)
print("Test labels shape:        ",y_test_cnn.shape)

y_train = to_categorical(y_train,10)
y_valid = to_categorical(y_valid, 10)
y_test = to_categorical(y_test, 10)
