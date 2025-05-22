from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
import time

X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

model = Sequential([
    Dense(512, activation='relu', input_shape=X_train.shape[1:]),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

batch_size = 64
epochs = 12
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1= 0.9 ,beta_2= 0.999, epsilon=1e-07)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Test and measure time
test_times = []
train_times = []
val_accuracies = []
for e in range(epochs):
      print(f"\nEpoch {e+1}/{epochs}")

      train_start = time.time()
      history = model.fit(
             X_train, y_train,
             validation_data=(X_valid, y_valid),
             epochs=1,
             batch_size=batch_size,
            #  callbacks=[reduce_lr, early_stopping],
             verbose=2
       )
      train_end = time.time()
      train_duration = train_end - train_start
      train_times.append(train_duration)
      val_accuracies.append(history.history['val_accuracy'][0])

      start = time.time()
      model.evaluate(X_test, y_test, verbose=0)
      end = time.time()
      test_duration = end - start
      test_times.append(end - start)
avg_train_time = np.mean(train_times)
avg_test_time = np.mean(test_times)

final_test_loss, final_test_acc = model.evaluate(X_test, y_test, verbose=0)

model.summary()

print("\n--- ANN Results ---")
print(f"Final Test Accuracy: {final_test_acc:.4f}")
print("Validation Accuracy per Epoch (first 5):",
      [f"{acc:.4f}" for acc in val_accuracies[:5]])
print(f"Total Parameters: {model.count_params()}")
print(f"Average Training Time per Epoch: {avg_train_time:.4f} seconds")
print(f"Average Testing Time per Epoch: {avg_test_time:.4f} seconds")
print(f"Layers: Input(784) -> Dense(512, relu) -> Dropout(0.2) -> Dense(256, relu) -> Dense(10, softmax)")
print("Learning Rate Used:", learning_rate)
#%% md
#SVM

#%%
import numpy as np
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

X_train_svm, _, y_train_svm, _ = train_test_split(X_train, y_train, train_size=8000, stratify=y_train, random_state=42)
X_test_svm, _, y_test_svm, _ = train_test_split(X_test, y_test, train_size=2000, stratify=y_test, random_state=42)
#%%
clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', gamma='scale', C=5))
start_train = time.time()
clf.fit(X_train_svm, y_train_svm)
end_train = time.time()
train_time = end_train - start_train

start_test = time.time()
y_pred = clf.predict(X_test_svm)
end_test = time.time()
test_time = end_test - start_test

acc = accuracy_score(y_test_svm, y_pred)

print("\n--- SVM Results ---")
print(f"Test Accuracy: {acc:.4f}")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Testing Time: {test_time:.4f} seconds")
#%% md
#CNN
#%%
from enum import Enum

class Optimizer(Enum):
    SGD = 'sgd'
    Adam = 'adam'
    RMSprop = 'rmsprop'

class ActivationFunction(Enum):
    ReLU = 'relu'
    Sigmoid = 'sigmoid'
    Tanh = 'tanh'
    Softmax = 'softmax'

#to use -> Optimizer.sgd

#%%
from enum import Enum
import tensorflow as tf
import numpy as np
import time


def CNN(activation_fun, conv_layers, epoch, optimizer, learning_rate, batch_size, fc_layers, fc_size, dropout_rate, dropout_pos=-1):
    model = tf.keras.Sequential()

    c_layers = min(conv_layers, 3)
    fc_layers = min(fc_layers, 4)

    for i in range(c_layers):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
                                             padding='same', activation=activation_fun.value, input_shape=X_train_cnn.shape[1:]))
        else:
            model.add(tf.keras.layers.Conv2D(32, (3,3), (1,1), padding='same', activation=activation_fun.value))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    model.add(tf.keras.layers.Flatten())

    for i in range(fc_layers):
        model.add(tf.keras.layers.Dense(fc_size, activation=activation_fun.value))
        if dropout_pos == i:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimizer_instance = tf.keras.optimizers.get({'class_name': optimizer.value, 'config': {'learning_rate': learning_rate}})
    model.compile(optimizer=optimizer_instance, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    train_times, test_times = [], []

    for i in range(epoch):
        start = time.time()
        model.fit(X_train_cnn, y_train_cnn, epochs=1, batch_size=batch_size, validation_data=(X_valid_cnn, y_valid_cnn))
        end_time = time.time()
        train_times.append(end_time - start)
        start = time.time()
        test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cnn)
        end = time.time()
        test_times.append(end - start)
    avg_train = np.mean(train_times)
    avg_test = np.mean(test_times)
    print(f"Total Training time: {np.sum(train_times)} seconds")
    print(f"AVG Training time per epoch: {avg_train} seconds")
    print(f"AVG Testing time per epoch: {avg_test} seconds")

    test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cnn)
    print(f"Test loss: {test_loss*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")

CNN(activation_fun=ActivationFunction.ReLU, conv_layers=3, epoch=10, optimizer=Optimizer.SGD, learning_rate=0.01, batch_size=32, fc_layers=2, fc_size=128, dropout_pos=-1, dropout_rate=0)
#%% md
### Testing different learning rates
#%%
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    print(f"\n--- Testing learning rate: {lr} ---")
    CNN(activation_fun=ActivationFunction.ReLU, conv_layers=3, epoch=10,
        optimizer=Optimizer.SGD, learning_rate=lr, batch_size=32,
        fc_layers=2, fc_size=128, dropout_pos=-1, dropout_rate=0)

#%% md
#Testing different number of FC layers
#%%
for fc_layer_count in [1, 2, 3, 4]:
    print(f"\n--- Testing FC layers: {fc_layer_count} ---")
    CNN(activation_fun=ActivationFunction.ReLU, conv_layers=3, epoch=10,
        optimizer=Optimizer.SGD, learning_rate=0.01, batch_size=32,
        fc_layers=fc_layer_count, fc_size=128, dropout_pos=-1, dropout_rate=0)

#%% md
### batch size = 2b
#%%
batch_sizes = [32, 64, 96]
for b in batch_sizes:
    print(f"\n--- Testing batch size: {b} ---")
    CNN(activation_fun=ActivationFunction.ReLU, conv_layers=3, epoch=10,
        optimizer=Optimizer.SGD, learning_rate=0.01, batch_size=b,
        fc_layers=2, fc_size=128, dropout_pos=-1, dropout_rate=0)

#%% md
##activation function
#%%
from enum import Enum

class ActivationFunction(Enum):
    ReLU = 'relu'
    Sigmoid = 'sigmoid'
    Tanh = 'tanh'
    ELU = 'elu'

activations = [ActivationFunction.ReLU, ActivationFunction.Sigmoid, ActivationFunction.Tanh, ActivationFunction.ELU]
for act in activations:
    print(f"\n--- Testing activation function: {act} ---")
    CNN(activation_fun=act, conv_layers=3, epoch=10, optimizer=Optimizer.SGD, learning_rate=0.01,
        batch_size=32, fc_layers=2, fc_size=128, dropout_pos=-1, dropout_rate=0)
#%% md
# optimizer

#%%
class Optimizer(Enum):
    SGD = 'sgd'
    Adam = 'adam'
    RMSprop = 'rmsprop'

optimizers = [Optimizer.SGD, Optimizer.Adam, Optimizer.RMSprop]

for opt in optimizers:
    print(f"\n--- Testing optimizer: {opt.value} ---")
    CNN(
        activation_fun=ActivationFunction.ReLU,
        conv_layers=3,
        epoch=10,
        optimizer=opt,
        learning_rate=0.01,
        batch_size=32,
        fc_layers=2,
        fc_size=128,
        dropout_pos=-1,
        dropout_rate=0
    )
#%% md
# dropout layer

#%%
dropout_rates = [0.0, 0.25, 0.5]
for rate in dropout_rates:
    print(f"\n--- Testing dropout rate: {rate} at position 1 ---")
    CNN(activation_fun=ActivationFunction.ReLU, conv_layers=3, epoch=10,
        optimizer=Optimizer.SGD, learning_rate=0.01, batch_size=32,
        fc_layers=3, fc_size=128, dropout_pos=1, dropout_rate=rate)
