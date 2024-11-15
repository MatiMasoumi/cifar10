import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, Callback
from tensorflow.keras.optimizers.schedules import PolynomialDecay, CosineDecay
import matplotlib.pyplot as plt
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train.shape
X_train[500]
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0
X_train.shape
X_test.shape
Y_train
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
datagen = ImageDataGenerator(
rotation_range=15,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.1,
zoom_range=0.1,
horizontal_flip=True,
fill_mode='nearest'
)

datagen.fit(X_train)
Augmented_Data = datagen.flow(X_train, Y_train, batch_size=50000, shuffle=False)
Augmented_Images, Augmented_Labels = next(Augmented_Data)
print(Augmented_Images.shape)
print(Augmented_Labels.shape)
Augmented_Images.shape
train_images_augmented = np.concatenate((X_train, Augmented_Images))
train_labels_augmented = np.concatenate((Y_train, Augmented_Labels))
train_images_augmented.shape
train_labels_augmented.shape
n1 = 32
n2 = 64
n3 = 128
n4 = 256

model = Sequential()

model.add(Conv2D(n1, (3,3), activation='relu', input_shape=(32,32,3), kernel_regularizer=regularizers.l2(0.001), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(layers.Dropout(0.25))

model.add(Conv2D(n2, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))


model.add(Conv2D(n3, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(Flatten())
model.add(Dense(n4, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_images_augmented, train_labels_augmented, batch_size=32, epochs=100, validation_split=0.2, callbacks=[reduce_lr,early_stopping])
def plot_loss_accuracy(history):
    # loss
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    #  accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.show()

plot_loss_accuracy(history)
model_pred = model.predict(X_test)
model_pred
model_loss, model_accuracy = model.evaluate(X_test, Y_test)
print(f"Accuracy: {model_accuracy} ==== Loss: {model_loss}")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(Y_test, axis=1)

Precision = precision_score(y_true_labels, y_pred_labels, average='macro')
Recall = recall_score(y_true_labels, y_pred_labels, average='macro')
F1 = f1_score(y_true_labels, y_pred_labels, average='macro')

print(f'Precision : {Precision}\nRecall : {Recall}\nF1-Score : {F1}')
