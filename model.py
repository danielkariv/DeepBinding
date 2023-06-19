import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        # define model, set hyperparamaters.
        # NOTE: not real model of the project, just placing info for now..
        # Create a Sequential model
        model = Sequential()
        # Add the layers to the model
        model.add(Conv1D(filters=512, kernel_size=8, strides=1, activation='relu', input_shape=(41, 4)))
        model.add(MaxPooling1D(pool_size=5, strides=5))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        #model.compile(loss='mean_squared_error', optimizer='adam') # NOTE: not sure if we need it.
        model.summary()
        self.model = model
        return

    def plot_loss_accuracy(self, history):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(history.history["loss"],'r-x', label="Train Loss")
        ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
        ax.legend()
        ax.set_title('cross_entropy loss')
        ax.grid(True)


        ax = fig.add_subplot(1, 2, 2)
        ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
        ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
        ax.legend()
        ax.set_title('accuracy')
        ax.grid(True)
    
    def train(self, x_train, y_train, x_test, y_test):
        # takes training data and train the models.
        batch_size = 32
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0005, decay=1e-6)

        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        history = self.model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=15,
                    validation_data=(x_test, y_test),
                    shuffle=True)
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # NOTE: not sure if it can print it on normal script (not ipynb).
        self.plot_loss_accuracy(history)
        return
    def process(self, outputFile, RNAcompetePath, RBNSinputPath, RBNS5nMPath, RBNS20nMPath, RBNS80nMPath, RBNS320nMPath, RBNS1300nMPath):
        # convert paths into data we can run on.
        # TODO: have no idea how to do this.
        input = None
        predictions = self.model.predict(input)
        print(predictions)

        # save predictions.
        with open(outputFile, 'w') as file:
            # Iterate through each prediction
            for prediction in predictions:
                # Write each prediction to a new line in the file
                file.write(f"{prediction}\n")
        print(f'Saved predictions at file: {outputFile}')
        return
    def saveModel(self,filepath = 'weights.h5'):
        # Save only the model weights (so we use the same model, but different weights)
        self.model.save_weights(filepath)
        print(f'Saved weights at file: {filepath}')
        return
    def loadModel(self,filepath = 'weights.h5'):
        # load a state of model.
        # TODO: error out if not valid filepath, or failed to load.
        self.model.load_weights(filepath)
        print(f'Loaded weights from file: {filepath}')
        return
