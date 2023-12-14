import os
import cv2
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Activation, BatchNormalization, Flatten, InputLayer, Input, Lambda
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

from Metrics import l1_distance, l1_distance_output_shape
from itertools import combinations

class DuplicateDetector:
    def __init__(self):
        self.filenames          = os.listdir('./datasets/mars/')
        self.gradientOptimizer  = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    def coreNetwork(self):

        model = Sequential([
            InputLayer(input_shape=(160, 60, 3)),
            Conv2D(64, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(64, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            MaxPool2D(pool_size=(2,2), strides=(2,2)),

            Conv2D(128, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(128, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            MaxPool2D(pool_size=(2,2), strides=(2,2)),

            Conv2D(256, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(256, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(256, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            MaxPool2D(pool_size=(2,2), strides=(2,2)),

            Conv2D(512, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(512, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(512, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            MaxPool2D(pool_size=(2,2), strides=(2,2)),

            Conv2D(512, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(512, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            Conv2D(512, (3,3), padding='same'),
            BatchNormalization(axis=1, momentum=0.99, epsilon=1e-3),
            Activation('relu'),
            MaxPool2D(pool_size=(2,2), strides=(2,2)),

            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(6, activation='relu')
            # Dense(22, activation='relu')
        ])

        return model

    def extendedNetwork(self):
        core            = self.coreNetwork()
        input1          = Input(shape=(160, 60, 3,))
        input2          = Input(shape=(160, 60, 3,))
        feature_vec1    = core(input1)
        feature_vec2    = core(input2)
        distance        = Lambda(l1_distance, output_shape=l1_distance_output_shape)([feature_vec1, feature_vec2])
        output          = Activation('sigmoid')(distance)
        return Model(inputs=[input1, input2], outputs=output)
        
    def load_data(self):
        x           = np.array([cv2.imread(os.path.join(os.path.abspath('./datasets/mars/'), filename)) for filename in self.filenames])
        x_resized   = np.array([cv2.resize(img, (60, 160)) for img in x])  # Resize images to (160, 60)
        labels      = np.array([int(filename[:4]) for filename in self.filenames])
            
        return x_resized, to_categorical(labels)
    
    def load_data_pairs(self):
        x           = np.array([cv2.imread(os.path.join(os.path.abspath('./datasets/mars/'), filename)) for filename in self.filenames])
        labels      = np.array([int(filename[:4]) for filename in self.filenames])
        idx         = np.argsort(labels)
        x           = x[idx]
        labels      = np.sort(labels)

        input           = zip(x[:100], labels[:100])
        input1, input2  = [], []
        labels          = []

        combs           = list(combinations(input, 2))
        input1          += [comb[0][0] for comb in combs]
        input2          += [comb[1][0] for comb in combs]
        labels          += [comb[0][1] == comb[1][1] for comb in combs]

        input1          = np.array(input1)
        input2          = np.array(input2)
        labels          = np.array(labels)

        return input1, input2, labels
    
    def trainCoreNetwork(self):
        x, labels = self.load_data()

        model = self.coreNetwork()
        model.compile(loss='categorical_crossentropy', 
                      optimizer=self.gradientOptimizer)
        
        model.fit(x, 
                  labels, 
                  batch_size=50, 
                  epochs=2)

        model.save_weights('model.h5py')

    def trainExtendedNetwork(self):
        input1, input2, labels  = self.load_data_pairs()

        model                   = self.extendedNetwork()

        model.compile(optimizer='rmsprop', 
                      loss='binary_crossentropy'
                      )
        
        self.weightfilepath     = './siamese/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        
        callbackCheckpoint      = ModelCheckpoint(self.filepath, 
                                                  monitor='val_loss', 
                                                  verbose=0, 
                                                  save_best_only=False, 
                                                  save_weights_only=True, 
                                                  mode='auto', 
                                                  period=1
                                                  )
        
        callbackTensorBoard     = TensorBoard(log_dir='./siamese/tensorboard', 
                                              histogram_freq=0, 
                                              write_graph=True, 
                                              write_images=True
                                              )

        model.fit([input1, input2], 
                  labels, 
                  epochs=100, 
                  batch_size=20, 
                  verbose=2, 
                  shuffle=True, 
                  validation_split=0.2, 
                  callbacks=[callbackCheckpoint, callbackTensorBoard]
                  )
        

    def train(self):
        self.trainCoreNetwork()
        self.trainExtendedNetwork()

    def detect(self, newFrame, oldFrame):
        # write detection