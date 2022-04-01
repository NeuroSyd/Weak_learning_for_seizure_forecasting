import os
import numpy as np

from keras.models import Sequential, Model
# from keras.layers import Merge, Input
from keras.layers import Input,DepthwiseConv2D,AveragePooling2D,SeparableConv2D,SpatialDropout2D,TimeDistributed,Bidirectional,ConvLSTM2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D,Conv1D
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import max_norm
from keras import backend as K
from keras.preprocessing import image
import math
from models.customCallbacks import MyEarlyStopping, MyModelCheckpoint
from keras.layers import Reshape
from keras import backend as K
K.set_image_data_format('channels_last')



# Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
def loss(y_true, y_pred):
    # y_pred = [max(min(pred[0], 1-K.epsilon()), K.epsilon()) for pred in y_pred]
    y_pred = K.maximum(K.minimum(y_pred, 1 - K.epsilon()), K.epsilon())
    t_loss = (-1) * (K.exp(y_true) * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred) / K.exp(y_pred))

    return K.mean(t_loss)


class ConvNN(object):

    def __init__(self, batch_size=1, nb_classes=2, epochs=100):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.epochs = epochs
    #nhan model for 7s
    def setup(self, X_train_shape):
        # print ('X_train shape', X_train_shape)
        # Input shape = (None,13,2,126,1)
        inputs = Input(shape=X_train_shape[1:])

        normal1 = BatchNormalization(
            axis=2,
            name='normal1')(inputs)

        convlstm1 = ConvLSTM2D(
            filters=16,
            kernel_size=(X_train_shape[2], 3),
            padding='valid', strides=(1, 2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=True,
            name='convlstm1')(normal1)

        convlstm2 = ConvLSTM2D(
            filters=32,
            kernel_size=(1, 3),
            padding='valid', strides=(1, 2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=True,
            name='convlstm2')(convlstm1)

        convlstm3 = ConvLSTM2D(
            filters=64,
            kernel_size=(1, 3),
            padding='valid', strides=(1, 2),
            activation='tanh',
            dropout=0.0, recurrent_dropout=0.0,
            return_sequences=False,
            name='convlstm3')(convlstm2)

        flat = Flatten()(convlstm3)

        drop1 = Dropout(0.5)(flat)

        dens1 = Dense(256, activation='sigmoid', name='dens1')(drop1)
        drop2 = Dropout(0.5)(dens1)

        dens2 = Dense(self.nb_classes, name='dens2')(drop2)

        # option to include temperature in softmax
        temp = 1.0
        temperature = Lambda(lambda x: x / temp)(dens2)
        last = Activation('softmax')(temperature)

        self.model = Model(inputs, last)
        # previous use 5e-4
        adam = Adam(lr=5e-8, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])

        # print(self.model.summary())
        return self

    def fit(self,X_train,Y_train,X_val=None, y_val=None):
        Y_train = Y_train.astype('uint8')
        #print(Y_train)
        Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
        #y_val = np_utils.to_categorical(y_val, self.nb_classes)

        early_stop = MyEarlyStopping(patience=5, verbose=0)
        checkpointer = MyModelCheckpoint(
            filepath="weights_update_by_detection_clstm_2011_2.h5",
            verbose=0, save_best_only=True)


        if (y_val is None):
            self.model.fit(X_train, Y_train, batch_size=self.batch_size,
                            epochs=self.epochs,validation_split=0.2,
                            callbacks=[early_stop,checkpointer], verbose=2
                            )
        else:
            self.model.fit(X_train, Y_train, batch_size=self.batch_size,
                            epochs=self.epochs,validation_data=(X_val,y_val),
                            callbacks=[early_stop,checkpointer], verbose=2
                            )
        self.model.load_weights("weights_update_by_detection_clstm_2011_1.h5")
        return self

    def fit_generator(self, training_generator, validation_generator,savepath,class_weights_train):
        early_stop = MyEarlyStopping(patience=5, verbose=0)
        #2channel_sftf_3_train_whole
        checkpointer = MyModelCheckpoint(
            filepath=savepath,
            verbose=0, save_best_only=True)
        #class_weight = {0: 20.,
                        #1: 1,
                        #}
        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 epochs=self.epochs,class_weight=class_weights_train,
                                 use_multiprocessing=False, workers=4,
                                 callbacks=[early_stop,checkpointer])

        return self



    def load_trained_weights(self, filename):
        self.model.load_weights(filename)
        print ('Loading pre-trained weights from %s.' %filename)
        return self

    def predict_proba(self,X):
        return self.model.predict([X])[:,1]

    def evaluate_generator(self,X):
        predictions = self.model.predict_generator(generator=X,verbose=1)[:,1]

        return predictions

    def evaluate(self, X, y):
        predictions = self.model.predict(X, verbose=0)[:,1]
        from sklearn.metrics import roc_auc_score
        auc_test = roc_auc_score(y, predictions)
        print('Test AUC is:', auc_test)
        return auc_test