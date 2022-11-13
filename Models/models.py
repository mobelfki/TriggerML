#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, AveragePooling2D, GlobalMaxPooling2D, BatchNormalization, LayerNormalization, LSTM, concatenate, Input, Conv2D, MaxPooling2D, Flatten, PReLU, LeakyReLU, LayerNormalization
from tensorflow.keras import initializers
seed = 412
import numpy as np
np.random.seed(seed)
from numpy.random import seed as random_seed
random_seed(seed)
tf.random.set_seed(seed)
# Supercells model 


def test_model():
    I1 = Input((64, 30, 1), name='Lr1')
    X1 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I1)
    X1 = MaxPooling2D((3,3), strides=(2,2)) (X1)
    X1 = Flatten(X1)

    I2 = Input((64, 30, 1), name='Lr2')
    X2 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I2)
    X2 = MaxPooling2D((3,3), strides=(2,2)) (X2)
    X2 = Flatten(X2)

    I3 = Input((64, 30, 1), name='Lr3')
    X3 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I3)
    X3 = MaxPooling2D((3,3), strides=(2,2)) (X3)
    X3 = Flatten(X3)

    I4 = Input((64, 30, 1), name='Lr4')
    X4 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I4)
    X4 = MaxPooling2D((3,3), strides=(2,2)) (X4)
    X4 = Flatten(X4)

    I5 = Input((64, 30, 1), name='Lr5')
    X5 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I5)
    X5 = MaxPooling2D((3,3), strides=(2,2)) (X5)
    X5 = Flatten(X5)

    I6_R0 = Input((64, 30, 1), name='Lr8_R0')
    X6_R0 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I6_R0)
    X6_R0 = MaxPooling2D((3,3), strides=(2,2)) (X6_R0)
    X6_R0 = Flatten(X6_R0)

    I7_R0 = Input((64, 30, 1), name='Lr8_R0')
    X7_R0 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I7_R0)
    X7_R0 = MaxPooling2D((3,3), strides=(2,2)) (X7_R0)
    X7_R0 = Flatten(X7_R0)

    I8_R0 = Input((64, 30, 1), name='Lr8_R0')
    X8_R0 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I8_R0)
    X8_R0 = MaxPooling2D((3,3), strides=(2,2)) (X8_R0)
    X8_R0 = Flatten(X8_R0)

    I6_R1 = Input((64, 30, 1), name='Lr6_R1')
    X6_R1 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I6_R1)
    X6_R1 = MaxPooling2D((3,3), strides=(2,2)) (X6_R1)
    X6_R1 = Flatten(X6_R1)

    I7_R1 = Input((64, 30, 1), name='Lr7_R1')
    X7_R1 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I7_R1)
    X7_R1 = MaxPooling2D((3,3), strides=(2,2)) (X7_R1)
    X7_R1 = Flatten(X7_R1)

    I8_R1 = Input((64, 30, 1), name='Lr8_R1')
    X8_R1 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (X8_R1)
    X8_R1 = MaxPooling2D((3,3), strides=(2,2)) (X8_R1)
    X8_R1 = Flatten(X8_R1)


    I21 = Input((64, 30, 1), name='Lr21')
    X21 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I21)
    X21 = MaxPooling2D((3,3), strides=(2,2)) (X12)
    X21 = Flatten(X12)

    I22 = Input((64, 30, 1), name='Lr22')
    X22 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I22)
    X22 = MaxPooling2D((3,3), strides=(2,2)) (X22)
    X22 = Flatten(X22)

    I23 = Input((64, 30, 1), name='Lr23')
    X23 = Conv2D(16,  activation='selu', kernel_size=(5,5), bias_initializer='zeros', kernel_initializer='he_normal') (I23)
    X23 = MaxPooling2D((3,3), strides=(2,2)) (X23)
    X23 = Flatten(X23)

    X = concatenate([X1, X2, X3, X4, X5, X6_R0, X7_R0, X8_R0, X6_R1, X7_R1, X8_R1, X21, X22, X23], name='Concatenate')
    X = LayerNormalization()(X)
	       
    X = Dense(128, activation='selu', bias_initializer='zeros', kernel_initializer='he_normal')(X)
    X = LayerNormalization()(X)
    X = Dropout(0.1)(X)
	       
    X = Dense(128, activation='selu', bias_initializer='zeros', kernel_initializer='he_normal')(X)
    X = LayerNormalization()(X)
    X = Dropout(0.1)(X)
	       
    X = Dense(128, activation='selu', bias_initializer='zeros', kernel_initializer='he_normal')(X)
    X = LayerNormalization()(X)
    X = Dropout(0.1)(X)
	       
    X = Dense(128, activation='selu', bias_initializer='zeros', kernel_initializer='he_normal')(X)
    X = LayerNormalization()(X)
    X = Dropout(0.1)(X)
	       
    input_layer = [I1, I2, I3, I4, I5, I6_R0, I7_R0, I8_R0, I6_R1, I7_R1, I8_R1, I21, I22, I23]
    output = Dense(2, activation='linear', bias_initializer='zeros', kernel_initializer='he_normal', name='output')(X)

    model = Model(inputs=input_layer, outputs=output)
    return model

class SuperCellsModel():

    def __init__(self, name, LayersNames, LayersShapes, Noutputs, Conv, Pool, Denses):

	    self.name = name
	    self.LayersNames  = LayersNames
	    self.LayersShapes = LayersShapes
	    self.Noutputs = Noutputs
	    self.Conv = Conv 
	    self.Pool = Pool
	    self.Denses = Denses
    
    def get_model(self):
        self.Inputs = []
        hidden_layers = []

        l = 0
        for layer in self.LayersNames:
            input_layer = Input(tuple(self.LayersShapes[layer]), name=layer)

            #norm_input_layer = LayerNormalization()(input_layer)
            i = 0
            self.x_layer = input_layer

            for conv, pool in zip(self.Conv[layer], self.Pool[layer]):

                kernel = initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
                if conv['initializer'] != 'variance': kernel = conv['initializer'] 
                self.x_layer = Conv2D(conv['neurons'],  activation=conv['activation'], kernel_size=tuple(conv['kernel']), bias_initializer='zeros', kernel_initializer=kernel, name='Conv{0}_'.format(i)+layer) (self.x_layer) 	   
                self.x_layer = MaxPooling2D(tuple(pool['kernel']), strides=tuple(pool['strides']), name='Pool{0}_'.format(i)+layer) (self.x_layer)
                i = i+1

            l = l+1

            self.x_layer = Flatten(name='Flatten_'+layer)(self.x_layer)
            self.Inputs.append(input_layer)

            hidden_layers.append(self.x_layer)

        X = concatenate(hidden_layers, name='Concatenate')
        X = LayerNormalization()(X)

        i = 0
        for item in self.Denses: 


            if item['type'] == 'Dense':
                kernel = initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
                if item['initializer'] != 'variance': kernel = item['initializer']

                X = Dense(item['neurons'], activation=item['activation'], bias_initializer='zeros', kernel_initializer=kernel, name='Dense{0}'.format(i))(X)
            if item['type'] == 'Norm':
                X = LayerNormalization()(X)

            if item['type'] == 'Dropout':
                X = Dropout(item['rate'], name='Drop{0}'.format(i))(X)

            i = i+1
            	    
        self.Output = Dense(self.Noutputs, activation='linear', bias_initializer='zeros', kernel_initializer='he_normal', name='output')(X) 	    	   

        return Model(inputs=self.Inputs, outputs=self.Output)
