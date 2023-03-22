#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Imports

#Pandas: Reading and analyzing data
import pandas as pd
#Numerical calcuations
import numpy as np
#statistical data visualization
import seaborn as sns
#Use Dates in Datetime Format
import datetime
#Tensorflow
import tensorflow as tf
#Keras: Open-Source deep-learning library 
from tensorflow import keras
#Building blocks of NN in Keras
from tensorflow.keras import layers
#EarlyStop to stop training early
from tensorflow.keras.callbacks import EarlyStopping
#Functional API: Layers for different models
from keras.layers import Dense, LSTM, Dropout
#Normalization
from sklearn.preprocessing import MinMaxScaler
#Standardization
from sklearn.preprocessing import StandardScaler
#Evaluate models
import math
#Evaluate MSE
from sklearn.metrics import mean_squared_error
#plot numpy array
import matplotlib.pyplot as plt
#Create Folder for modelling checkpoint
import os
#Callback to logg model fitting time
import time
from timeit import default_timer as timer
#Clear output after training
import IPython
import IPython.display
#Normalization
from sklearn.preprocessing import MinMaxScaler
#Standardization
from sklearn.preprocessing import StandardScaler


# Create a function to implement a ModelCheckpoint callback with a specific filename 

def create_model_checkpoint(save_path):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path), # create filepath to save model
        verbose=0, # only output a limited amount of text
        save_best_only=True) # save only the best model to file

#Helper function for time callback to log training time per epoch
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append((epoch, timer()-self.starttime))
        
timetaken = timecallback()


class LSTM_Model:
    @staticmethod
    def build(
        input_shape, num_LSTM_cells, num_LSTM_layers, num_LSTM_dense_layers, num_LSTM_dense_units, LSTM_dropout,
        output_steps, num_features, model_name
    ):
        #Input dimensions are (N x n_past x #features)
        inputs = keras.Input(shape=input_shape, name="Input")
        #Hidden LSTM Layers
        x = layers.LSTM(num_LSTM_cells, return_sequences=True)(inputs)

        for _ in range(num_LSTM_layers):
            x = layers.LSTM(num_LSTM_cells, return_sequences=True)(x)

        x = layers.LSTM(num_LSTM_cells, return_sequences=False)(x)

        #Dropout and Dense Layers
        for _ in range(num_LSTM_dense_layers):
            x = layers.Dense(num_LSTM_dense_units, activation="relu")(x)
            x = layers.Dropout(LSTM_dropout)(x)

        #Output
        x = layers.Dense(output_steps*num_features, kernel_initializer=tf.initializers.zeros())(x)
        outputs = layers.Reshape([output_steps, num_features])(x)

        multi_lstm_model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        return multi_lstm_model
        
class CNN_Model:
    @staticmethod
    def build(
        input_shape, conv_width, num_CNN_layers, num_CNN_filters, num_CNN_dense_layers, 
        num_CNN_dense_units, CNN_dropout, output_steps, num_features, model_name
    ):
        #Input dimensions are (N x n_past x #features)
        inputs = keras.Input(shape=input_shape, name="Input")
        #Hidden LSTM Layers
        x = layers.Lambda(lambda x: x[:, -conv_width:, :])(inputs)

        for _ in range(num_CNN_layers):
            x = layers.Conv1D(num_CNN_filters, kernel_size=(conv_width), activation='relu', padding="same")(x)
            x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)

        for _ in range(num_CNN_dense_layers):
            x = layers.Dense(num_CNN_dense_units, activation="relu")(x)

        x = layers.Dropout(CNN_dropout)(x)
        x = layers.Dense(output_steps*num_features, kernel_initializer=tf.initializers.zeros())(x)
        outputs = layers.Reshape([output_steps, num_features])(x)
        #Create model
        multi_conv_model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        return multi_conv_model

class Transformer_Model:
    @staticmethod
    def build(input_shape, output_steps, num_features, model_name):
        
        encoder_inputs = keras.layers.Input(shape=input_shape)

        #ENCODER
        #Encoder Layer 1
        x = keras.layers.MultiHeadAttention(key_dim=4, num_heads=2, dropout=0.2)(encoder_inputs, encoder_inputs)
        res1 = x + encoder_inputs #Add
        x = keras.layers.Dense(input_shape[1], activation='relu')(x) # Feed Forward Part
        res2 = x + res1 #Add
        x_out = keras.layers.LayerNormalization(epsilon=1e-6)(res2) #Norm


        #Encoder Layer 2
        x = keras.layers.MultiHeadAttention(key_dim=4, num_heads=2, dropout=0.2)(x_out, x_out)
        res1 = x + x_out #Add
        x = keras.layers.Dense(input_shape[1], activation='relu')(x) # Feed Forward Part
        res2 = x + res1 #Add
        x_out = keras.layers.LayerNormalization(epsilon=1e-6)(res2) #Norm

        encoder = keras.layers.LSTM(input_shape[1], return_state=True)
        encoder_outputs_and_states = encoder(x_out)
        encoder_states = encoder_outputs_and_states[1:]
        #-----------------------------------------------------------------------------------------
        decoder_inputs = encoder_inputs

        #DECODER
        #Decoder Layer 1
        x = keras.layers.MultiHeadAttention(key_dim=4, num_heads=2, dropout=0.2)(decoder_inputs, decoder_inputs)
        res1 = x + decoder_inputs #Add
        x = keras.layers.Dense(input_shape[1], activation='relu')(x) # Feed Forward Part
        res2 = x + res1 #Add
        x_out = keras.layers.LayerNormalization(epsilon=1e-6)(res2) #Norm

        decoder = keras.layers.LSTM(input_shape[1], return_state=True, return_sequences=True)
        decoder_outputs_and_states = decoder(x_out, initial_state=encoder_states)
        decoder_outputs = decoder_outputs_and_states[0] # Only select the output of the decoder (not the states)

        #Decoder Layer 2
        x = keras.layers.MultiHeadAttention(key_dim=4, num_heads=2, dropout=0.2)(decoder_outputs, x_out)
        res1 = x + decoder_outputs #Add
        x = keras.layers.Dense(input_shape[1], activation='relu')(x) # Feed Forward Part
        res2 = x + res1 #Add
        x_out = keras.layers.LayerNormalization(epsilon=1e-6)(res2) #Norm

        decoder = keras.layers.LSTM(input_shape[1], return_state=True, return_sequences=True)
        decoder_outputs_and_states = decoder(x_out, initial_state=encoder_states)
        decoder_outputs = decoder_outputs_and_states[0] # Only select the output of the decoder (not the states)


        #Feedforward
        x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(decoder_outputs)
        x = keras.layers.Dense(output_steps*num_features, kernel_initializer=tf.initializers.zeros())(x)
        outputs = keras.layers.Reshape([output_steps, num_features])(x)

        return keras.models.Model(inputs=encoder_inputs, outputs=outputs, name=model_name) 
    


def test_model(window, model, client_name, MAX_EPOCHS):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    model.compile(loss=tf.keras.losses.MeanSquaredError(), #FL_OPTIMIZER
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[
                    tf.keras.metrics.RootMeanSquaredError(), 
                    tf.keras.metrics.MeanAbsolutePercentageError(),
                    tf.keras.metrics.MeanAbsoluteError(),
                ])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping, create_model_checkpoint(
                          save_path=f"model_experiments/Federated/{model.name}/{client_name}"
                      ), timetaken])
    
    #Load best model
    model = tf.keras.models.load_model(f"model_experiments/Federated/{model.name}/{client_name}")
    model_evaluation_test = model.evaluate(window.test)
   
    return model_evaluation_test

