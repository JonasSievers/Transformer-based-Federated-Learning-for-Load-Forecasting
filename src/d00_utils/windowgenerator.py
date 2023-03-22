#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[1]:


"""
The WindowGenerator class can:
- Handle the indexes and offsets.
- Split windows of features into (features, labels) pairs.
- Plot the content of the resulting windows.
- Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.

Main Features of the input window:
- Input width/ window (given time steps of hirstory values for forecasting)
- offset/ shift (Time between history values and predictions -> for predicting next value: Offset equals Label width)
- Label width/ Window (time steps you predict into the future)
- Total width/ window = Input width + Offset
"""
class WindowGenerator():
    
    """
    The __init__ method includes all the necessary logic for the input and label indices.
    INPUT: 
    - windows: input_width, label_width, shift/offset
    - datasets: train, test, val
    - label_columns (optional) -> 1 or multiple outputs you want to predict
    
    PROCESS:
    - Store: 
        Datasets: train, val, test 
        Width: input, label, shift, total_window_size
    - Identifies: 
        Label_columns_indices: Indice of the column with the label
        column_indices: All indices of features + label(s)
        input_slice: Slice object slice(0, input_width)
        input_indices: Indices of Input
        label_start: Index where label starts
        labels_slice: slice object
        label_indices: label indexs
    """
    def __init__(self, input_width, label_width, shift,train_df, val_df, test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        
    #Method to represent class's object as a String
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    """
    Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a window of labels.
    
    Input: Features
    
    Returns: Input, Output
    """
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    """
    Stores an example window to visualize partial data
    """
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
  

    """
    Method to visualize the split window
    Input:
        model: If non, method visulaizes Input and Labels, if model is given also predictions are plotted
        plot_col: Which Column to plot
        max_subplots: Number of subplots, should match the input size of split_window example

    """
    def plot(self, model=None, plot_col='0213-ZE01-71', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c')
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        
        
    """
    INPUT: Dataframe
    
    OUTPUT: tf.data.Dataset of (input_window, label_window) pairs 
        using the tf.keras.utils.timeseries_dataset_from_array function
    """
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=False,
          batch_size=32,)

        ds = ds.map(self.split_window)

        return ds
        
    """
    3 Methods that take a dataframe and return a dataset for training, validation, testing
    """
    
    #The @property lets a method to be accessed as an attribute instead of as a method with a '()'
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

