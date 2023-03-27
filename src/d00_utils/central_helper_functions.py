# Imports

#Pandas: Reading and analyzing data
import pandas as pd
#Numerical calcuations
import numpy as np


#Keras: Open-Source deep-learning library 
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from model_helper_functions import *
from windowgenerator import *


def createCentralDataWindows(INPUT_STEPS, OUT_STEPS, train_df_F7, val_df_F7, test_df_F7, train_df_F5, val_df_F5, test_df_F5): 
    """
    Create a window for every client considering each horizon (12, 24) and each featureset (5,7)
    
    :param smart_meter_names: names of clients
    :return: dictionary with structure windows_dict[client_i_smart_meter_names][0-3] 
        -> 0:window_F5_H12 , 1:window_F5_H24 , 2:window_F7_H12 , 3:window_F7_H24
    """
    
    #window_F5_H12
    window_F5_H12 = WindowGenerator(
        input_width=INPUT_STEPS, label_width=OUT_STEPS[0], shift=OUT_STEPS[0], 
        train_df = train_df_F5, val_df = val_df_F5, test_df = test_df_F5, label_columns=['load_value']
    )
    example_window = tf.stack([np.array(train_df_F5[10100:10100+window_F5_H12.total_window_size]),
                            np.array(train_df_F5[2000:2000+window_F5_H12.total_window_size]),
                            np.array(train_df_F5[3000:3000+window_F5_H12.total_window_size])])
    example_inputs, example_labels = window_F5_H12.split_window(example_window)
    window_F5_H12.example = example_inputs, example_labels

    #window_F5_H24
    window_F5_H24 = WindowGenerator(
        input_width=INPUT_STEPS, label_width=OUT_STEPS[1], shift=OUT_STEPS[1], 
        train_df = train_df_F5, val_df = val_df_F5, test_df = test_df_F5, label_columns=['load_value']
    )
    example_window = tf.stack([np.array(train_df_F5[10100:10100+window_F5_H24.total_window_size]),
                            np.array(train_df_F5[2000:2000+window_F5_H24.total_window_size]),
                            np.array(train_df_F5[3000:3000+window_F5_H24.total_window_size])])
    example_inputs, example_labels = window_F5_H24.split_window(example_window)
    window_F5_H24.example = example_inputs, example_labels

    #window_F7_H12
    window_F7_H12 = WindowGenerator(
        input_width=INPUT_STEPS, label_width=OUT_STEPS[0], shift=OUT_STEPS[0], 
        train_df = train_df_F7, val_df = val_df_F7, test_df = test_df_F7, label_columns=['load_value']
    )
    example_window = tf.stack([np.array(train_df_F7[10100:10100+window_F7_H12.total_window_size]),
                            np.array(train_df_F7[2000:2000+window_F7_H12.total_window_size]),
                            np.array(train_df_F7[3000:3000+window_F7_H12.total_window_size])])
    example_inputs, example_labels = window_F7_H12.split_window(example_window)
    window_F7_H12.example = example_inputs, example_labels

    #window_F5_H24
    window_F7_H24 = WindowGenerator(
        input_width=INPUT_STEPS, label_width=OUT_STEPS[1], shift=OUT_STEPS[1], 
        train_df = train_df_F7, val_df = val_df_F7, test_df = test_df_F7, label_columns=['load_value']
    )
    example_window = tf.stack([np.array(train_df_F7[10100:10100+window_F7_H24.total_window_size]),
                            np.array(train_df_F7[2000:2000+window_F7_H24.total_window_size]),
                            np.array(train_df_F7[3000:3000+window_F7_H24.total_window_size])])
    example_inputs, example_labels = window_F7_H24.split_window(example_window)
    window_F7_H24.example = example_inputs, example_labels

    return window_F5_H12, window_F5_H24, window_F7_H12, window_F7_H24



def createCentralModels(
        INPUT_SHAPE, OUT_STEPS, NUM_FEATURES, LSTM_NAME, CNN_NAME, Tansformer_NAME, 
        NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
        CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
    ):
    """
    Create a local LSTM, CNN, and Transofrmer model for each of the clusters
    
    :param: architecture parameters of the models
    :return: 3 arrays with number of clusters LSTM, CNN, and Transofrmer models
    """
    #Build Models
    central_LSTM_model = LSTM_Model().build(
        input_shape = INPUT_SHAPE, 
        num_LSTM_cells = NUM_LSTM_CELLS,
        num_LSTM_layers = NUM_LSTM_LAYERS,
        num_LSTM_dense_layers = NUM_LSTM_DENSE_LAYERS,
        num_LSTM_dense_units = NUM_LSTM_DENSE_UNITS,
        LSTM_dropout = LSTM_DROPOUT,
        output_steps = OUT_STEPS,
        num_features = NUM_FEATURES,
        model_name = LSTM_NAME
    )
    #CNN        
    central_CNN_model = CNN_Model().build(
        input_shape = INPUT_SHAPE, 
        conv_width = CONV_WIDTH,
        num_CNN_layers = NUM_CNN_LAYERS,
        num_CNN_filters = NUM_CNN_FILTERS,
        num_CNN_dense_layers = NUM_CNN_DENSE_LAYERS,
        num_CNN_dense_units = NUM_CNN_DENSE_UNITS,
        CNN_dropout = CNN_DROPOUT,
        output_steps = OUT_STEPS,
        num_features = NUM_FEATURES,
        model_name = CNN_NAME
    )
    #Transformer
    central_Transformer_model = Transformer_Model().build(
        input_shape = INPUT_SHAPE,
        output_steps = OUT_STEPS,
        num_features = NUM_FEATURES,
        model_name = Tansformer_NAME    
    )
         
    return central_LSTM_model, central_CNN_model, central_Transformer_model

def initiallySaveAllCentralModels(cwd, central_LSTM_model, central_CNN_model, central_Transformer_model):
    """
    Saves the initial local models in file /data/d05_models/cluser{i}/MODELNAME/FederatedRound{i}
    
    :param: current working directory,  models
    """
        
    # LSTM
    central_LSTM_model.save(cwd + f"/data/d05_models/central/{central_LSTM_model.name}/Round{0}")
    #Cnn
    central_CNN_model.save(cwd + f"/data/d05_models/central/{central_CNN_model.name}/Round{0}")
    #Transformer
    central_Transformer_model.save(cwd + f"/data/d05_models/central/{central_Transformer_model.name}/Round{0}")
    

def loadCentralModels( cwd, central_LSTM_model, central_CNN_model, central_Transformer_model):
    """
    load the local model of the last federated training round. If called in federated round 0, then the initial local model is retuned
    
    :param: path, local models, cluster index, index of federated round
    :return: local models
    """
               
    #load initial model
    central_LSTM_model = keras.models.load_model(cwd + f"/data/d05_models/central/{central_LSTM_model.name}/Round{0}", compile=False)
    central_CNN_model = keras.models.load_model(cwd + f"/data/d05_models/central/{central_CNN_model.name}/Round{0}", compile=False)
    central_Transformer_model = keras.models.load_model(cwd + f"/data/d05_models/central/{central_Transformer_model.name}/Round{0}", compile=False)
    
    return central_LSTM_model, central_CNN_model, central_Transformer_model

def compile_fit(central_model, window, MAX_EPOCHS):
    """
    Takes a model, compiles it, sets local weights, fits the model and retunrs new weights
    
    :param: model, local weights, the window to train and validate with
    :return: array of sclaed weights
    """
    #Compile Model (define loss, optimizer, metrics)
    central_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), 
            tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.MeanAbsoluteError(),
        ]
    )
       
    #fit local model with client's data
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
    central_model.fit(
        window.train, 
        epochs=MAX_EPOCHS, 
        verbose=1, 
        validation_data=window.val,
        callbacks=[
            timetaken,
            early_stopping, 
            #create_model_checkpoint(save_path=save_path), 
        ]
    )
      
    return central_model

def saveCentralModels(cwd, central_LSTM_model, central_CNN_model, central_Transformer_model):
    """
    Save the Local models  
    
    :param: local models, cluster idx und federated round idx
    """
    # LSTM
    central_LSTM_model.save(cwd + f"/data/d05_models/central/{central_LSTM_model.name}/Round{100}")
    #Cnn
    central_CNN_model.save(cwd + f"/data/d05_models/central/{central_CNN_model.name}/Round{100}")
    #Transformer
    central_Transformer_model.save(cwd + f"/data/d05_models/central/{central_Transformer_model.name}/Round{100}")



        
