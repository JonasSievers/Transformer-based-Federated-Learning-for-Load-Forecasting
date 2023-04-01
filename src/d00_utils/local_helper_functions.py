# Imports

#Pandas: Reading and analyzing data
import pandas as pd
#Numerical calcuations
import numpy as np
#Evaluate models
import math

#Keras: Open-Source deep-learning library 
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from model_helper_functions import *
from windowgenerator import *


def createLocalDataWindows(smart_meter_names, INPUT_STEPS, OUT_STEPS, ds_dict): 
    """
    Create a window for every client considering each horizon (12, 24) and each featureset (5,7)
    
    :param smart_meter_names: names of clients
    :return: dictionary with structure windows_dict[client_i_smart_meter_names][0-3] 
        -> 0:window_F5_H12 , 1:window_F5_H24 , 2:window_F7_H12 , 3:window_F7_H24
    """
    
    windows_dict = {}
    
    for i, client in enumerate(smart_meter_names):
        
        #window_F5_H12
        window_F5_H12 = WindowGenerator(
            input_width=INPUT_STEPS, label_width=OUT_STEPS[0], shift=OUT_STEPS[0], 
            train_df = ds_dict[client][3], val_df = ds_dict[client][4], test_df = ds_dict[client][5], label_columns=[client]
        )
        example_window = tf.stack([np.array(ds_dict[client][3][10100:10100+window_F5_H12.total_window_size]),
                                   np.array(ds_dict[client][3][2000:2000+window_F5_H12.total_window_size]),
                                   np.array(ds_dict[client][3][3000:3000+window_F5_H12.total_window_size])])
        example_inputs, example_labels = window_F5_H12.split_window(example_window)
        window_F5_H12.example = example_inputs, example_labels

        #window_F5_H24
        window_F5_H24 = WindowGenerator(
            input_width=INPUT_STEPS, label_width=OUT_STEPS[1], shift=OUT_STEPS[1], 
            train_df = ds_dict[client][3], val_df = ds_dict[client][4], test_df = ds_dict[client][5], label_columns=[client]
        )
        example_window = tf.stack([np.array(ds_dict[client][3][10100:10100+window_F5_H24.total_window_size]),
                                   np.array(ds_dict[client][3][2000:2000+window_F5_H24.total_window_size]),
                                   np.array(ds_dict[client][3][3000:3000+window_F5_H24.total_window_size])])
        example_inputs, example_labels = window_F5_H24.split_window(example_window)
        window_F5_H24.example = example_inputs, example_labels

        #window_F7_H12
        window_F7_H12 = WindowGenerator(
            input_width=INPUT_STEPS, label_width=OUT_STEPS[0], shift=OUT_STEPS[0], 
            train_df = ds_dict[client][0], val_df = ds_dict[client][1], test_df = ds_dict[client][2], label_columns=[client]
        )
        example_window = tf.stack([np.array(ds_dict[client][0][10100:10100+window_F7_H12.total_window_size]),
                                   np.array(ds_dict[client][0][2000:2000+window_F7_H12.total_window_size]),
                                   np.array(ds_dict[client][0][3000:3000+window_F7_H12.total_window_size])])
        example_inputs, example_labels = window_F7_H12.split_window(example_window)
        window_F7_H12.example = example_inputs, example_labels

        #window_F5_H24
        window_F7_H24 = WindowGenerator(
            input_width=INPUT_STEPS, label_width=OUT_STEPS[1], shift=OUT_STEPS[1], 
            train_df = ds_dict[client][0], val_df = ds_dict[client][1], test_df = ds_dict[client][2], label_columns=[client]
        )
        example_window = tf.stack([np.array(ds_dict[client][0][10100:10100+window_F7_H24.total_window_size]),
                                   np.array(ds_dict[client][0][2000:2000+window_F7_H24.total_window_size]),
                                   np.array(ds_dict[client][0][3000:3000+window_F7_H24.total_window_size])])
        example_inputs, example_labels = window_F7_H24.split_window(example_window)
        window_F7_H24.example = example_inputs, example_labels

        windows_dict['{}'.format(client)] = [window_F5_H12, window_F5_H24, window_F7_H12, window_F7_H24]
    
    return windows_dict

def createLocalModels(
        smart_meter_names, INPUT_SHAPE, OUT_STEPS, NUM_FEATURES, LSTM_NAME, CNN_NAME, Tansformer_NAME, 
        NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
        CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
    ):
    """
    Create a local LSTM, CNN, and Transofrmer model for each of the clusters
    
    :param: architecture parameters of the models
    :return: 3 arrays with number of clusters LSTM, CNN, and Transofrmer models
    """
    
    ### Features 5, Horizon 12
    local_LSTM_models = []
    local_CNN_models = []
    local_Transformer_models = []

    for idx, client in enumerate(smart_meter_names):

        #Build Models
        local_LSTM_models.append(LSTM_Model().build(
            input_shape = INPUT_SHAPE, 
            num_LSTM_cells = NUM_LSTM_CELLS,
            num_LSTM_layers = NUM_LSTM_LAYERS,
            num_LSTM_dense_layers = NUM_LSTM_DENSE_LAYERS,
            num_LSTM_dense_units = NUM_LSTM_DENSE_UNITS,
            LSTM_dropout = LSTM_DROPOUT,
            output_steps = OUT_STEPS,
            num_features = NUM_FEATURES,
            model_name = LSTM_NAME
        ))
        #CNN        
        local_CNN_models.append(CNN_Model().build(
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
        ))
        #Transformer
        local_Transformer_models.append(Transformer_Model().build(
            input_shape = INPUT_SHAPE,
            output_steps = OUT_STEPS,
            num_features = NUM_FEATURES,
            model_name = Tansformer_NAME    
        ))
         
    return local_LSTM_models, local_CNN_models, local_Transformer_models

def initiallySaveAllLocalModels(cwd, smart_meter_names, local_LSTM_models, local_CNN_models, local_Transformer_models):
    """
    Saves the initial local models in file /data/d05_models/cluser{i}/MODELNAME/FederatedRound{i}
    
    :param: current working directory,  models
    """
        
    for idx, client in enumerate(smart_meter_names):
        # LSTM
        local_LSTM_models[idx].save(cwd + f"/data/d05_models/local/{client}/{local_LSTM_models[idx].name}/Round{0}")
        #Cnn
        local_CNN_models[idx].save(cwd + f"/data/d05_models/local/{client}/{local_CNN_models[idx].name}/Round{0}")
        #Transformer
        local_Transformer_models[idx].save(cwd + f"/data/d05_models/local/{client}/{local_Transformer_models[idx].name}/Round{0}")

def initiallySaveAllLocalModelsW168(cwd, smart_meter_names, local_LSTM_models, local_CNN_models, local_Transformer_models):
    """
    Saves the initial local models in file /data/d05_models/cluser{i}/MODELNAME/FederatedRound{i}
    
    :param: current working directory,  models
    """
        
    for idx, client in enumerate(smart_meter_names):
        # LSTM
        local_LSTM_models[idx].save(cwd + f"/data/d05_models/local/W168/{client}/{local_LSTM_models[idx].name}/Round{0}")
        #Cnn
        local_CNN_models[idx].save(cwd + f"/data/d05_models/local/W168/{client}/{local_CNN_models[idx].name}/Round{0}")
        #Transformer
        local_Transformer_models[idx].save(cwd + f"/data/d05_models/local/W168/{client}/{local_Transformer_models[idx].name}/Round{0}")

def loadLocalModels( cwd, local_LSTM_models, local_CNN_models, local_Transformer_models, idx, client):
    """
    load the local model of the last federated training round. If called in federated round 0, then the initial local model is retuned
    
    :param: path, local models, cluster index, index of federated round
    :return: local models
    """
               
    #load initial model
    local_LSTM_model = keras.models.load_model(cwd + f"/data/d05_models/local/{client}/{local_LSTM_models[idx].name}/Round{0}", compile=False)
    local_CNN_model = keras.models.load_model(cwd + f"/data/d05_models/local/{client}/{local_CNN_models[idx].name}/Round{0}", compile=False)
    local_Transformer_model = keras.models.load_model(cwd + f"/data/d05_models/local/{client}/{local_Transformer_models[idx].name}/Round{0}", compile=False)
    
    return local_LSTM_model, local_CNN_model, local_Transformer_model

def loadLocalModelsW168( cwd, local_LSTM_models, local_CNN_models, local_Transformer_models, idx, client):
    """
    load the local model of the last federated training round. If called in federated round 0, then the initial local model is retuned
    
    :param: path, local models, cluster index, index of federated round
    :return: local models
    """
               
    #load initial model
    local_LSTM_model = keras.models.load_model(cwd + f"/data/d05_models/local/W168/{client}/{local_LSTM_models[idx].name}/Round{0}", compile=False)
    local_CNN_model = keras.models.load_model(cwd + f"/data/d05_models/local/W168/{client}/{local_CNN_models[idx].name}/Round{0}", compile=False)
    local_Transformer_model = keras.models.load_model(cwd + f"/data/d05_models/local/W168/{client}/{local_Transformer_models[idx].name}/Round{0}", compile=False)
    
    return local_LSTM_model, local_CNN_model, local_Transformer_model

def compile_fit(local_model, window, MAX_EPOCHS):
    """
    Takes a model, compiles it, sets local weights, fits the model and retunrs new weights
    
    :param: model, local weights, the window to train and validate with
    :return: array of sclaed weights
    """
    #Compile Model (define loss, optimizer, metrics)
    local_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), 
            tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.MeanAbsoluteError(),
        ]
    )
       
    #fit local model with client's data
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
    local_model.fit(
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
      
    return local_model

def saveLocalModels(cwd, local_LSTM_model, local_CNN_model, local_Transformer_model, client):
    """
    Save the Local models  
    
    :param: local models, cluster idx und federated round idx
    """
    # LSTM
    local_LSTM_model.save(cwd + f"/data/d05_models/local/{client}/{local_LSTM_model.name}/Round{100}")
    #Cnn
    local_CNN_model.save(cwd + f"/data/d05_models/local/{client}/{local_CNN_model.name}/Round{100}")
    #Transformer
    local_Transformer_model.save(cwd + f"/data/d05_models/local/{client}/{local_Transformer_model.name}/Round{100}")

def saveLocalModelsW168(cwd, local_LSTM_model, local_CNN_model, local_Transformer_model, client):
    """
    Save the Local models  
    
    :param: local models, cluster idx und federated round idx
    """
    # LSTM
    local_LSTM_model.save(cwd + f"/data/d05_models/local/W168/{client}/{local_LSTM_model.name}/Round{100}")
    #Cnn
    local_CNN_model.save(cwd + f"/data/d05_models/local/W168/{client}/{local_CNN_model.name}/Round{100}")
    #Transformer
    local_Transformer_model.save(cwd + f"/data/d05_models/local/W168/{client}/{local_Transformer_model.name}/Round{100}")