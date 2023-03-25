#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

def weight_scalling_factor(clients_trn_data, client_name, client_names):
    """
    weight_scalling_factor calculates the proportion of a client’s local training data 
    with the overall training data held by all clients. First the client’s batch size is obtained and used 
    to calculate its number of data points.Then the overall global training data size is obtained (global_count).
    Finally we calculated the scaling factor as a fraction (return). 
    """
    #get the bs
    bs = list(clients_trn_data)[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data).numpy()*bs
    return local_count/global_count

def scale_model_weights(weight, scalar):
    """
    scale_model_weights scales each of the local model’s weights based the value of their scaling factor calculated in weight_scalling_factor
    """
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    """
    Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights
    """
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad
  
    
def createGlobalModelsForClusters(
        windows_dict, INPUT_SHAPE, OUT_STEPS, NUM_FEATURES, LSTM_NAME, CNN_NAME, Tansformer_NAME, 
        NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
        CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
    ):
    """
    Create a global LSTM, CNN, and Transofrmer model for each of the clusters
    
    :param: architecture parameters of the models
    :return: 3 arrays with number of clusters LSTM, CNN, and Transofrmer models
    """
    
    ### Features 5, Horizon 12
    global_LSTM_models = []
    global_CNN_models = []
    global_Transformer_models = []

    for idx, cluster in enumerate(windows_dict):

        #Build Models
        global_LSTM_models.append(LSTM_Model().build(
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
        global_CNN_models.append(CNN_Model().build(
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
        global_Transformer_models.append(Transformer_Model().build(
            input_shape = INPUT_SHAPE,
            output_steps = OUT_STEPS,
            num_features = NUM_FEATURES,
            model_name = Tansformer_NAME    
        ))
         
    return global_LSTM_models, global_CNN_models, global_Transformer_models
    
    
def getClientNamesOfCluster(windows_dict, cluster):
    """
    Get a list of all clients within the current cluster 
    
    :param windows_dict: dictionary with data windos sorted by cluster
    :return: list of client names within current cluster
    """
    
    #Get names of clients within cluster
    client_names = list()
    for client in windows_dict[cluster]:
        client_names.append(client)
        
    return client_names


def compile_fit_set_weights(local_model, global_weights, window, client, client_names, MAX_EPOCHS):
    """
    Takes a model, compiles it, sets global weights, fits the model and retunrs new weights
    
    :param: model, global weights, the window to train and validate with
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
    
    #set local model weight to the weight of the global model
    local_model.set_weights(global_weights)
    
    #fit local model with client's data
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
    local_model.fit(
        window.train, 
        epochs=MAX_EPOCHS, 
        verbose=1, 
        validation_data=window.val,
        callbacks=[
            timetaken,
            #early_stopping, 
            #create_model_checkpoint(save_path=save_path), 
        ]
    )
    
    #scale the model weights and add to list        
    scaling_factor = weight_scalling_factor(window.train, client, client_names)
    scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
    
    return scaled_weights
    

def loadGlobalModels( cwd, global_LSTM_models, global_CNN_models, global_Transformer_models, idx, idx_com):
    """
    load the global model of the last federated training round. If called in federated round 0, then the initial global model is retuned
    
    :param: path, global models, cluster index, index of federated round
    :return: global models
    """
        
    #load global model of last federated round if not first round
    if idx_com != 0:
        idx_com = idx_com-1
        
    #load model
    global_LSTM_model = keras.models.load_model(cwd + f"/data/d05_models/cluster{idx}/{global_LSTM_models[idx].name}/FederatedRound{idx_com}", compile=False)
    global_CNN_model = keras.models.load_model(cwd + f"/data/d05_models/cluster{idx}/{global_CNN_models[idx].name}/FederatedRound{idx_com}", compile=False)
    global_Transformer_model = keras.models.load_model(cwd + f"/data/d05_models/cluster{idx}/{global_Transformer_models[idx].name}/FederatedRound{idx_com}", compile=False)
    
    return global_LSTM_model, global_CNN_model, global_Transformer_model
    
def getGlobalModelWeights(global_LSTM_model, global_CNN_model, global_Transformer_model):
    """
    retunrs the weights of the gobal models
    """      
    #Get model weights
    global_LSTM_weights = global_LSTM_model.get_weights()
    global_CNN_weights = global_CNN_model.get_weights()
    global_Transformer_weights = global_Transformer_model.get_weights()
                                                       
    return global_LSTM_weights, global_CNN_weights, global_Transformer_weights

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------Change clusters
def initiallySaveAllGlobalModels(cwd, global_LSTM_models, global_CNN_models, global_Transformer_models):
    """
    Saves the initial global models in file /data/d05_models/cluser{i}/MODELNAME/FederatedRound{i}
    
    :param: current working directory,  models
    """
        
    for cluster_idx in range(6):
        # LSTM
        global_LSTM_models[cluster_idx].save(cwd + f"/data/d05_models/cluster{cluster_idx}/{global_LSTM_models[cluster_idx].name}/FederatedRound{0}")
        #Cnn
        global_CNN_models[cluster_idx].save(cwd + f"/data/d05_models/cluster{cluster_idx}/{global_CNN_models[cluster_idx].name}/FederatedRound{0}")
        #Transformer
        global_Transformer_models[cluster_idx].save(cwd + f"/data/d05_models/cluster{cluster_idx}/{global_Transformer_models[cluster_idx].name}/FederatedRound{0}")

def saveGlobalModels(cwd, global_LSTM_model, global_CNN_model, global_Transformer_model, idx, idx_com):
    """
    Save the global models  
    
    :param: global models, cluster idx und federated round idx
    """
    # LSTM
    global_LSTM_model.save(cwd + f"/data/d05_models/cluster{idx}/{global_LSTM_model.name}/FederatedRound{idx_com}")
    #Cnn
    global_CNN_model.save(cwd + f"/data/d05_models/cluster{idx}/{global_CNN_model.name}/FederatedRound{idx_com}")
    #Transformer
    global_Transformer_model.save(cwd + f"/data/d05_models/cluster{idx}/{global_Transformer_model.name}/FederatedRound{idx_com}")

