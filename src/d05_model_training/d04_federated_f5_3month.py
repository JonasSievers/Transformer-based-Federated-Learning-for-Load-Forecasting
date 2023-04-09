#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Imports

# pandas for reading and analyzing data
import pandas as pd
# numpy for numerical calcuations
import numpy as np
# os to find path of files 
import os

# tensorflow as machine learning library
import tensorflow as tf
# keras as open-source deep-learning library 
from tensorflow import keras
# building blocks of NN in Keras
from keras import layers
from keras import backend as K

# IPython to Clear terminal output
import IPython
import IPython.display
# time and timeit to provie a callback to logg model fitting time
from timeit import default_timer as timer
# logging to logg debug, errors, info, warning, error information
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

# helper functions
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
cwd = os.path.normpath(os.getcwd())
sys.path.insert(1, cwd + "/src/d00_utils/") 
#Add path to utils folder to VS Code (Strg + ,) -> python.analysis.extraPaths -> add item
# C:/Users/jonas/transformerBasedFederatedLearningForSecureSTLFInSG/src/d00_utils/
from federated_helper_functions import *
from model_helper_functions import *
from windowgenerator import *


#Data Analytics
print("Get data")
# get current working directory and go back one folder to main working directory
cwd = os.path.normpath(os.getcwd())
#Read CSV file to pandas dataframe; encoding= 'unicode_escape': Decode from Latin-1 source code. Default UTF-8.
df = pd.read_csv(cwd+'/data/d03_data_processed/d03_data_processed_3month.csv', encoding= 'unicode_escape', index_col='Date')
#Display smart meter names and amount
smart_meter_names = df.columns[2:-4]
print("Selected clients: ", len(smart_meter_names))


#Get clustered clients
N_CLUSTERS = 6
y = np.loadtxt(cwd+'/data/d04_clients_clustered/d04_clients_clustered.csv', delimiter=',').astype(int)
print("Clustered clients: ", y)

# Make Datasets for the 33 clients and for 5 and 7 features
ds_dict = makeDatasetsForclientsAndfeatures(smart_meter_names, df)
print("Created dictionary with datasets")

#Set Hyperparameter
#Data Shape
OUT_STEPS = [12, 24] #Next 12 or 24 hours
NUM_FEATURES = [5, 7] # [F_T, F_TW] load_value, hour sin, hour cos, dayofweek sin, dayofweek cos + (temp, rhum)
INPUT_STEPS = 24
INPUT_SHAPE = [(INPUT_STEPS, NUM_FEATURES[0]), (INPUT_STEPS, NUM_FEATURES[1])]

#LSTM
NUM_LSTM_LAYERS = 4
NUM_LSTM_CELLS = 32
NUM_LSTM_DENSE_LAYERS=1
NUM_LSTM_DENSE_UNITS = 32
LSTM_DROPOUT = 0.2

#CNN
CONV_WIDTH = 3
NUM_CNN_LAYERS = 4
NUM_CNN_FILTERS = 24
NUM_CNN_DENSE_LAYERS = 1
NUM_CNN_DENSE_UNITS = 32
CNN_DROPOUT = 0.2

#Federated Learning
comms_round = 20
#Training epochs
MAX_EPOCHS = 2


# Create Windows 
windows_dict = createDataWindows(y, smart_meter_names, INPUT_STEPS, OUT_STEPS, ds_dict, N_CLUSTERS)
print("Created Data windows")

#Select smallest cluster only for testing
#windows_dict = {k: v for k, v in windows_dict.items() if k == 4}
#print(windows_dict)

# Federated Learning
# Set random seed for as reproducible results as possible
tf.random.set_seed(42)

#Create Global models
global_LSTM_models, global_CNN_models, global_Transformer_models = createGlobalModelsForClusters(
        windows_dict, INPUT_SHAPE[0], OUT_STEPS[0], NUM_FEATURES[0], 'Federated_LSTM_F5_H12', 'Federated_CNN_F5_H12', 'Federated_Transformer_F5_H12',
        NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
        CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
    )
# Save global models to not loose any data when experiment crashes
initiallySaveAllGlobalModelsGeneral(cwd, '3month', global_LSTM_models, global_CNN_models, global_Transformer_models)
print("Created and saved global models for each cluster")

# Iterate through federated learning for number of training rounds

# For testing only 2 federated rounds
# for idx_com, comm_round in enumerate(range(3,5)):
for idx_com, comm_round in enumerate(range(comms_round)):

    # Clear terminal and print current training round
    IPython.display.clear_output()
    print("Started Federated training round ----------", idx_com+1, "/", comms_round)

    #Train and update models for each cluster
    for idx, cluster in enumerate(windows_dict):

        print("Cluster--------", idx+1, "/", N_CLUSTERS)

        #Get global models
        global_LSTM_model, global_CNN_model, global_Transformer_model = loadGlobalModelsGeneral( 
            cwd, '3month', global_LSTM_models, global_CNN_models, global_Transformer_models, idx, idx_com
        )
        # Get the global model's weights 
        global_LSTM_weights, global_CNN_weights, global_Transformer_weights = getGlobalModelWeights(
            global_LSTM_model, global_CNN_model, global_Transformer_model)
        print("Got global models")

        #initial list for local model weights after scalling
        scaled_local_weight_LSTM_list = list()
        scaled_local_weight_CNN_list = list()
        scaled_local_weight_Transformer_list = list()

        client_names = getClientNamesOfCluster(windows_dict, cluster)

        for client in windows_dict[cluster].keys():

            #LSTM
            local_LSTM_model = LSTM_Model().build(
                INPUT_SHAPE[0], NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS,
                LSTM_DROPOUT, OUT_STEPS[0], NUM_FEATURES[0], 'Federated_local_LSTM_F5_H12'
            )
            scaled_weights = compile_fit_set_weights(
                local_LSTM_model, 
                global_LSTM_weights, 
                windows_dict[cluster][client][0], 
                client, 
                client_names, 
                MAX_EPOCHS, 

            )
            scaled_local_weight_LSTM_list.append(scaled_weights)
            print("Trained local LSTM")

            #CNN
            local_CNN_model = CNN_Model().build(
                INPUT_SHAPE[0], CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS,
                CNN_DROPOUT, OUT_STEPS[0], NUM_FEATURES[0],'Federated_local_CNN_F5_H24'
            )    
            scaled_weights = compile_fit_set_weights(
                local_CNN_model, 
                global_CNN_weights, 
                windows_dict[cluster][client][0], 
                client, 
                client_names, 
                MAX_EPOCHS, 
            )
            scaled_local_weight_CNN_list.append(scaled_weights)
            print("Trained local CNN")

            #Transformer
            local_Transformer_model = Transformer_Model().build(
                INPUT_SHAPE[0],OUT_STEPS[0],NUM_FEATURES[0],'Federated_local_Transformer_F5_H24'    
            )
            scaled_weights = compile_fit_set_weights(
                local_Transformer_model, 
                global_Transformer_weights, 
                windows_dict[cluster][client][0], 
                client, 
                client_names, 
                MAX_EPOCHS, 
            )
            scaled_local_weight_Transformer_list.append(scaled_weights)
            print("Trained local Transformer")
            
            #clear session to free memory after each communication round
            K.clear_session()

        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights_LSTM = sum_scaled_weights(scaled_local_weight_LSTM_list)
        average_weights_CNN = sum_scaled_weights(scaled_local_weight_CNN_list)
        average_weights_Transformer = sum_scaled_weights(scaled_local_weight_Transformer_list)

        #update global model 
        global_LSTM_models[idx].set_weights(average_weights_LSTM)
        global_CNN_models[idx].set_weights(average_weights_CNN)
        global_Transformer_models[idx].set_weights(average_weights_Transformer)

        #Save global models
        saveGlobalModelsGeneral(cwd, '3month', global_LSTM_models[idx], global_CNN_models[idx], global_Transformer_models[idx], idx, idx_com)
        print("Saved Global models")


#h24
#Create Global models
global_LSTM_models, global_CNN_models, global_Transformer_models = createGlobalModelsForClusters(
        windows_dict, INPUT_SHAPE[0], OUT_STEPS[1], NUM_FEATURES[0], 'Federated_LSTM_F5_H24', 'Federated_CNN_F5_H24', 'Federated_Transformer_F5_H24',
        NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
        CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
    )
# Save global models to not loose any data when experiment crashes
initiallySaveAllGlobalModelsGeneral(cwd, '3month', global_LSTM_models, global_CNN_models, global_Transformer_models)
print("Created and saved global models for each cluster")

# Iterate through federated learning for number of training rounds

# For testing only 2 federated rounds
# for idx_com, comm_round in enumerate(range(3,5)):
for idx_com, comm_round in enumerate(range(comms_round)):

    # Clear terminal and print current training round
    IPython.display.clear_output()
    print("Started Federated training round ----------", idx_com+1, "/", comms_round)

    #Train and update models for each cluster
    for idx, cluster in enumerate(windows_dict):

        print("Cluster--------", idx+1, "/", N_CLUSTERS)

        #Get global models
        global_LSTM_model, global_CNN_model, global_Transformer_model = loadGlobalModelsGeneral( 
            cwd, '3month', global_LSTM_models, global_CNN_models, global_Transformer_models, idx, idx_com
        )
        # Get the global model's weights 
        global_LSTM_weights, global_CNN_weights, global_Transformer_weights = getGlobalModelWeights(
            global_LSTM_model, global_CNN_model, global_Transformer_model)
        print("Got global models")

        #initial list for local model weights after scalling
        scaled_local_weight_LSTM_list = list()
        scaled_local_weight_CNN_list = list()
        scaled_local_weight_Transformer_list = list()

        client_names = getClientNamesOfCluster(windows_dict, cluster)

        for client in windows_dict[cluster].keys():

            #LSTM
            local_LSTM_model = LSTM_Model().build(
                INPUT_SHAPE[0], NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS,
                LSTM_DROPOUT, OUT_STEPS[1], NUM_FEATURES[0], 'Federated_local_LSTM_F5_H24'
            )
            scaled_weights = compile_fit_set_weights(
                local_LSTM_model, 
                global_LSTM_weights, 
                windows_dict[cluster][client][1], 
                client, 
                client_names, 
                MAX_EPOCHS, 

            )
            scaled_local_weight_LSTM_list.append(scaled_weights)
            print("Trained local LSTM")

            #CNN
            local_CNN_model = CNN_Model().build(
                INPUT_SHAPE[0], CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS,
                CNN_DROPOUT, OUT_STEPS[1], NUM_FEATURES[0],'Federated_local_CNN_F5_H24'
            )    
            scaled_weights = compile_fit_set_weights(
                local_CNN_model, 
                global_CNN_weights, 
                windows_dict[cluster][client][1], #F5H24
                client, 
                client_names, 
                MAX_EPOCHS, 
            )
            scaled_local_weight_CNN_list.append(scaled_weights)
            print("Trained local CNN")

            #Transformer
            local_Transformer_model = Transformer_Model().build(
                INPUT_SHAPE[0],OUT_STEPS[1],NUM_FEATURES[0],'Federated_local_Transformer_F5_H24'    
            )
            scaled_weights = compile_fit_set_weights(
                local_Transformer_model, 
                global_Transformer_weights, 
                windows_dict[cluster][client][1], 
                client, 
                client_names, 
                MAX_EPOCHS, 
            )
            scaled_local_weight_Transformer_list.append(scaled_weights)
            print("Trained local Transformer")
            
            #clear session to free memory after each communication round
            K.clear_session()

        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights_LSTM = sum_scaled_weights(scaled_local_weight_LSTM_list)
        average_weights_CNN = sum_scaled_weights(scaled_local_weight_CNN_list)
        average_weights_Transformer = sum_scaled_weights(scaled_local_weight_Transformer_list)

        #update global model 
        global_LSTM_models[idx].set_weights(average_weights_LSTM)
        global_CNN_models[idx].set_weights(average_weights_CNN)
        global_Transformer_models[idx].set_weights(average_weights_Transformer)

        #Save global models
        saveGlobalModelsGeneral(cwd, '3month', global_LSTM_models[idx], global_CNN_models[idx], global_Transformer_models[idx], idx, idx_com)
        print("Saved Global models h24")

