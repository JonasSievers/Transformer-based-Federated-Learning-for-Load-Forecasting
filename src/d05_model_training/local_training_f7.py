#Imports

#Pandas: Reading and analyzing data
import pandas as pd
#Tensorflow
import tensorflow as tf
#Create Folder for modelling checkpoint
import os
#Clear output after training
import IPython
import IPython.display
#Helper Class (Export Notebook as .py)
# helper functions
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
cwd = os.path.normpath(os.getcwd())
sys.path.insert(1, cwd + "/src/d00_utils") 
from local_helper_functions import *
from model_helper_functions import *
from windowgenerator import *

#Data Analytics

print("Get data")
#Read CSV file to pandas dataframe; encoding= 'unicode_escape': Decode from Latin-1 source code. Default UTF-8.
df = pd.read_csv(cwd+'/data/d03_data_processed/d03_data_processed.csv', encoding= 'unicode_escape', index_col='Date')
#Display smart meter names and amount
smart_meter_names = df.columns[2:-4]
print("Selected clients: ", len(smart_meter_names))

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

#Training epochs
MAX_EPOCHS = 100

# Create Windows 
windows_dict = createLocalDataWindows(smart_meter_names, INPUT_STEPS, OUT_STEPS, ds_dict)
print("Created Data windows")

# Local Learning
# Set random seed for as reproducible results as possible
tf.random.set_seed(42)


#h12 f7

#Build and save local models

#Build local models (LSTM, CNN, Transformer)
local_LSTM_models, local_CNN_models, local_Transformer_models = createLocalModels(
    smart_meter_names, INPUT_SHAPE[1], OUT_STEPS[0], NUM_FEATURES[1], 'Local_LSTM_F7_H12', 'Local_CNN_F7_H12', 'Local_Transformer_F7_H12',
    NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
    CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
)
#initilally save local models
initiallySaveAllLocalModels(cwd, smart_meter_names, local_LSTM_models, local_CNN_models, local_Transformer_models)
print("Saved local models for h12 f7") 


#Train local models
for idx, client in enumerate(smart_meter_names): 
    
    # Clear terminal and print current training round
    IPython.display.clear_output()
    print("Started with new client -----------------", idx+1, "/33")
    
    #Get local models
    local_LSTM_model, local_CNN_model, local_Transformer_model = loadLocalModels( 
        cwd, local_LSTM_models, local_CNN_models, local_Transformer_models, idx, client
    )
    print("Loaded local models")
    #compile and fit for n rounds
    local_LSTM_model = compile_fit(
        local_LSTM_model,
        windows_dict[client][2],
        MAX_EPOCHS
    )
    print("compiled and fitted LSTM")

    #compile and fit n rounds
    local_CNN_model = compile_fit(
        local_CNN_model,
        windows_dict[client][2],
        MAX_EPOCHS
    )
    print("compiled and fitted CNN")

    #Compile and fit n rounds
    local_Transformer_model = compile_fit(
        local_Transformer_model,
        windows_dict[client][2],
        MAX_EPOCHS
    )
    print("compiled and fitted Transformer")
    
    #Save Transformer model
    saveLocalModels(cwd, local_LSTM_model, local_CNN_model, local_Transformer_model, client)
    print("Saved local models")

print("Done h12 f7")
    
#h24 f7

#Build and save local models

#Build local models (LSTM, CNN, Transformer)
local_LSTM_models, local_CNN_models, local_Transformer_models = createLocalModels(
    smart_meter_names, INPUT_SHAPE[1], OUT_STEPS[1], NUM_FEATURES[1], 'Local_LSTM_F7_H24', 'Local_CNN_F7_H24', 'Local_Transformer_F7_H24',
    NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
    CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
)
#initilally save local models
initiallySaveAllLocalModels(cwd, smart_meter_names, local_LSTM_models, local_CNN_models, local_Transformer_models)
  


#Train local models
for idx, client in enumerate(smart_meter_names): 
    
    # Clear terminal and print current training round
    IPython.display.clear_output()
    print("Started with new client -----------------", idx+1, "/33")
    
    #Get local models
    local_LSTM_model, local_CNN_model, local_Transformer_model = loadLocalModels( 
        cwd, local_LSTM_models, local_CNN_models, local_Transformer_models, idx, client
    )
    
    #compile and fit for n rounds
    local_LSTM_model = compile_fit(
        local_LSTM_model,
        windows_dict[client][3],
        MAX_EPOCHS
    )
       
    #compile and fit n rounds
    local_CNN_model = compile_fit(
        local_CNN_model,
        windows_dict[client][3],
        MAX_EPOCHS
    )
    
    #Compile and fit n rounds
    local_Transformer_model = compile_fit(
        local_Transformer_model,
        windows_dict[client][3],
        MAX_EPOCHS
    )
        
    #Save Transformer model
    saveLocalModels(cwd, local_LSTM_model, local_CNN_model, local_Transformer_model, client)
    print("Saved local models")

print("Done h24 f7")