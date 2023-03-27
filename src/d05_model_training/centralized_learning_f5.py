# Imports

#Pandas: Reading and analyzing data
import pandas as pd
#Numerical calcuations
import numpy as np
#statistical data visualization
import seaborn as sns
#Use Dates in Datetime Format
from datetime import timedelta
#Tensorflow
import tensorflow as tf
#Keras: Open-Source deep-learning library 
import os
#Clear output after training
import IPython
import IPython.display


import sys
# caution: path[0] is reserved for script path (or '' in REPL)
cwd = os.path.normpath(os.getcwd())
sys.path.insert(1, cwd + "/src/d00_utils") 
from central_helper_functions import *
from model_helper_functions import *
from windowgenerator import *

#Data Analytics
#Data Analytics

print("Get data")
#Read CSV file to pandas dataframe; encoding= 'unicode_escape': Decode from Latin-1 source code. Default UTF-8.
df = pd.read_csv(cwd+'/data/d03_data_processed/d03_data_processed.csv', encoding= 'unicode_escape', index_col='Date')
#Display smart meter names and amount
smart_meter_names = df.columns[2:-4]
print("Selected clients: ", len(smart_meter_names))

#Centralized Dataframe

time_series_data = df
time_series_data.reset_index(inplace=True)

df_final = pd.DataFrame()
for name in smart_meter_names:
    df2 = pd.DataFrame()
    df2[['Date', 'load_value', 'temp', 'rhum', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']] = time_series_data[['Date', name, 'temp', 'rhum', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']].copy()
    df2['smart_meter'] = name
    df_final = pd.concat([df_final, df2], ignore_index=True)

df_final.set_index('Date', drop=True, inplace=True)
df_final.index = pd.to_datetime(df_final.index)

#Dataset
#Datasets
#Calculate the Date equal to 70% of the Dateerange 01.01.2019-31.12.2019 -> devide by number clients as dates exist 33 times(each per client)
last_date_train = pd.to_datetime('01.01.2019 00:00:00', format='%d.%m.%Y %H:%M:%S') + timedelta(hours=int((len(df_final)/33)*0.7))
#Calculate the Date equal to 90% of the Dateerange 01.01.2019-31.12.2019
last_date_val = pd.to_datetime('01.01.2019 00:00:00', format='%d.%m.%Y %H:%M:%S') + timedelta(hours=int((len(df_final)/33)*0.9))

#Split Datasets considering the dates and then droping the Dates column as only time features are considered
train_df_F7 = df_final.loc[(df_final.index <= last_date_train), ['load_value', 'temp', 'rhum', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']]
val_df_F7 = df_final.loc[((df_final.index > last_date_train)&(df_final.index >= last_date_val)), ['load_value', 'temp', 'rhum', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']]
test_df_F7 = df_final.loc[(df_final.index > last_date_val), ['load_value', 'temp', 'rhum', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']]

#Split Datasets considering the dates and then droping the Dates column as only time features are considered
train_df_F5 = df_final.loc[(df_final.index <= last_date_train), ['load_value', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']]
val_df_F5 = df_final.loc[((df_final.index > last_date_train)&(df_final.index >= last_date_val)), ['load_value', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']]
test_df_F5 = df_final.loc[(df_final.index > last_date_val), ['load_value', 'hour sin', 'hour cos', 'dayofweek sin', 'dayofweek cos']]

print("Done Train, Val, Test split")
#Hyperparameter

#ITERATING
OUT_STEPS = [12, 24] #Next 12 or 24 hours
NUM_FEATURES = [5, 7] # [F_T, F_TW] load_value, hour sin, hour cos, dayofweek sin, dayofweek cos + (temp, rhum)
INPUT_STEPS = 24
INPUT_SHAPE = [(INPUT_STEPS, NUM_FEATURES[0]), (INPUT_STEPS, NUM_FEATURES[1])]

#Forecasting
MAX_EPOCHS = 100

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

# Create Windows 
window_F5_H12, window_F5_H24, window_F7_H12, window_F7_H24 = createCentralDataWindows(INPUT_STEPS, OUT_STEPS, train_df_F7, val_df_F7, test_df_F7, train_df_F5, val_df_F5, test_df_F5)
print("Created Data windows")

# Local Learning
# Set random seed for as reproducible results as possible
tf.random.set_seed(42)


#h12 f5

#Build and save local models

#Build local models (LSTM, CNN, Transformer)
central_LSTM_model, central_CNN_model, central_Transformer_model = createCentralModels(
    INPUT_SHAPE[0], OUT_STEPS[0], NUM_FEATURES[0], 'Central_LSTM_F5_H12', 'Central_CNN_F5_H12', 'Central_Transformer_F5_H12',
    NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
    CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
)
#initilally save local models
initiallySaveAllCentralModels(cwd, central_LSTM_model, central_CNN_model, central_Transformer_model)
print("Saved central models for h12 f5")

#Get central models
central_LSTM_model, central_CNN_model, central_Transformer_model = loadCentralModels( 
    cwd, central_LSTM_model, central_CNN_model, central_Transformer_model
)
print("Loaded local models")

#compile and fit for n rounds
central_LSTM_model = compile_fit(
    central_LSTM_model,
    window_F5_H12,
    MAX_EPOCHS
)
print("compiled and fitted LSTM")

#compile and fit n rounds
central_CNN_model = compile_fit(
    central_CNN_model,
    window_F5_H12,
    MAX_EPOCHS
)
print("compiled and fitted CNN")

#Compile and fit n rounds
central_Transformer_model = compile_fit(
    central_Transformer_model,
    window_F5_H12,
    MAX_EPOCHS
)
print("compiled and fitted Transformer")

#Save Transformer model
saveCentralModels(cwd, central_LSTM_model, central_CNN_model, central_Transformer_model)
print("Saved local models")

print("Done h12 f5")



#h24 f5

#Build and save local models

#Build local models (LSTM, CNN, Transformer)
central_LSTM_model, central_CNN_model, central_Transformer_model = createCentralModels(
    INPUT_SHAPE[0], OUT_STEPS[1], NUM_FEATURES[0], 'Central_LSTM_F5_H24', 'Central_CNN_F5_H24', 'Central_Transformer_F5_H24',
    NUM_LSTM_CELLS, NUM_LSTM_LAYERS, NUM_LSTM_DENSE_LAYERS, NUM_LSTM_DENSE_UNITS, LSTM_DROPOUT, 
    CONV_WIDTH, NUM_CNN_LAYERS, NUM_CNN_FILTERS, NUM_CNN_DENSE_LAYERS, NUM_CNN_DENSE_UNITS, CNN_DROPOUT, 
)
#initilally save local models
initiallySaveAllCentralModels(cwd, central_LSTM_model, central_CNN_model, central_Transformer_model)
print("Saved central models for h24 f5")

#Get central models
central_LSTM_model, central_CNN_model, central_Transformer_model = loadCentralModels( 
    cwd, central_LSTM_model, central_CNN_model, central_Transformer_model
)
print("Loaded local models")

#compile and fit for n rounds
central_LSTM_model = compile_fit(
    central_LSTM_model,
    window_F5_H24,
    MAX_EPOCHS
)
print("compiled and fitted LSTM")

#compile and fit n rounds
central_CNN_model = compile_fit(
    central_CNN_model,
    window_F5_H24,
    MAX_EPOCHS
)
print("compiled and fitted CNN")

#Compile and fit n rounds
central_Transformer_model = compile_fit(
    central_Transformer_model,
    window_F5_H24,
    MAX_EPOCHS
)
print("compiled and fitted Transformer")

#Save Transformer model
saveCentralModels(cwd, central_LSTM_model, central_CNN_model, central_Transformer_model)
print("Saved local models")

print("Done h24 f5")
