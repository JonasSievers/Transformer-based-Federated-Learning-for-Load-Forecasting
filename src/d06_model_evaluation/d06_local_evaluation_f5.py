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
# earlyStop to stop training early
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras import backend as K

# IPython to Clear terminal output
import IPython
import IPython.display

# pickle to save dictionary in file
import pickle 
# time and timeit to provie a callback to logg model fitting time
from timeit import default_timer as timer

# helper functions
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
cwd = os.path.normpath(os.getcwd())
sys.path.insert(1, cwd + "/src/d00_utils") 
from local_helper_functions import *
from model_helper_functions import *
from windowgenerator import *
from evaluation_helper_functions import *



# Hyperparameter
OUT_STEPS = [12, 24] #Next 12 or 24 hours
INPUT_STEPS = 24
#Training epochs
MAX_EPOCHS = 100

#Data Analytics
print("Get data")
# get current working directory and go back one folder to main working directory
cwd = os.path.normpath(os.getcwd())
#Read CSV file to pandas dataframe; encoding= 'unicode_escape': Decode from Latin-1 source code. Default UTF-8.
df = pd.read_csv(cwd+'/data/d03_data_processed/d03_data_processed.csv', encoding= 'unicode_escape', index_col='Date')
#Display smart meter names and amount
smart_meter_names = df.columns[2:-4]
print("Selected clients: ", len(smart_meter_names))


# Make Datasets for the 33 clients and for 5 and 7 features
ds_dict = makeDatasetsForclientsAndfeatures(smart_meter_names, df)
print("Created dictionary with datasets")

# Create Windows 
windows_dict = createLocalDataWindows(smart_meter_names, INPUT_STEPS, OUT_STEPS, ds_dict)
print("Created Data windows")


#windows_dict[smart_meter_names][0-3] 
#    -> 0:window_F5_H12 , 1:window_F5_H24 , 2:window_F7_H12 , 3:window_F7_H24
forecasts_dict_LSTM_F5_H12 = {}
forecasts_dict_CNN_F5_H12 = {}
forecasts_dict_Transformer_F5_H12 = {}

for idx, client in enumerate(smart_meter_names):
    IPython.display.clear_output()
    print("-----------------------", idx, "/32")

            #LSTM
    model_evaluation_test = loadCompileEvaluateModel(
        path = cwd + f"/data/d05_models/local/{client}/Local_LSTM_F5_H12/Round100",
        window = windows_dict[client][0], 
        MAX_EPOCHS = MAX_EPOCHS
    )
    #Save
    forecasts_dict_LSTM_F5_H12[client] = {
        'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
        'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
    }
    print("Saved LSTM")
    
    #CNN
    model_evaluation_test = loadCompileEvaluateModel(
        path = cwd + f"/data/d05_models/local/{client}/Local_CNN_F5_H12/Round100",
        window = windows_dict[client][0], 
        MAX_EPOCHS = MAX_EPOCHS
    )
    #Save
    forecasts_dict_CNN_F5_H12[client] = {
        'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
        'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
    }    
    print("Saved CNN")
    
    #Transformer
    model_evaluation_test = loadCompileEvaluateModel(
        path = cwd + f"/data/d05_models/local/{client}/Local_Transformer_F5_H12/Round100",
        window = windows_dict[client][0], 
        MAX_EPOCHS = MAX_EPOCHS
    )
    #Save
    forecasts_dict_Transformer_F5_H12[client] = {
        'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
        'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
    }
    print("Saved Transformer")

#Inintialize result dictionary
final_dict = InititalizeResultDictionary(learning_style="Local")
final_dict['Local']['LSTM']['H12']['F5'] = forecasts_dict_LSTM_F5_H12
final_dict['Local']['CNN']['H12']['F5'] = forecasts_dict_CNN_F5_H12
final_dict['Local']['Transformer']['H12']['F5'] = forecasts_dict_Transformer_F5_H12

with open(cwd + '/results/Local_results_H12_F5.pkl', 'wb') as f:
    pickle.dump(final_dict, f)

print("Done - saved h12 f5")
#f5 h24

#windows_dict[smart_meter_names][0-3] 
#    -> 0:window_F5_H12 , 1:window_F5_H24 , 2:window_F7_H12 , 3:window_F7_H24
forecasts_dict_LSTM_F5_H24 = {}
forecasts_dict_CNN_F5_H24 = {}
forecasts_dict_Transformer_F5_H24 = {}

for idx, client in enumerate(smart_meter_names):
    IPython.display.clear_output()
    print("-----------------------", idx, "/32")

            #LSTM
    model_evaluation_test = loadCompileEvaluateModel(
        path = cwd + f"/data/d05_models/local/{client}/Local_LSTM_F5_H24/Round100",
        window = windows_dict[client][1], 
        MAX_EPOCHS = MAX_EPOCHS
    )
    #Save
    forecasts_dict_LSTM_F5_H24[client] = {
        'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
        'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
    }
    print("Saved LSTM")
    
    #CNN
    model_evaluation_test = loadCompileEvaluateModel(
        path = cwd + f"/data/d05_models/local/{client}/Local_CNN_F5_H24/Round100",
        window = windows_dict[client][1], 
        MAX_EPOCHS = MAX_EPOCHS
    )
    #Save
    forecasts_dict_CNN_F5_H24[client] = {
        'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
        'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
    }    
    print("Saved CNN")
    
    #Transformer
    model_evaluation_test = loadCompileEvaluateModel(
        path = cwd + f"/data/d05_models/local/{client}/Local_Transformer_F5_H24/Round100",
        window = windows_dict[client][1], 
        MAX_EPOCHS = MAX_EPOCHS
    )
    #Save
    forecasts_dict_Transformer_F5_H24[client] = {
        'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
        'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
    }
    print("Saved Transformer")

#Inintialize result dictionary
final_dict = InititalizeResultDictionary(learning_style="Local")
final_dict['Local']['LSTM']['H24']['F5'] = forecasts_dict_LSTM_F5_H24
final_dict['Local']['CNN']['H24']['F5'] = forecasts_dict_CNN_F5_H24
final_dict['Local']['Transformer']['H24']['F5'] = forecasts_dict_Transformer_F5_H24

with open(cwd + '/results/Local_results_H24_F5.pkl', 'wb') as f:
    pickle.dump(final_dict, f)

print("Done - saved h24 f5")