# Imports

#Pandas: Reading and analyzing data
import pandas as pd
#Use Dates in Datetime Format
from datetime import timedelta
#Tensorflow
import tensorflow as tf
#Keras: Open-Source deep-learning library 
import os
#Clear output after training
import IPython.display
# pickle to save dictionary in file
import pickle 
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
cwd = os.path.normpath(os.getcwd())
sys.path.insert(1, cwd + "/src/d00_utils") 
from central_helper_functions import *
from evaluation_helper_functions import *
from model_helper_functions import *
from windowgenerator import *

# Hyperparameter
OUT_STEPS = [12, 24] #Next 12 or 24 hours
INPUT_STEPS = 24
#Training epochs
MAX_EPOCHS = 100



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

# Create Windows 
window_F5_H12, window_F5_H24, window_F7_H12, window_F7_H24 = createCentralDataWindows(INPUT_STEPS, OUT_STEPS, train_df_F7, val_df_F7, test_df_F7, train_df_F5, val_df_F5, test_df_F5)
print("Created Data windows")

# Local Learning
# Set random seed for as reproducible results as possible
tf.random.set_seed(42)


#h12 f5
#windows_dict[smart_meter_names][0-3] 
#    -> 0:window_F5_H12 , 1:window_F5_H24 , 2:window_F7_H12 , 3:window_F7_H24
forecasts_dict_LSTM_F7_H12 = {}
forecasts_dict_CNN_F7_H12 = {}
forecasts_dict_Transformer_F7_H12 = {}


#LSTM
model_evaluation_test = loadCompileEvaluateModel(
    path = cwd + f"/data/d05_models/central/Central_LSTM_F7_H12/Round100",
    window = window_F7_H12, 
    MAX_EPOCHS = MAX_EPOCHS
)
#Save
forecasts_dict_LSTM_F7_H12 = {
    'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
    'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
}
print("Saved LSTM")


#CNN
model_evaluation_test = loadCompileEvaluateModel(
    path = cwd + f"/data/d05_models/central/Central_CNN_F7_H12/Round100",
    window = window_F7_H12, 
    MAX_EPOCHS = MAX_EPOCHS
)
#Save
forecasts_dict_CNN_F7_H12 = {
    'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
    'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
}
print("Saved CNN")


#Transformer
model_evaluation_test = loadCompileEvaluateModel(
    path = cwd + f"/data/d05_models/central/Central_Transformer_F7_H12/Round100",
    window = window_F7_H12, 
    MAX_EPOCHS = MAX_EPOCHS
)
#Save
forecasts_dict_Transformer_F7_H12 = {
    'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
    'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
}
print("Saved Transformer")

#Inintialize result dictionary
final_dict = InititalizeResultDictionary(learning_style="Central")
final_dict['Central']['LSTM']['H12']['F7'] = forecasts_dict_LSTM_F7_H12
final_dict['Central']['CNN']['H12']['F7'] = forecasts_dict_CNN_F7_H12
final_dict['Central']['Transformer']['H12']['F7'] = forecasts_dict_Transformer_F7_H12

with open(cwd + '/results/Central_results_H12_F7.pkl', 'wb') as f:
    pickle.dump(final_dict, f)

print("Done - saved h12 f7")



#h24 f7
#windows_dict[smart_meter_names][0-3] 
#    -> 0:window_F5_H12 , 1:window_F5_H24 , 2:window_F7_H12 , 3:window_F7_H24
forecasts_dict_LSTM_F7_H24 = {}
forecasts_dict_CNN_F7_H24 = {}
forecasts_dict_Transformer_F7_H24 = {}


#LSTM
model_evaluation_test = loadCompileEvaluateModel(
    path = cwd + f"/data/d05_models/central/Central_LSTM_F7_H24/Round100",
    window = window_F7_H24, 
    MAX_EPOCHS = MAX_EPOCHS
)
#Save
forecasts_dict_LSTM_F7_H24 = {
    'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
    'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
}
print("Saved LSTM")


#CNN
model_evaluation_test = loadCompileEvaluateModel(
    path = cwd + f"/data/d05_models/central/Central_CNN_F7_H24/Round100",
    window = window_F7_H24, 
    MAX_EPOCHS = MAX_EPOCHS
)
#Save
forecasts_dict_CNN_F7_H24 = {
    'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
    'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
}
print("Saved CNN")


#Transformer
model_evaluation_test = loadCompileEvaluateModel(
    path = cwd + f"/data/d05_models/central/Central_Transformer_F7_H24/Round100",
    window = window_F7_H24, 
    MAX_EPOCHS = MAX_EPOCHS
)
#Save
forecasts_dict_Transformer_F7_H24 = {
    'MSE':model_evaluation_test[0], 'RMSE':model_evaluation_test[1], 'MAPE':model_evaluation_test[2],
    'MAE':model_evaluation_test[3], 'Time':((timetaken.logs[-1][1]) / (timetaken.logs[-1][0]+1)) 
}
print("Saved Transformer")

#Inintialize result dictionary
final_dict = InititalizeResultDictionary(learning_style="Central")
final_dict['Central']['LSTM']['H24']['F7'] = forecasts_dict_LSTM_F7_H24
final_dict['Central']['CNN']['H24']['F7'] = forecasts_dict_CNN_F7_H24
final_dict['Central']['Transformer']['H24']['F7'] = forecasts_dict_Transformer_F7_H24

with open(cwd + '/results/Central_results_H24_F7.pkl', 'wb') as f:
    pickle.dump(final_dict, f)

print("Done - saved h24 f7")