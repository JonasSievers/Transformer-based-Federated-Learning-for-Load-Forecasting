# pandas for reading and analyzing data
import pandas as pd
# os to find path of files 
import os
# tensorflow as machine learning library
import tensorflow as tf
# helper functions
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
cwd = os.path.normpath(os.getcwd())
sys.path.insert(1, cwd + "/src/d00_utils") 
from model_helper_functions import *

def loadCompileEvaluateModel(path, window, MAX_EPOCHS):
    """
    load model, compile and evaluate on test window  
    
    :param: model path and window
    """           
    model = tf.keras.models.load_model(path, compile=False)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), 
            tf.keras.metrics.MeanAbsolutePercentageError(),
            tf.keras.metrics.MeanAbsoluteError(),
        ]
    )
    
    #fit local model with client's data
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min')
    model.fit(
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
    
    model_evaluation_test = model.evaluate(window.test)
    return model_evaluation_test