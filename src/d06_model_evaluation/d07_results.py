#Create Folder for modelling checkpoint
import os

# pickle to save dictionary in file
import pickle 
cwd = os.path.normpath(os.getcwd())

with open(cwd + '/results/Federated_results_H12_F7.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
print(loaded_dict)