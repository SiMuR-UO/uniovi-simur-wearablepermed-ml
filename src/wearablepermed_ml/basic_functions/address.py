import os

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Data/'
results_grid_search = str(path_here)+'/Results/Params/'
path_results_metrics = str(path_here)+'/Results/'
path_models = str(path_here)+'/Models/'

def get_param_path(modelID):
    return os.path.join(results_grid_search,modelID+'.csv')

def get_model_path(modelID):
    if modelID == 'SiMuRModel_RandomForest_data_tot' or modelID == 'SiMuRModel_RandomForest_data_thigh' or modelID == 'SiMuRModel_RandomForest_data_wrist':
        return os.path.join(path_models,modelID+'.pkl')
    else:
        return os.path.join(path_models,modelID+'.h5')