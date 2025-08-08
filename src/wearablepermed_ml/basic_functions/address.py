import os

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Data/'
results_grid_search = str(path_here)+'/Results/Params/'
path_results_metrics = str(path_here)+'/Results/'


def get_param_path(modelID):
    return os.path.join(results_grid_search,modelID+'.csv')

def get_model_path(modelID, args):
    path_models = os.path.join(args.case_id_folder, args.case_id)
    if modelID == 'SiMuRModel_RandomForest_data_tot' or modelID == 'SiMuRModel_RandomForest_data_thigh' or modelID == 'SiMuRModel_RandomForest_data_wrist':
        return os.path.join(path_models,modelID+'.pkl')
    else:
        return os.path.join(path_models,modelID+'.weights.h5')