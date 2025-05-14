from Models import *
from address import *
import pandas as pd
from Data import DataReader
import ast

# Model factory pattern
def modelGenerator(modelID:str, data, params:dict={}, verbose=False, debug=False):
    '''
    ARGUMENTS
        modelID (str)                       ID that indicates the model type
        data    (featExtraction object)     Data object needed to train
        params  (dict)                      the params that define the model 
    '''
    # data = data
    # modelID = modelID
    # params  = params

    if verbose:
        print("Building model")

    if not params and not debug:
        if verbose:
            print("loading best hyperparameters")
        params_path  = get_param_path(modelID)
        df_params    = pd.read_csv(params_path,index_col=0)
        params       = ast.literal_eval(df_params.loc[data.dataID,'params'])[0]

    
    # Aquí se llamarían a todos los modelos según su ID
    # 3 CNNs ESANN
    if modelID == "SiMuRModel_ESANN_data_tot":
        model = SiMuRModel_ESANN(data, params)
    elif modelID == "SiMuRModel_ESANN_data_thigh":
        model = SiMuRModel_ESANN(data, params)
    elif modelID == "SiMuRModel_ESANN_data_wrist":
        model = SiMuRModel_ESANN(data, params)
    
    # 3 CNNs CAPTURE-24
    elif modelID == "SiMuRModel_CAPTURE24_data_tot":
        model = SiMuRModel_CAPTURE24(data, params)
    elif modelID == "SiMuRModel_CAPTURE24_data_thigh":
        model = SiMuRModel_CAPTURE24(data, params)
    elif modelID == "SiMuRModel_CAPTURE24_data_wrist":
        model = SiMuRModel_CAPTURE24(data, params)
    
    # 3 Random Forests
    elif modelID == "SiMuRModel_RandomForest_data_tot":
        model = SiMuRModel_RandomForest(data, params)
    elif modelID == "SiMuRModel_RandomForest_data_thigh":
        model = SiMuRModel_RandomForest(data, params)
    elif modelID == "SiMuRModel_RandomForest_data_wrist":
        model = SiMuRModel_RandomForest(data, params)
    else:
        model = None
        raise Exception("Model not implemented")
    return model
    

# Unit testing
if __name__ == "__main__": 
    
    # Test time models
    # modelID = "SiMuRModel_ESANN_data_tot"
    modelID = "SiMuRModel_CAPTURE24_data_tot"
    # params = {"N_capas":3}
    params = {"N_capas":6}
    data = DataReader(p_train=0.7, dataset='data_tot')
    
    model = modelGenerator(modelID=modelID, data=data, params=params, debug=False)
    model.train()