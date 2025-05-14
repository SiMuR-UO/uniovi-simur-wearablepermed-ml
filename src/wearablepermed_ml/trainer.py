from Data import DataReader
from model_generator import modelGenerator
from address import *
import keras

if __name__ == "__main__":  
    RutaModelos = "/home/simur/directorio_SiMuR_MachineLearning/WPM_24_02_2025/Models/"
    
    # Test time models
    # 3 CNNs propuestas para ESANN
    modelID_ESANN_data_tot = "SiMuRModel_ESANN_data_tot"
    modelID_ESANN_data_thigh = "SiMuRModel_ESANN_data_thigh"
    modelID_ESANN_data_wrist = "SiMuRModel_ESANN_data_wrist"
    
    # 3 CNNs diseñadas según CAPTURE-24
    modelID_CAPTURE24_data_tot = "SiMuRModel_CAPTURE24_data_tot"
    modelID_CAPTURE24_data_thigh = "SiMuRModel_CAPTURE24_data_thigh"
    modelID_CAPTURE24_data_wrist = "SiMuRModel_CAPTURE24_data_wrist"
    
    # 3 Random Forests
    modelID_RandomForest_data_tot = "SiMuRModel_RandomForest_data_tot"
    modelID_RandomForest_data_thigh = "SiMuRModel_RandomForest_data_thigh"
    modelID_RandomForest_data_wrist = "SiMuRModel_RandomForest_data_wrist"
    
    
    # Comentar y desomentar las siguientes líneas para entrenar uno u otro modelo de clasificación:
    modelID = modelID_ESANN_data_tot
    # modelID = modelID_ESANN_data_thigh
    # modelID = modelID_ESANN_data_wrist
    
    # modelID = modelID_CAPTURE24_data_tot
    # modelID = modelID_CAPTURE24_data_thigh
    # modelID = modelID_CAPTURE24_data_wrist
    
    # modelID = modelID_RandomForest_data_tot
    # modelID = modelID_RandomForest_data_thigh
    # modelID = modelID_RandomForest_data_wrist
    

    if modelID == modelID_ESANN_data_tot:
        # IMUs muslo + muñeca
        data_tot = DataReader(p_train = 0.7, dataset='data_tot')
        params_ESANN = {"N_capas":2}
        model_ESANN_data_tot = modelGenerator(modelID=modelID_ESANN_data_tot, data=data_tot, params=params_ESANN, debug=False)
        Ruta_model_ESANN_data_tot = get_model_path(modelID_ESANN_data_tot)
        # Si ya existe el modelo, se carga el fichero .h5. En caso contrario, se entrenan y salvan los modelos.
        # 3 CNNs ESANN
        if os.path.isfile(Ruta_model_ESANN_data_tot):
            model_ESANN_data_tot.load(modelID_ESANN_data_tot, RutaModelos)
        else:
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            model_ESANN_data_tot.train()
            model_ESANN_data_tot.store(modelID_ESANN_data_tot, RutaModelos)
            
    elif modelID == modelID_ESANN_data_thigh:
        # IMU muslo
        data_thigh = DataReader(p_train = 0.7, dataset='data_thigh')
        params_ESANN = {"N_capas":2}
        model_ESANN_data_thigh = modelGenerator(modelID=modelID_ESANN_data_thigh, data=data_thigh, params=params_ESANN, debug=False)
        Ruta_model_ESANN_data_thigh = get_model_path(modelID_ESANN_data_thigh)      
        if os.path.isfile(Ruta_model_ESANN_data_thigh):
            model_ESANN_data_thigh.load(modelID_ESANN_data_thigh, RutaModelos)
        else:
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            model_ESANN_data_thigh.train()
            model_ESANN_data_thigh.store(modelID_ESANN_data_thigh, RutaModelos)
            
    elif modelID == modelID_ESANN_data_wrist:
        # IMU muñeca
        data_wrist = DataReader(p_train = 0.7, dataset='data_wrist')
        params_ESANN = {"N_capas":2}
        model_ESANN_data_wrist = modelGenerator(modelID=modelID_ESANN_data_wrist, data=data_wrist, params=params_ESANN, debug=False)
        Ruta_model_ESANN_data_wrist = get_model_path(modelID_ESANN_data_wrist)    
        if os.path.isfile(Ruta_model_ESANN_data_wrist):
            model_ESANN_data_wrist.load(modelID_ESANN_data_wrist, RutaModelos)
        else:
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            model_ESANN_data_wrist.train()
            model_ESANN_data_wrist.store(modelID_ESANN_data_wrist, RutaModelos)
            
    elif modelID == modelID_CAPTURE24_data_tot:
        # IMUs muslo + muñeca
        data_tot = DataReader(p_train = 0.7, dataset='data_tot')
        params_CAPTURE24 = {"N_capas":6}
        model_CAPTURE24_data_tot = modelGenerator(modelID=modelID_CAPTURE24_data_tot, data=data_tot, params=params_CAPTURE24, debug=False)
        Ruta_model_CAPTURE24_data_tot = get_model_path(modelID_CAPTURE24_data_tot)
        if os.path.isfile(Ruta_model_CAPTURE24_data_tot):
            model_CAPTURE24_data_tot.load(modelID_CAPTURE24_data_tot, RutaModelos)
        else:
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            model_CAPTURE24_data_tot.train()
            model_CAPTURE24_data_tot.store(modelID_CAPTURE24_data_tot, RutaModelos)
            
    elif modelID == modelID_CAPTURE24_data_thigh:
        # IMU muslo
        data_thigh = DataReader(p_train = 0.7, dataset='data_thigh')
        params_CAPTURE24 = {"N_capas":6}
        model_CAPTURE24_data_thigh = modelGenerator(modelID=modelID_CAPTURE24_data_thigh, data=data_thigh, params=params_CAPTURE24, debug=False)
        Ruta_model_CAPTURE24_data_thigh = get_model_path(modelID_CAPTURE24_data_thigh)
        if os.path.isfile(Ruta_model_CAPTURE24_data_thigh):
            model_CAPTURE24_data_thigh.load(modelID_CAPTURE24_data_thigh, RutaModelos)
        else:
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            model_CAPTURE24_data_thigh.train()
            model_CAPTURE24_data_thigh.store(modelID_CAPTURE24_data_thigh, RutaModelos)
            
    elif modelID == modelID_CAPTURE24_data_wrist:
        # IMU muñeca
        data_wrist = DataReader(p_train = 0.7, dataset='data_wrist')
        params_CAPTURE24 = {"N_capas":6}
        model_CAPTURE24_data_wrist = modelGenerator(modelID=modelID_CAPTURE24_data_wrist, data=data_wrist, params=params_CAPTURE24, debug=False)
        Ruta_model_CAPTURE24_data_wrist = get_model_path(modelID_CAPTURE24_data_wrist)  
        if os.path.isfile(Ruta_model_CAPTURE24_data_wrist):
            model_CAPTURE24_data_wrist.load(modelID_CAPTURE24_data_wrist, RutaModelos)
        else:
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            model_CAPTURE24_data_wrist.train()
            model_CAPTURE24_data_wrist.store(modelID_CAPTURE24_data_wrist, RutaModelos)
            
    elif modelID == modelID_RandomForest_data_tot:
        # IMUs muslo + muñeca
        data_tot = DataReader(p_train = 0.7, dataset='data_tot')
        params_RandomForest = {"n_estimators": 3000}
        model_RandomForest_data_tot = modelGenerator(modelID=modelID_RandomForest_data_tot, data=data_tot, params=params_RandomForest, debug=False)
        Ruta_model_RandomForest_data_tot = get_model_path(modelID_RandomForest_data_tot)
        if os.path.isfile(Ruta_model_RandomForest_data_tot):
            model_RandomForest_data_tot.load(modelID_RandomForest_data_tot, RutaModelos)
        else:
            model_RandomForest_data_tot.train()
            model_RandomForest_data_tot.store(modelID_RandomForest_data_tot, RutaModelos)
            
    elif modelID == modelID_RandomForest_data_thigh:
        # IMU muslo
        data_thigh = DataReader(p_train = 0.7, dataset='data_thigh')
        params_RandomForest = {"n_estimators": 3000}
        model_RandomForest_data_thigh = modelGenerator(modelID=modelID_RandomForest_data_thigh, data=data_thigh, params=params_RandomForest, debug=False)
        Ruta_model_RandomForest_data_thigh = get_model_path(modelID_RandomForest_data_thigh)
        if os.path.isfile(Ruta_model_RandomForest_data_thigh):
            model_RandomForest_data_thigh.load(modelID_RandomForest_data_thigh, RutaModelos)
        else:
            model_RandomForest_data_thigh.train()
            model_RandomForest_data_thigh.store(modelID_RandomForest_data_thigh, RutaModelos)
            
    elif modelID == modelID_RandomForest_data_wrist:
        # IMU muñeca
        data_wrist = DataReader(p_train = 0.7, dataset='data_wrist')
        params_RandomForest = {"n_estimators": 3000}
        model_RandomForest_data_wrist = modelGenerator(modelID=modelID_RandomForest_data_wrist, data=data_wrist, params=params_RandomForest, debug=False)
        Ruta_model_RandomForest_data_wrist = get_model_path(modelID_RandomForest_data_wrist)
        if os.path.isfile(Ruta_model_RandomForest_data_wrist):
            model_RandomForest_data_wrist.load(modelID_RandomForest_data_wrist, RutaModelos)
        else:
            model_RandomForest_data_wrist.train()
            model_RandomForest_data_wrist.store(modelID_RandomForest_data_wrist, RutaModelos)
            