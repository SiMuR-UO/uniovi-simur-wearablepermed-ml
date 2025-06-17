import os
import sys
import argparse
import logging
from enum import Enum

import numpy as np
from Data import DataReader
from model_generator import modelGenerator
from address import *
import keras

__author__ = "Miguel Salinas <uo34525@uniovi.es>, Alejandro <uo265351@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'

class ML_Sensor(Enum):
    THIGH = 'thigh'
    WRIST = 'wrist'
    TOTAL = 'tot'

def parse_ml_model(value):
    try:
        """Parse a comma-separated list of CML Models lor values into a list of ML_Sensor enums."""
        values = [v.strip() for v in value.split(',') if v.strip()]
        result = []
        invalid = []
        for v in values:
            try:
                result.append(ML_Model(v))
            except ValueError:
                invalid.append(v)
        if invalid:
            valid = ', '.join(c.value for c in ML_Model)
            raise argparse.ArgumentTypeError(
                f"Invalid color(s): {', '.join(invalid)}. "
                f"Choose from: {valid}"
            )
        return result
    except ValueError:
        valid = ', '.join(ml_model.value for ml_model in ML_Model)
        raise argparse.ArgumentTypeError(f"Invalid ML Model '{value}'. Choose from: {valid}")
    
def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Machine Learning Model Trainer")
    parser.add_argument(
        "-ml-models",
        "--ml-models",
        type=parse_ml_model,
        nargs='+',
        dest="ml_models",        
        required=True,
        help=f"Available ML models: {[c.value for c in ML_Model]}."
    )
    parser.add_argument(
        "-ml-sensor",
        "--ml-sensor",
        type=str,
        choices=[ml_sensor.value for ml_sensor in ML_Sensor],
        dest="ml_sensor",
        required=True,
        help=f"Choose a ML sensor: {[c.value for c in ML_Sensor]}."
    )
    parser.add_argument(
        "-dataset-folder",
        "--dataset-folder",
        dest="dataset_folder",
        required=True,
        help="Choose the dataset root folder."
    )
    parser.add_argument(
        "-participants-file",
        "--participants-file",
        type=argparse.FileType("r"),
        required=True,
        help="Choose the dataset participant text file"
    )        
    parser.add_argument(
        '-training-percent',
        '--training-percent',
        dest='training_percent',
        type=int,
        default=70,
        help="Training percent"
    )    
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO.",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG.",
        action="store_const",
        const=logging.DEBUG,
    )    
    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def convolution_model_selected(models):
    for model in models:
        if model.value in [ML_Model.CAPTURE24, ML_Model.ESANN]:
            return True
        
    return False

def feature_model_selected(models):
    for model in models:
        if model.value in [ML_Model.RANDOM_FOREST, ML_Model.XGBOOST]:
            return True
        
    return False

def combine_participant_dataset(dataset_folder, participants, models):
    for participant in participants:
        participant_folder = os.path.join(dataset_folder, participant)
        participant_files = [f for f in os.listdir(participant_folder) if os.path.isfile(os.path.join(participant_folder, f)) and ".npz" in f]
    
        participant_dataset = []
        participant_label_dataset = []
        
        participant_feature_dataset = []
        participant_label_feature_dataset = []
        
        for participant_file in participant_files:
            if "all" not in participant_file and "features" not in participant_file and convolution_model_selected(models):
                participant_sensor_dataset = np.load(participant_file)
                
                participant_dataset.append(participant_sensor_dataset['concatenated_data'], axis=0)
                participant_label_dataset.append(participant_sensor_dataset['all_labels'], axis=0)
                
                participant_sensor_file = os.path.join(participant_folder, 'data_' + participant + "_all.npz")
                np.savez(participant_sensor_file, participant_dataset, participant_label_dataset)
                
            if "all" not in participant_file and "features" in participant_file and feature_model_selected(models):
                participant_sensor_feature_dataset = np.load(participant_file)
                
                participant_feature_dataset.append(participant_sensor_feature_dataset['concatenated_data'], axis=0)
                participant_label_feature_dataset.append(participant_sensor_feature_dataset['all_labels'], axis=0)
                
                participant_sensor_feature_file = os.path.join(participant_folder, 'data_' + participant + "_feature_all.npz")
                np.savez(participant_sensor_feature_file, participant_feature_dataset, participant_label_feature_dataset)                

def combine_datasets(dataset_folder, participants, models, ml_sensor):
    dataset = []
    dataset_feature = []
    
    for participant in participants:
        participant_folder = os.path.join(dataset_folder, participant)
        participant_files = [f for f in os.listdir(participant_folder) if os.path.isfile(os.path.join(participant_folder, f)) and ".npz" in f]        

        for participant_file in participant_files:
            if ml_sensor == ML_Sensor.WRIST.value:
                if (convolution_model_selected(models) and "_M.npz" in participant_file):
                    participant_sensor_dataset = np.load(participant_file)
                    dataset.append(participant_sensor_dataset, axis=0)
                
                if (feature_model_selected(models) and "_M_features.npz" in participant_file):    
                    participant_sensor_feature_dataset = np.load(participant_file)
                    dataset_feature.append(participant_sensor_feature_dataset, axis=0)
            elif ml_sensor == ML_Sensor.THIGH.value:
                if (convolution_model_selected(models) and "_PI.npz" in participant_file):
                    participant_sensor_dataset = np.load(participant_file)
                    dataset.append(participant_sensor_dataset, axis=0)
                
                if (feature_model_selected(models) and "_PI_features.npz" in participant_file):    
                    participant_sensor_feature_dataset = np.load(participant_file)
                    dataset_feature.append(participant_sensor_feature_dataset, axis=0)                
            else:
                if (convolution_model_selected(models) and "_all.npz" in participant_file):
                    participant_sensor_dataset = np.load(participant_file)
                    dataset.append(participant_sensor_dataset, axis=0)
                
                if (feature_model_selected(models) and "_all_features.npz" in participant_file):    
                    participant_sensor_feature_dataset = np.load(participant_file)
                    dataset_feature.append(participant_sensor_feature_dataset, axis=0)
    
    if dataset_feature.size == 0:
        if ml_sensor == ML_Sensor.WRIST.value:
            dataset_file = os.path.join(dataset_folder, "dataset_M.npz")                
        elif ml_sensor == ML_Sensor.THIGH.value:
            dataset_file = os.path.join(dataset_folder, "dataset_PI.npz")
        else:
            dataset_file = os.path.join(dataset_folder, "dataset_all.npz")
            
        np.savez(dataset_file, dataset)      
    else:
        if ml_sensor == ML_Sensor.WRIST.value:
            dataset_file = os.path.join(dataset_folder, "dataset_M_feature.npz")                
        elif ml_sensor == ML_Sensor.THIGH.value:
            dataset_file = os.path.join(dataset_folder, "dataset_PI_feature.npz")
        else:
            dataset_file = os.path.join(dataset_folder, "dataset_all_feature.npz")
            
        np.savez(dataset_file, dataset_feature)                               

def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.debug("Starting training ...")

    participants = []
    for line in args.participants_file:
        participants = participants + line.strip().split(',')

    # Agregacion de acelerometria para caso all
    if args.ml_sensor == "all":
        combine_participant_dataset(args.dataset_folder, participants, args.ml_models[0])
    
    combine_datasets(args.dataset_folder, participants, args.ml_models[0], args.ml_sensor.value)
     
    for ml_model in args.ml_models[0]:        
        modelID = 'modelID_' + ml_model.value + '_data_' + args.ml_sensor

        if ml_model.value == ML_Model.ESANN.value and args.ml_sensor == ML_Sensor.TOTAL.value:
            # IMUs muslo + muñeca
            data_tot = DataReader(p_train = args.training_percent / 100, dataset='data_tot')
            params_ESANN = {"N_capas": 2}
            model_ESANN_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_ESANN, debug=False)
            Ruta_model_ESANN_data_tot = get_model_path(modelID)
            # Si ya existe el modelo, se carga el fichero .h5. En caso contrario, se entrenan y salvan los modelos.
            # 3 CNNs ESANN
            if os.path.isfile(Ruta_model_ESANN_data_tot):
                model_ESANN_data_tot.load(modelID, args.dataset_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_ESANN_data_tot.train()
                model_ESANN_data_tot.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.ESANN.value and args.ml_sensor == ML_Sensor.THIGH.value:
            # IMU muslo
            data_thigh = DataReader(p_train = args.training_percent / 100, dataset='data_thigh')
            params_ESANN = {"N_capas": 2}
            model_ESANN_data_thigh = modelGenerator(modelID=modelID, data=data_thigh, params=params_ESANN, debug=False)
            Ruta_model_ESANN_data_thigh = get_model_path(modelID)      
            if os.path.isfile(Ruta_model_ESANN_data_thigh):
                model_ESANN_data_thigh.load(modelID, args.dataset_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_ESANN_data_thigh.train()
                model_ESANN_data_thigh.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.ESANN.value and args.ml_sensor == ML_Sensor.WRIST.value:
            # IMU muñeca
            data_wrist = DataReader(p_train = args.training_percent / 100, dataset='data_wrist')
            params_ESANN = {"N_capas": 2}
            model_ESANN_data_wrist = modelGenerator(modelID=modelID, data=data_wrist, params=params_ESANN, debug=False)
            Ruta_model_ESANN_data_wrist = get_model_path(modelID)    
            if os.path.isfile(Ruta_model_ESANN_data_wrist):
                model_ESANN_data_wrist.load(modelID, args.dataset_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_ESANN_data_wrist.train()
                model_ESANN_data_wrist.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.CAPTURE24.value and args.ml_sensor == ML_Sensor.TOTAL.value:
            # IMUs muslo + muñeca
            data_tot = DataReader(p_train = args.training_percent / 100, dataset='data_tot')
            params_CAPTURE24 = {"N_capas": 6}
            model_CAPTURE24_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_CAPTURE24, debug=False)
            Ruta_model_CAPTURE24_data_tot = get_model_path(modelID)
            if os.path.isfile(Ruta_model_CAPTURE24_data_tot):
                model_CAPTURE24_data_tot.load(modelID, args.dataset_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_CAPTURE24_data_tot.train()
                model_CAPTURE24_data_tot.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.CAPTURE24.value and args.ml_sensor == ML_Sensor.THIGH.value:
            # IMU muslo
            data_thigh = DataReader(p_train = args.training_percent / 100, dataset='data_thigh')
            params_CAPTURE24 = {"N_capas": 6}
            model_CAPTURE24_data_thigh = modelGenerator(modelID=modelID, data=data_thigh, params=params_CAPTURE24, debug=False)
            Ruta_model_CAPTURE24_data_thigh = get_model_path(modelID)
            if os.path.isfile(Ruta_model_CAPTURE24_data_thigh):
                model_CAPTURE24_data_thigh.load(modelID, args.dataset_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_CAPTURE24_data_thigh.train()
                model_CAPTURE24_data_thigh.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.CAPTURE24.value and args.ml_sensor == ML_Sensor.WRIST.value:
            # IMU muñeca
            data_wrist = DataReader(p_train = args.training_percent / 100, dataset='data_wrist')
            params_CAPTURE24 = {"N_capas": 6}
            model_CAPTURE24_data_wrist = modelGenerator(modelID=modelID, data=data_wrist, params=params_CAPTURE24, debug=False)
            Ruta_model_CAPTURE24_data_wrist = get_model_path(modelID)  
            if os.path.isfile(Ruta_model_CAPTURE24_data_wrist):
                model_CAPTURE24_data_wrist.load(modelID, args.dataset_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_CAPTURE24_data_wrist.train()
                model_CAPTURE24_data_wrist.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.RANDOM_FOREST.value and args.ml_sensor == ML_Sensor.TOTAL.value:
            # IMUs muslo + muñeca
            data_tot = DataReader(p_train = args.training_percent / 100, dataset='data_tot')
            params_RandomForest = {"n_estimators": 3000}
            model_RandomForest_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_RandomForest, debug=False)
            Ruta_model_RandomForest_data_tot = get_model_path(modelID)
            if os.path.isfile(Ruta_model_RandomForest_data_tot):
                model_RandomForest_data_tot.load(modelID, args.dataset_folder)
            else:
                model_RandomForest_data_tot.train()
                model_RandomForest_data_tot.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.RANDOM_FOREST.value and args.ml_sensor == ML_Sensor.THIGH.value:            
            # IMU muslo
            data_thigh = DataReader(p_train = args.training_percent / 100, dataset='data_thigh')
            params_RandomForest = {"n_estimators": 3000}
            model_RandomForest_data_thigh = modelGenerator(modelID=modelID, data=data_thigh, params=params_RandomForest, debug=False)
            Ruta_model_RandomForest_data_thigh = get_model_path(modelID)
            if os.path.isfile(Ruta_model_RandomForest_data_thigh):
                model_RandomForest_data_thigh.load(modelID, args.dataset_folder)
            else:
                model_RandomForest_data_thigh.train()
                model_RandomForest_data_thigh.store(modelID, args.dataset_folder)
                
        elif ml_model.value == ML_Model.RANDOM_FOREST.value and args.ml_sensor == ML_Sensor.WRIST.value:                        
            # IMU muñeca
            data_wrist = DataReader(p_train = args.training_percent / 100, dataset='data_wrist')
            params_RandomForest = {"n_estimators": 3000}
            model_RandomForest_data_wrist = modelGenerator(modelID=modelID, data=data_wrist, params=params_RandomForest, debug=False)
            Ruta_model_RandomForest_data_wrist = get_model_path(modelID)
            if os.path.isfile(Ruta_model_RandomForest_data_wrist):
                model_RandomForest_data_wrist.load(modelID, args.dataset_folder)
            else:
                model_RandomForest_data_wrist.train()
                model_RandomForest_data_wrist.store(modelID, args.dataset_folder)

        _logger.info("Script ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()            