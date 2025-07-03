import os
import sys
import argparse
import logging
from enum import Enum

import numpy as np
from data import DataReader
from models.model_generator import modelGenerator
from basic_functions.address import *
from tensorflow import keras

import tensorflow as tf

# Configuration du GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) detected and VRAM set to crossover mode..")
    except RuntimeError as e:
        print(f"GPU configuration error : {e}")
else:
    print("⚠️ I also discovered the GPU. Training takes place on the CPU.")

__author__ = "Miguel Salinas <uo34525@uniovi.es>, Alejandro <uo265351@uniovi.es>"
__copyright__ = "Uniovi"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

WINDOW_CONCATENATED_DATA = "arr_0"
WINDOW_ALL_LABELS = "arr_1"

CONVOLUTIONAL_DATASET_FILE = "data_all.npz"
FEATURE_DATASET_FILE = "data_feature_all.npz"
LABEL_ENCODER_FILE = "label_encoder.pkl"

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'

class ML_Sensor(Enum):
    PI = 'thigh'
    M = 'wrist'
    C = 'hip'

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
        "-case-id",
        "--case-id",
        dest="case_id",
        required=True,
        help="Case unique identifier."
    )
    parser.add_argument(
        "-case-id-folder",
        "--case-id-folder",
        dest="case_id_folder",
        required=True,
        help="Choose the case id root folder."
    )        
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
        "-dataset-folder",
        "--dataset-folder",
        dest="dataset_folder",
        required=True,
        help="Choose the dataset root folder."
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
        '-add-sintetic-data',
        '--add-sintetic-data',
        dest='add_sintetic_data',
        type=bool,
        default=False,
        help="Add sintetic data for training"
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
        if model.value in [ML_Model.CAPTURE24.value, ML_Model.ESANN.value]:
            return True
        
    return False

def feature_model_selected(models):
    for model in models:
        if model.value in [ML_Model.RANDOM_FOREST.value, ML_Model.XGBOOST.value]:
            return True
        
    return False

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

    _logger.info("Trainer starts here")

    # create the output case id folder if not exist
    case_id_folder = os.path.join(args.case_id_folder, args.case_id)
    os.makedirs(case_id_folder, exist_ok=True)

    for ml_model in args.ml_models[0]:        
        modelID = ml_model.value

        if modelID == ML_Model.ESANN.value:
            dataset_file = os.path.join(case_id_folder, CONVOLUTIONAL_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)

            # IMUs muslo + muñeca
            data_tot = DataReader(modelID=modelID, p_train = args.training_percent / 100, file_path=dataset_file, label_encoder_path=label_encoder_file)
            params_ESANN = {"N_capas": 2}
            model_ESANN_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_ESANN, debug=False)
            Ruta_model_ESANN_data_tot = get_model_path(modelID)
            # Si ya existe el modelo, se carga el fichero .h5. En caso contrario, se entrenan y salvan los modelos.
            # 3 CNNs ESANN
            if os.path.isfile(Ruta_model_ESANN_data_tot):
                model_ESANN_data_tot.load(modelID, case_id_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_ESANN_data_tot.train()
                model_ESANN_data_tot.store(modelID, case_id_folder)
                
        elif modelID == ML_Model.CAPTURE24.value:
            dataset_file = os.path.join(case_id_folder, CONVOLUTIONAL_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)

            # IMUs muslo + muñeca
            data_tot = DataReader(modelID=modelID, p_train = args.training_percent / 100, file_path=dataset_file, label_encoder_path=label_encoder_file)
            params_CAPTURE24 = {"N_capas": 6}
            model_CAPTURE24_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_CAPTURE24, debug=False)
            Ruta_model_CAPTURE24_data_tot = get_model_path(modelID)
            if os.path.isfile(Ruta_model_CAPTURE24_data_tot):
                model_CAPTURE24_data_tot.load(modelID, case_id_folder)
            else:
                callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                model_CAPTURE24_data_tot.train()
                model_CAPTURE24_data_tot.store(modelID, case_id_folder)
         
        elif modelID == ML_Model.RANDOM_FOREST.value:
            dataset_file = os.path.join(case_id_folder, FEATURE_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)

            # IMUs muslo + muñeca
            data_tot = DataReader(modelID=modelID, p_train = args.training_percent / 100, file_path=dataset_file, label_encoder_path=label_encoder_file)
            params_RandomForest = {"n_estimators": 3000}
            model_RandomForest_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_RandomForest, debug=False)
            Ruta_model_RandomForest_data_tot = get_model_path(modelID)
            if os.path.isfile(Ruta_model_RandomForest_data_tot):
                model_RandomForest_data_tot.load(modelID, case_id_folder)
            else:
                model_RandomForest_data_tot.train()
                model_RandomForest_data_tot.store(modelID, case_id_folder)
         
        _logger.info("Script ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()            