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

import keras_tuner 
import json

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

CONVOLUTIONAL_DATASET_FILE = "data_all.npz"
FEATURE_DATASET_FILE = "data_feature_all.npz"
LABEL_ENCODER_FILE = "label_encoder.pkl"
CONFIG_FILE = "config.cfg"

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
        "-create-superclasses",
        "--create-superclasses",
        dest="create_superclasses",
        action='store_true',
        help="Create activity superclasses (true/false)."
    )          
    parser.add_argument(
        '-training-percent',
        '--training-percent',
        dest='training_percent',
        type=int,
        default=70,
        required=True,
        help="Training percent"
    )
    parser.add_argument(
        '-validation-percent',
        '--validation-percent',
        dest='validation_percent',
        type=int,
        default=20,
        help="Validation percent"
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

# ------------------------------------------------------------------------
# if searching optimal hyperparameter:
# Obtener modelo base para la optimización de los hiperparámetros:
def add_optimized_hyperparameters_CNN(hp, model, data):
    hiperparametros_busqueda ={ 
                    "N_capas":            hp.Int("N_capas", min_value=2, max_value=7),                     # número de capas ocultas de la red
                    "optimizador":        hp.Choice("optimizer", ["adam", "rmsprop", "SGD"]),              # optimizador a utilizar durante el entrenamiento
                    "funcion_activacion": hp.Choice("activation", ["relu", "tanh", "sigmoid"]),            # función de activación asociada a las neuronas de las capas ocultas
                    "tamanho_minilote":   hp.Int("miniBatchSize", min_value=10, max_value=30, step=7),     # tamaño del mini-lote de entrenamiento
                    "numero_filtros":     hp.Int("numFilters", min_value=12, max_value=30, step=4),        # número de filtros utilizados en las capas ocultas de la red
                    "tamanho_filtro":     hp.Int("filterSize", min_value=3, max_value=15, step=2),         # tamaño de los filtros de las capas ocultas
                    "tasa_aprendizaje":   hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")   # learning-rate empleado durante el entrenamiento
                    }
    # Construir el modelo con la selección de hiperparámetros
    class_simur_model = model.get_model_Obj()                      # Obtenemos la clase SiMuRModel a partir de model_data_tot
    modelObj = class_simur_model(data, hiperparametros_busqueda)   # Instanciamos la clase class_simur_model

    return modelObj.model                                          # Devuelve el objeto de la clase class_simur_model
    
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
            config_file = os.path.join(case_id_folder, CONFIG_FILE)
            data_tot = DataReader(modelID=modelID, create_superclasses=args.create_superclasses, p_train = args.training_percent, p_validation = args.validation_percent, 
                                  file_path=dataset_file, label_encoder_path=label_encoder_file, config_path = config_file)
            params_ESANN = {"N_capas": 2}
            model_ESANN_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_ESANN, debug=False)
            Ruta_model_ESANN_data_tot = get_model_path(modelID, args)
            if os.path.isfile(Ruta_model_ESANN_data_tot):      # Si ya existe el modelo, se carga el fichero .h5. En caso contrario, se entrenan y salvan los modelos.
                model_ESANN_data_tot.load(modelID, case_id_folder)
            else:
                hp_json_path = os.path.join(case_id_folder, "best_hyperparameters.json")
                if os.path.isfile(hp_json_path):
                    with open(hp_json_path, "r") as f:
                        best_hp_values = json.load(f)   
                    hp = keras_tuner.HyperParameters()
                    for param, value in best_hp_values.items():
                        hp.values[param] = value 
                    params_ESANN = hp.values
                    model_ESANN_data_tot = modelGenerator(modelID=modelID, data=data_tot, params=params_ESANN, debug=False)
                    model_ESANN_data_tot.train()
                    model_ESANN_data_tot.store(modelID, case_id_folder)
                else:
                    tuner = keras_tuner.Hyperband(  # Crear el obj de búsqueda keras_tuner.Hyperband(ASHA algorithm)
                        hypermodel           = lambda hp: add_optimized_hyperparameters_CNN(hp=hp, model=model_ESANN_data_tot, data=data_tot),                         # modelo construido con los hiperparámetros seleccionados en cada iteración 
                        objective            = "val_accuracy",                                        # función objetivo a optimizar
                        max_epochs           = 100,                                                   # número máximo de épocas en cada trial a realizar durante la búsqueda de hiperparámetros
                        executions_per_trial = 3,                                                     # número de modelos que se construyen y entrenan en cada experimento
                        overwrite            = True,                                                  # sobreescribir los resultados
                        directory            = case_id_folder,                                        # directorio en el que se guardarán los resultados de la búsqueda de hiperparámetros óptimos
                        project_name         = "Busqueda_Hiperparametros_SiMuRModel_ESANN_NET",       # nombre del proyecto asociado al ajuste de hiperparámetros
                    ) 
                    tuner.search(data_tot.X_train, data_tot.y_train,                                  # Realizar la búsqueda de hiperparámetros óptimos para el modelo:
                        epochs=5,
                        validation_data=(data_tot.X_validation, data_tot.y_validation),       
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)]) # Implementación de early-stopping para evitar el sobreentrenamiento (over-fitting) del modelo
                    tuner.search_space_summary()
                    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                    filtered_values = {   # Filtra solo los hiperparámetros que no empiecen por 'tuner/'
                        k: v for k, v in best_hps.values.items() if not k.startswith("tuner/")
                    }
                    with open(hp_json_path, "w") as f:                                        # Guardar los hiperparámetros limpios en el archivo JSON
                        json.dump(filtered_values, f, indent=4)
                    models_after_hyperparameter_search = tuner.get_best_models(num_models=1)  # Recuperación del mejor modelo en base a los hiperparámetros calculados:
                    best_model = models_after_hyperparameter_search[0]                        # El mejor modelo según keras será:
                    best_model.summary()                                                      # Obtenemos un resumen del mejor modelo
                    model_ESANN_data_tot.model = best_model                                   # Asignar el mejor modelo al objeto model_ESANN_data_tot
                    model_ESANN_data_tot.store(modelID, case_id_folder)                    
                    loss, accuracy = best_model.evaluate(data_tot.X_validation, data_tot.y_validation, verbose=1)  # Evaluar el mejor modelo en los datos de validación o test
                    print(f"Validation accuracy: {accuracy:.4f}")
                
        elif modelID == ML_Model.CAPTURE24.value:
            dataset_file = os.path.join(case_id_folder, CONVOLUTIONAL_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)
            config_file = os.path.join(case_id_folder, CONFIG_FILE)

            # IMUs muslo + muñeca
            data_tot = DataReader(modelID=modelID, create_superclasses=args.create_superclasses, p_train = args.training_percent, p_validation = args.validation_percent, 
                                  file_path=dataset_file, label_encoder_path=label_encoder_file, config_path = config_file)
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
            config_file = os.path.join(case_id_folder, CONFIG_FILE)

            # IMUs muslo + muñeca
            data_tot = DataReader(modelID=modelID, create_superclasses=args.create_superclasses, p_train = args.training_percent, p_validation = args.validation_percent,
                                   file_path=dataset_file, label_encoder_path=label_encoder_file, config_path = config_file)
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