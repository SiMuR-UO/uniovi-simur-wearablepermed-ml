import os
import sys
import argparse
import logging
from enum import Enum

import numpy as np
from wearablepermed_ml.data import DataReader
from wearablepermed_ml.models.model_generator import modelGenerator
from wearablepermed_ml.basic_functions.address import *

import tensorflow as tf

# import keras_tuner 
import json

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig, CheckpointConfig
from ray.air.config import FailureConfig
from ray.air import session

from ray.tune.tuner import TuneConfig

from wearablepermed_ml.models import SiMuRModel_ESANN, SiMuRModel_CAPTURE24, SiMuRModel_RandomForest, SiMuRModel_XGBoost
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score


# Configuration of GPU
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
        '-run-index',
        '--run-index',
        dest='run_index',
        type=str,
        default=1,
        help="Run index of each iteration of the test step."
    )
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
        "-create-superclasses-CPA-METs",
        "--create-superclasses-CPA-METs",
        dest="create_superclasses_CPA_METs",
        action='store_true',
        help="Create activity superclasses (true/false) with the CPA/METs method."
    )
    parser.add_argument(
        "-create-9-superclasses-CAPTURE24",
        "--create-9-superclasses-CAPTURE24",
        dest="create_9_superclasses_CAPTURE24",
        action='store_true',
        help="Create 9 activity superclasses (true/false) with the CAPTURE24 strategy."
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
        default=0,
        help="Validation percent"
    )    
    parser.add_argument(
        '-add-sintetic-data',
        '--add-sintetic-data',
        dest='add_sintetic_data',
        # type=bool,
        action='store_true',
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
def train_cnn_ray_tune(config, model_class, data):
    params = {
        "N_capas": config["N_capas"],
        "optimizador": config["optimizador"],
        "funcion_activacion": config["funcion_activacion"],
        "tamanho_minilote": config["tamanho_minilote"],
        "numero_filtros": config["numero_filtros"],
        "tamanho_filtro": config["tamanho_filtro"],
        "tasa_aprendizaje": config["tasa_aprendizaje"],
        "epochs": config["epochs"]
    }
    model = model_class(data, params)
    model.train(config["epochs"])
    y_pred = model.predict(data.X_validation)
    # Si devuelve probabilidades, convierte a clases
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)
    val_acc = accuracy_score(data.y_validation, y_pred)
    session.report({"val_accuracy": float(val_acc)})
    
    
def train_brf_ray_tune(config, model_class, data):
    params = {                                                 # Extraer hiperparámetros desde config
        "n_estimators": config["n_estimators"],                # Número de árboles en el bosque
        "max_depth": config["max_depth"],                      # Profundidad máxima de los árboles
        "min_samples_split": config["min_samples_split"],      # Muestras mínimas para dividir un nodo
        "min_samples_leaf": config["min_samples_leaf"],        # Muestras mínimas por hoja
        "max_features": config["max_features"],                 # Número de características consideradas por división
    }
    model = model_class(data, params)                          # Instanciar el modelo usando model_class
    model.train()                                              # Entrenar el modelo
    y_pred = model.predict(data.X_test)                        # Predecir sobre conjunto de test
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:                # Si devuelve probabilidades, convertir a clases
        y_pred = y_pred.argmax(axis=1)
    test_acc = accuracy_score(data.y_test, y_pred)             # Calcular precisión
    session.report({"test_accuracy": float(test_acc)})         # Reportar precisión a Ray Tune
    
    
def train_xgb_ray_tune(config, model_class, data):
    params = {                                                      # Extraer hiperparámetros desde config
        "num_boost_round": config["num_boost_round"],               # Número de árboles (rondas) de boosting
        "max_depth": config["max_depth"],                           # Profundidad máxima
        "learning_rate": config["learning_rate"],                   # Tasa de aprendizaje
        "subsample": config["subsample"],                           # Fracción de muestras por árbol
        "colsample_bytree": config["colsample_bytree"],             # Fracción de columnas por árbol
        "gamma": config["gamma"],                                   # Regularización mínima de pérdida
        "min_child_weight": config["min_child_weight"],             # Peso mínimo de hijos
        "reg_alpha": config["reg_alpha"],                           # L1 regularization
        "reg_lambda": config["reg_lambda"]                          # L2 regularization
    }
    model = model_class(data, params)                               # Instanciar el modelo usando model_class (ej. SiMuRModel_XGBoost)
    model.train()                                                   # Entrenar el modelo
    y_pred = model.predict(data.X_validation)                       # Predecir sobre conjunto de validación
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:                     # Si devuelve probabilidades, convertir a clases
        y_pred = y_pred.argmax(axis=1)
    validation_acc = accuracy_score(data.y_validation, y_pred)      # Calcular precisión
    session.report({"validation_accuracy": float(validation_acc)})  # Reportar precisión a Ray Tune

   
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
        
        # **********
        # Modelo A *
        # **********
        if modelID == ML_Model.ESANN.value:
            dataset_file = os.path.join(case_id_folder, CONVOLUTIONAL_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)
            config_file = os.path.join(case_id_folder, CONFIG_FILE)
            
            data_tot = DataReader(modelID=modelID, 
                                  create_superclasses=args.create_superclasses, 
                                  create_superclasses_CPA_METs = args.create_superclasses_CPA_METs,
                                  p_train = args.training_percent, 
                                  p_validation = args.validation_percent, 
                                  file_path=dataset_file, 
                                  label_encoder_path=label_encoder_file, 
                                  config_path = config_file)
            
            # Se entrenan y salvan los modelos (fichero .h5).
            # Ruta al archivo de hiperparámetros guardados
            hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_ESANN.json")
            # Verifica que el archivo existe
            if os.path.isfile(hp_json_path):
                # Cargar hiperparámetros desde el archivo JSON
                with open(hp_json_path, "r") as f:
                    best_hp_values = json.load(f)  # Diccionario: {param: valor}
                # Construir modelo usando modelGenerator y los hiperparámetros
                model_ESANN_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    params=best_hp_values,  # Pasamos directamente el diccionario
                    debug=False
                )
                # Entrenar el modelo con todos los datos
                model_ESANN_data_tot.train(best_hp_values['epochs'])
                # Guardar los pesos del modelo en formato .weights.h5
                model_ESANN_data_tot.store(modelID, case_id_folder)
            else:
                print(f"Se lanza la búsqueda de hiperparámetros óptimos del modelo")
                # -----------------------------------------------------------------------------------------------
                # Búsqueda de hiperparámetros óptimos del modelo, implementando el algoritmo ASHA según Ray Tune.
                # -----------------------------------------------------------------------------------------------
                # Espacio de búsqueda
                search_space = {
                    "N_capas": tune.randint(2, 5),                                   # Número de capas entre 2 y 7 (el límite superior es exclusivo)
                    "optimizador": tune.choice(["adam", "rmsprop"]),          # Algoritmo de optimización a usar
                    "funcion_activacion": tune.choice(["relu", "tanh"]),  # Función de activación en las capas
                    "tamanho_minilote": tune.choice([32, 64, 128]),               # Tamaño del minibatch (batch size)
                    "numero_filtros": tune.choice([32, 64, 128]),         # Cantidad de filtros para capas convolucionales
                    "tamanho_filtro": tune.choice([3, 5]),         # Tamaño del kernel (filtro) en capas convolucionales
                    "tasa_aprendizaje": tune.loguniform(1e-5, 5e-3),                 # Tasa de aprendizaje entre 0.0001 y 0.1 (escala logarítmica)
                    "epochs": tune.randint(50, 150)                                    # Número de épocas de entrenamiento entre 5 y 50
                }

                # Configuración del scheduler
                scheduler = ASHAScheduler(
                    metric="val_accuracy",        # Métrica a optimizar: precisión en el conjunto de validación
                    mode="max",                   # Se busca maximizar la métrica especificada
                    max_t=10,                     # Número máximo de iteraciones (por ejemplo, épocas) por prueba
                    grace_period=1,               # Número mínimo de iteraciones antes de detener una prueba prematuramente
                    reduction_factor=2            # Factor por el cual se reduce el número de pruebas en cada ronda de selección
                )

                # Envolver función con parámetros adicionales
                wrapped_train_fn = tune.with_parameters(
                    train_cnn_ray_tune,                # Función de entrenamiento base que se usará en la búsqueda
                    model_class=SiMuRModel_ESANN,      # Clase del modelo a usar
                    data=data_tot                      # Conjunto de datos completo que se pasará a cada ejecución de entrenamiento
                )

                # Crear el tuner
                tuner = tune.Tuner(
                    wrapped_train_fn,                                                      # Función de entrenamiento envuelta con parámetros fijos
                    param_space=search_space,                                              # Espacio de búsqueda de hiperparámetros definido antes
                    tune_config=TuneConfig(
                        scheduler=scheduler,                                               # Scheduler para manejar la parada temprana (ASHAScheduler)
                        num_samples=20,                                                    # Número de configuraciones (experimentos) a probar
                        trial_name_creator=lambda trial: f"trial_{trial.trial_id[:5]}",    # Nombre personalizado para cada prueba
                        trial_dirname_creator=lambda trial: f"dir_{trial.trial_id[:5]}"    # Carpeta personalizada para cada prueba
                    ),
                    run_config=RunConfig(
                        name="ESANN_hyperparameters_tuning",                               # Nombre general del experimento
                        storage_path=case_id_folder,                                       # Ruta donde se guardan los resultados y checkpoints
                        checkpoint_config=CheckpointConfig(num_to_keep=1),                 # Guardar solo el último checkpoint por prueba
                        failure_config=FailureConfig(fail_fast=False, max_failures=10),    # Permite hasta 10 fallos antes de parar
                        verbose=2,                                                         # Nivel de detalle en los logs (más detallado)
                        log_to_file=False                                                  # No guardar logs en archivos (evita problemas con rutas largas)
                    )
                )

                # Ejecutar búsqueda de hiperparámetros
                results = tuner.fit()

                # Obtener mejor resultado
                best_result = results.get_best_result(metric="val_accuracy", mode="max")
                print("Mejores hiperparámetros:", best_result.config)
                
                # Obtener la configuración óptima como diccionario
                mejores_hiperparametros = best_result.config

                # Guardar en un archivo JSON
                with open(os.path.join(case_id_folder,"mejores_hiperparametros_ESANN.json"), "w") as f:
                    json.dump(mejores_hiperparametros, f, indent=4)
                
                # Obtener los resultados como DataFrame
                df = results.get_dataframe()
                df.to_json(os.path.join(case_id_folder, "resultados_busqueda_ray_tune_ESANN.json"), orient="records", lines=True)
                
                # Construir modelo usando modelGenerator y los mejores hiperparámetros
                model_ESANN_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    params=mejores_hiperparametros,  # Pasamos directamente el diccionario
                    debug=False
                )
                # Entrenar el modelo con todos los datos
                model_ESANN_data_tot.train(mejores_hiperparametros['epochs'])
                # Guardar los pesos del modelo en formato .weights.h5
                model_ESANN_data_tot.store(modelID, case_id_folder)


        # **********
        # Modelo B *
        # **********
        elif modelID == ML_Model.CAPTURE24.value:
            dataset_file = os.path.join(case_id_folder, CONVOLUTIONAL_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)
            config_file = os.path.join(case_id_folder, CONFIG_FILE)
            
            data_tot = DataReader(modelID=modelID, 
                                  create_superclasses=args.create_superclasses, 
                                  create_superclasses_CPA_METs=args.create_superclasses_CPA_METs,
                                  p_train = args.training_percent, 
                                  p_validation = args.validation_percent, 
                                  file_path=dataset_file, 
                                  label_encoder_path=label_encoder_file, 
                                  config_path = config_file)
            
            # Se entrenan y salvan los modelos (fichero .h5).
            # Ruta al archivo de hiperparámetros guardados
            hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_CAPTURE24.json")
            # Verifica que el archivo existe
            if os.path.isfile(hp_json_path):
                # Cargar hiperparámetros desde el archivo JSON
                with open(hp_json_path, "r") as f:
                    best_hp_values = json.load(f)  # Diccionario: {param: valor}
                # Construir modelo usando modelGenerator y los hiperparámetros
                model_CAPTURE24_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    params=best_hp_values,  # Pasamos directamente el diccionario
                    debug=False
                )
                # Entrenar el modelo con todos los datos
                model_CAPTURE24_data_tot.train(best_hp_values['epochs'])
                # Guardar los pesos del modelo en formato .weights.h5
                model_CAPTURE24_data_tot.store(modelID, case_id_folder)
            else:
                print(f"Se lanza la búsqueda de hiperparámetros óptimos del modelo")
                # -----------------------------------------------------------------------------------------------
                # Búsqueda de hiperparámetros óptimos del modelo, implementando el algoritmo ASHA según Ray Tune.
                # -----------------------------------------------------------------------------------------------
                # Espacio de búsqueda
                # search_space = {
                #     "N_capas": tune.randint(2, 8),                                   # Número de capas entre 2 y 7 (el límite superior es exclusivo)
                #     "optimizador": tune.choice(["adam", "rmsprop", "sgd"]),          # Algoritmo de optimización a usar
                #     "funcion_activacion": tune.choice(["relu", "tanh", "sigmoid"]),  # Función de activación en las capas
                #     "tamanho_minilote": tune.choice([4, 8, 16]),               # Tamaño del minibatch (batch size)
                #     "numero_filtros": tune.choice([64, 96, 128]),         # Cantidad de filtros para capas convolucionales
                #     "tamanho_filtro": tune.choice([3, 5, 7]),         # Tamaño del kernel (filtro) en capas convolucionales
                #     "tasa_aprendizaje": tune.loguniform(1e-4, 1e-1),                 # Tasa de aprendizaje entre 0.0001 y 0.1 (escala logarítmica)
                #     "epochs": tune.randint(5, 51)                                    # Número de épocas de entrenamiento entre 5 y 50
                # }
                
                search_space = {
                    "N_capas": tune.randint(2, 4),                     # 2–4 capas
                    "optimizador": tune.choice(["adam", "rmsprop"]),
                    "funcion_activacion": tune.choice(["relu", "tanh"]),  # activaciones que funcionan mejor en CNN
                    "tamanho_minilote": tune.choice([16, 32]),        # batch pequeño para memoria limitada
                    "numero_filtros": tune.choice([32, 64, 96]),      # filtros moderados
                    "tamanho_filtro": tune.choice([3, 5]),         # tamaño de kernel razonable
                    "num_resblocks": tune.choice([0, 1]),             # 1 o 2 ResBlocks por etapa
                    "tasa_aprendizaje": tune.loguniform(1e-5, 5e-3),  # learning rate conservador
                    "epochs": tune.randint(15, 30)                    # número de epochs moderado
                }


                # Configuración del scheduler
                scheduler = ASHAScheduler(
                    metric="val_accuracy",        # Métrica a optimizar: precisión en el conjunto de validación
                    mode="max",                   # Se busca maximizar la métrica especificada
                    max_t=10,                     # Número máximo de iteraciones (por ejemplo, épocas) por prueba
                    grace_period=1,               # Número mínimo de iteraciones antes de detener una prueba prematuramente
                    reduction_factor=2            # Factor por el cual se reduce el número de pruebas en cada ronda de selección
                )

                # Envolver función con parámetros adicionales
                wrapped_train_fn = tune.with_parameters(
                    train_cnn_ray_tune,                # Función de entrenamiento base que se usará en la búsqueda
                    model_class=SiMuRModel_CAPTURE24,  # Clase del modelo a usar
                    data=data_tot                      # Conjunto de datos completo que se pasará a cada ejecución de entrenamiento
                )
                
                tuner = tune.Tuner(
                    wrapped_train_fn,
                    param_space=search_space,
                    tune_config=TuneConfig(
                        num_samples=20,
                        trial_name_creator=lambda trial: f"trial_{trial.trial_id[:5]}",
                        trial_dirname_creator=lambda trial: f"dir_{trial.trial_id[:5]}",
                    ),
                    run_config=RunConfig(
                        name="CAPTURE24_hyperparameters_tuning",
                        storage_path=case_id_folder,
                        checkpoint_config=CheckpointConfig(num_to_keep=1),
                        failure_config=FailureConfig(fail_fast=False, max_failures=10),
                        verbose=2,
                        log_to_file=False
                    )
                )
                results = tuner.fit()

                # Obtener mejor resultado
                best_result = results.get_best_result(metric="val_accuracy", mode="max")
                print("Mejores hiperparámetros:", best_result.config)
                
                # Obtener la configuración óptima como diccionario
                mejores_hiperparametros = best_result.config

                # Guardar en un archivo JSON
                with open(os.path.join(case_id_folder,"mejores_hiperparametros_CAPTURE24.json"), "w") as f:
                    json.dump(mejores_hiperparametros, f, indent=4)
                
                # Obtener los resultados como DataFrame
                df = results.get_dataframe()
                df.to_json(os.path.join(case_id_folder, "resultados_busqueda_ray_tune_CAPTURE24.json"), orient="records", lines=True)
                
                # Construir modelo usando modelGenerator y los mejores hiperparámetros
                model_CAPTURE24_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    params=mejores_hiperparametros,  # Pasamos directamente el diccionario
                    debug=False
                )
                # Entrenar el modelo con todos los datos
                model_CAPTURE24_data_tot.train(mejores_hiperparametros['epochs'])
                # Guardar los pesos del modelo en formato .weights.h5
                model_CAPTURE24_data_tot.store(modelID, case_id_folder)
                
        
        # **********
        # Modelo C *
        # **********
        elif modelID == ML_Model.RANDOM_FOREST.value:
            dataset_file = os.path.join(case_id_folder, FEATURE_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)
            config_file = os.path.join(case_id_folder, CONFIG_FILE)
            
            data_tot = DataReader(modelID=modelID,
                                  create_superclasses=args.create_superclasses,
                                  create_superclasses_CPA_METs= args.create_superclasses_CPA_METs,
                                  p_train = args.training_percent,
                                  p_validation = args.validation_percent,
                                  file_path=dataset_file, 
                                  label_encoder_path=label_encoder_file,
                                  config_path = config_file,
                                  create_9_superclasses_CAPTURE24 = args.create_9_superclasses_CAPTURE24)
            
            # Se entrenan y salvan los modelos (fichero .pkl).
            # Ruta al archivo de hiperparámetros guardados
            hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_BRF.json")
            # Verifica que el archivo existe
            if os.path.isfile(hp_json_path):
                # Cargar hiperparámetros desde el archivo JSON
                with open(hp_json_path, "r") as f:
                    best_hp_values = json.load(f)  # Diccionario: {param: valor}
                # Construir modelo usando modelGenerator y los hiperparámetros
                model_RandomForest_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    params=best_hp_values,  # Pasamos directamente el diccionario
                    debug=False
                )
                # Entrenar el modelo con todos los datos
                model_RandomForest_data_tot.train()
                # Guardar los pesos del modelo en formato .weights.h5
                model_RandomForest_data_tot.store(modelID, case_id_folder, args.run_index)
            else:
                print(f"Se lanza la búsqueda de hiperparámetros óptimos del modelo")       
                # ------------------------------------------------------------------------------------------------------
                # Búsqueda de hiperparámetros óptimos del modelo BalancedRandomForestClassifier, usando Ray Tune y ASHA
                # ------------------------------------------------------------------------------------------------------
                # Espacio de búsqueda
                search_space = {
                    "n_estimators": tune.randint(200, 500),                                  # Número de árboles entre 50 y 300
                    "max_depth": tune.choice([4, 5, 6]),                        # Profundidad máxima del árbol
                    "min_samples_split": tune.randint(40, 100),                               # Muestras mínimas para dividir un nodo
                    "min_samples_leaf": tune.randint(20, 60),                                # Muestras mínimas por hoja
                    "max_features": tune.choice(["sqrt", "log2", 0.2]),                     # Número de características por división
                    "random_state": tune.randint(0, 10000)
                }

                # Configuración del scheduler
                scheduler = ASHAScheduler(
                    metric="test_accuracy",       # Métrica a optimizar: precisión en el conjunto de test
                    mode="max",                   # Se busca maximizar la métrica especificada
                    max_t=10,                     # Número máximo de iteraciones (no se usa directamente en Random Forest, pero requerido por ASHA)
                    grace_period=1,               # Número mínimo de iteraciones antes de detener una prueba prematuramente
                    reduction_factor=2            # Factor por el cual se reduce el número de pruebas en cada ronda de selección
                )

                # Envolver función con parámetros adicionales
                wrapped_train_fn = tune.with_parameters(
                    train_brf_ray_tune,                     # Función de entrenamiento base que se usará en la búsqueda
                    model_class=SiMuRModel_RandomForest,    # Clase del modelo a usar
                    data=data_tot                           # Conjunto de datos completo que se pasará a cada ejecución de entrenamiento
                )

                # Crear el tuner
                tuner = tune.Tuner(
                    wrapped_train_fn,                                                       # Función de entrenamiento envuelta con parámetros fijos
                    param_space=search_space,                                               # Espacio de búsqueda de hiperparámetros definido antes
                    tune_config=TuneConfig(
                        scheduler=scheduler,                                                # Scheduler para manejar la parada temprana (ASHAScheduler)
                        num_samples=20,                                                     # Número de configuraciones (experimentos) a probar
                        trial_name_creator=lambda trial: f"trial_{trial.trial_id[:5]}",     # Nombre personalizado para cada prueba
                        trial_dirname_creator=lambda trial: f"dir_{trial.trial_id[:5]}"     # Carpeta personalizada para cada prueba
                    ),
                    run_config=RunConfig(
                        name="BalancedRF_hyperparameters_tuning",                           # Nombre general del experimento
                        storage_path=case_id_folder,                                        # Ruta donde se guardan los resultados y checkpoints
                        checkpoint_config=CheckpointConfig(num_to_keep=1),                  # Guardar solo el último checkpoint por prueba
                        failure_config=FailureConfig(fail_fast=False, max_failures=10),     # Permite hasta 10 fallos antes de parar
                        verbose=2,                                                          # Nivel de detalle en los logs (más detallado)
                        log_to_file=False                                                   # No guardar logs en archivos (evita problemas con rutas largas)
                    )
                )

                # Ejecutar búsqueda de hiperparámetros
                results = tuner.fit()

                # Obtener mejor resultado
                best_result = results.get_best_result(metric="test_accuracy", mode="max")
                print("Mejores hiperparámetros:", best_result.config)

                # Obtener la configuración óptima como diccionario
                mejores_hiperparametros = best_result.config

                # Guardar en un archivo JSON
                with open(os.path.join(case_id_folder,"mejores_hiperparametros_BRF.json"), "w") as f:
                    json.dump(mejores_hiperparametros, f, indent=4)

                # Obtener los resultados como DataFrame
                df = results.get_dataframe()
                df.to_json(os.path.join(case_id_folder, "resultados_busqueda_ray_tune_BRF.json"), orient="records", lines=True)
                
                # Construir modelo usando modelGenerator y los mejores hiperparámetros
                model_RandomForest_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    params=mejores_hiperparametros,  # Pasamos directamente el diccionario
                    debug=False
                )
                # Entrenar el modelo con todos los datos
                model_RandomForest_data_tot.train()
                # Guardar los pesos del modelo en formato .weights.h5
                model_RandomForest_data_tot.store(modelID, case_id_folder)
                
                
        # **********
        # Modelo D *
        # **********
        elif modelID == ML_Model.XGBOOST.value:
            dataset_file = os.path.join(case_id_folder, FEATURE_DATASET_FILE)
            label_encoder_file = os.path.join(case_id_folder, LABEL_ENCODER_FILE)
            config_file = os.path.join(case_id_folder, CONFIG_FILE)
            
            data_tot = DataReader(modelID=modelID, 
                                  create_superclasses=args.create_superclasses,
                                  create_superclasses_CPA_METs = args.create_superclasses_CPA_METs, 
                                  p_train = args.training_percent,
                                  p_validation = args.validation_percent,
                                  file_path=dataset_file,
                                  label_encoder_path=label_encoder_file,
                                  config_path = config_file)
            
            # Se entrenan y salvan los modelos (fichero .pkl).
            # Ruta al archivo de hiperparámetros guardados
            hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_XGB.json")
            # Verifica que el archivo existe
            if os.path.isfile(hp_json_path):
                # Cargar hiperparámetros desde el archivo JSON
                with open(hp_json_path, "r") as f:
                    best_hp_values = json.load(f)  # Diccionario: {param: valor}
                # Construir modelo usando modelGenerator y los hiperparámetros
                model_XGBoost_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    # params=best_hp_values,  # Pasamos directamente el diccionario
                    params=best_hp_values,
                    debug=False
                )
                # Entrenar el modelo con todos los datos de (X_train, y_train), implementando la validación con (X_validation, y_validation) 
                model_XGBoost_data_tot.train()
                # Guardar los pesos del modelo en formato .weights.h5
                model_XGBoost_data_tot.store(modelID, case_id_folder)
                
            else:
                print(f"Se lanza la búsqueda de hiperparámetros óptimos del modelo.")       
                # ------------------------------------------------------------------------------------------------------
                # Búsqueda de hiperparámetros óptimos del modelo XGBoost, usando Ray Tune (ASHA)
                # ------------------------------------------------------------------------------------------------------
                # Espacio de búsqueda para XGBoost
                search_space_xgb = {
                    "num_boost_round": tune.randint(200, 800),             # Árboles (rondas) de boosting
                    "max_depth": tune.randint(2, 4),                      # Profundidad máxima
                    "learning_rate": tune.uniform(0.005, 0.02),              # Tasa de aprendizaje
                    "subsample": tune.uniform(0.5, 0.85),                   # Fracción de muestras por árbol
                    "colsample_bytree": tune.uniform(0.5, 0.85),            # Fracción de columnas por árbol
                    "gamma": tune.uniform(5, 30),                           # Regularización mínima de pérdida
                    "min_child_weight": tune.randint(50, 200),               # Peso mínimo de hijos
                    "reg_alpha": tune.loguniform(1, 1000),                       # L1 regularization
                    "reg_lambda": tune.loguniform(5, 1000),                       # L2 regularization
                    "random_state": tune.randint(0, 10000)
                }

                # Configuración del scheduler (igual que en RF)
                scheduler = ASHAScheduler(            # Crea una instancia del scheduler ASHA para optimizar entrenamientos
                    metric="validation_accuracy",     # Métrica a optimizar (precisión en validación)
                    mode="max",                       # Indica que la métrica debe maximizarse
                    max_t=10,                         # Número máximo de iteraciones/épocas por configuración
                    grace_period=1,                   # Número mínimo de iteraciones antes de detener un trial por bajo rendimiento
                    reduction_factor=2                # Factor de reducción para descartar configuraciones poco prometedoras
                )

                # Envolver función de entrenamiento
                wrapped_train_fn_xgb = tune.with_parameters(   # Crea una versión de la función con parámetros fijos predefinidos
                    train_xgb_ray_tune,                        # Función de entrenamiento adaptada a XGBoost
                    model_class=SiMuRModel_XGBoost,            # Clase del modelo a utilizar (implementación XGBoost personalizada)
                    data=data_tot                              # Conjunto de datos completo que se usará en el entrenamiento
                )

                # Crear el tuner
                tuner_xgb = tune.Tuner(                                                   # Crea un objeto Tuner para ejecutar la búsqueda de hiperparámetros
                    wrapped_train_fn_xgb,                                                 # Función de entrenamiento envuelta con parámetros fijos
                    param_space=search_space_xgb,                                         # Espacio de búsqueda de hiperparámetros
                    tune_config=TuneConfig(                                               # Configuración de la optimización
                        scheduler=scheduler,                                              # Planificador (scheduler) para gestionar recursos y early stopping
                        num_samples=20,                                                   # Número de configuraciones distintas a probar
                        trial_name_creator=lambda trial: f"trial_{trial.trial_id[:5]}",   # Nombre personalizado para cada experimento
                        trial_dirname_creator=lambda trial: f"dir_{trial.trial_id[:5]}"   # Carpeta personalizada para cada experimento
                    ),
                    run_config=RunConfig(                                                 # Configuración de ejecución de los experimentos
                        name="XGBoost_hyperparameters_tuning",                            # Nombre general de la ejecución
                        storage_path=case_id_folder,                                      # Carpeta donde guardar resultados y checkpoints
                        checkpoint_config=CheckpointConfig(num_to_keep=1),                # Mantener solo el último checkpoint por trial
                        failure_config=FailureConfig(fail_fast=False, max_failures=10),   # Permitir hasta 10 fallos sin abortar
                        verbose=2,                                                        # Nivel de detalle en la salida por consola
                        log_to_file=False                                                 # No guardar logs en archivo (solo consola)
                    )
                )

                # Ejecutar la búsqueda
                results_xgb = tuner_xgb.fit()

                # Mejor resultado
                best_result_xgb = results_xgb.get_best_result(metric="validation_accuracy", mode="max")
                print("Mejores hiperparámetros XGBoost:", best_result_xgb.config)

                # Guardar en JSON
                mejores_hiperparametros_xgb = best_result_xgb.config
                with open(os.path.join(case_id_folder, "mejores_hiperparametros_XGB.json"), "w") as f:
                    json.dump(mejores_hiperparametros_xgb, f, indent=4)

                # Guardar todos los resultados
                df_xgb = results_xgb.get_dataframe()
                df_xgb.to_json(os.path.join(case_id_folder, "resultados_busqueda_ray_tune_XGB.json"), orient="records", lines=True)

                # Construir y entrenar modelo con mejores hiperparámetros
                model_XGB_data_tot = modelGenerator(
                    modelID=modelID,
                    data=data_tot,
                    params=mejores_hiperparametros_xgb,
                    debug=False
                )
                model_XGB_data_tot.train()
                model_XGB_data_tot.store(modelID, case_id_folder)

         
        _logger.info("Script ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()            