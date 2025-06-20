from enum import Enum
from data import DataReader
from models.model_generator import modelGenerator
from basic_functions.address import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from pandas import DataFrame as df
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'
    
def tester(model_id, dataset_folder, training_percent):
    # Cargar el LabelEncoder
    # Ver las clases asociadas a cada número
    label_encoder_path = os.path.join(dataset_folder, "label_encoder.pkl")
    label_encoder = joblib.load(label_encoder_path)

    print(label_encoder.classes_)

    class_names_total = ['CAMINAR CON LA COMPRA', 'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR USUAL SPEED',
    'CAMINAR ZIGZAG', 'DE PIE BARRIENDO', 'DE PIE DOBLANDO TOALLAS',
    'DE PIE MOVIENDO LIBROS', 'DE PIE USANDO PC', 'FASE REPOSO CON K5',
    'INCREMENTAL CICLOERGOMETRO', 'SENTADO LEYENDO', 'SENTADO USANDO PC',
    'SENTADO VIENDO LA TV', 'SIT TO STAND 30 s', 'SUBIR Y BAJAR ESCALERAS',
    'TAPIZ RODANTE', 'TROTAR', 'YOGA']

    print(len(class_names_total))

    # Obtener el mapeo de cada etiqueta a su número asignado
    mapeo = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Mapeo de etiquetas:", mapeo)

    # get model type from filename
    mode_tokens = model_id.split("_")

    if (mode_tokens[1] == ML_Model.ESANN.value):
        test_dataset_path = os.path.join(dataset_folder, "data_all.npz")

        params = {
            "optimizer": "rmsprop",
            "miniBatchSize": 10,
            "lr": 0.00045493796608069996,
            "N_capas": 2,
            "activation": "relu",
            "numFilters": 12,
            "filterSize": 7
        }
    elif (mode_tokens[1] == ML_Model.CAPTURE24.value):
        test_dataset_path = os.path.join(dataset_folder, "data_all.npz")

        params = {
            "optimizer": "rmsprop",
            "miniBatchSize": 10,
            "lr": 0.00045493796608069996,
            "N_capas": 6,
            "activation": "relu",
            "numFilters": 12,
            "filterSize": 7
        }
    elif (mode_tokens[1] == ML_Model.RANDOM_FOREST.value):
        test_dataset_path = os.path.join(dataset_folder, "data_feature_all.npz")

        params = {
            "n_estimators": 500
        }

    # Testeamos el rendimiento del modelo de clasificación con los DATOS TOTALES
    data = DataReader(p_train = training_percent, dataset=test_dataset_path, random_state=42)

    model = modelGenerator(modelID=model_id, data=data, params=params, debug=False)

    model_trained_path = os.path.join(dataset_folder, model_id + ".weights.h5")
    model.load(model_id, model_trained_path)