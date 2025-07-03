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
    
def tester(case_id_folder, model_id, training_percent):
    # Cargar el LabelEncoder
    # Ver las clases asociadas a cada número
    test_label_encoder_path = os.path.join(case_id_folder, "label_encoder.pkl")
    label_encoder = joblib.load(test_label_encoder_path)

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

    if (model_id == ML_Model.ESANN.value):
        test_dataset_path = os.path.join(case_id_folder, "data_all.npz")

        params = {
            "optimizer": "rmsprop",
            "miniBatchSize": 10,
            "lr": 0.00045493796608069996,
            "N_capas": 2,
            "activation": "relu",
            "numFilters": 12,
            "filterSize": 7
        }
    elif (model_id == ML_Model.CAPTURE24.value):
        test_dataset_path = os.path.join(case_id_folder, "data_all.npz")

        params = {
            "optimizer": "rmsprop",
            "miniBatchSize": 10,
            "lr": 0.00045493796608069996,
            "N_capas": 6,
            "activation": "relu",
            "numFilters": 12,
            "filterSize": 7
        }
    elif (model_id == ML_Model.RANDOM_FOREST.value):
        test_dataset_path = os.path.join(case_id_folder, "data_feature_all.npz")

        params = {
            "n_estimators": 500
        }

    elif (model_id == ML_Model.XGBOOST.value):
        raise Exception("Model training not implemented")
        
    # Testeamos el rendimiento del modelo de clasificación con los DATOS TOTALES
    data = DataReader(modelID=model_id, p_train = training_percent, file_path=test_dataset_path, label_encoder_path=test_label_encoder_path)

    model = modelGenerator(modelID=model_id, data=data, params=params, debug=False)

    model.load(model_id, case_id_folder)

    # print train/test sizes
    print(model.X_test.shape)
    print(model.X_train.shape)

    # testing the model
    y_predicted = model.predict(model.X_test)

    # get the class with the highest probability
    if (model_id == ML_Model.ESANN.value or model_id == ML_Model.CAPTURE24.value):
        y_final_predicton = np.argmax(y_predicted, axis=1)  # Trabajamos con clasificación multicategoría, no necesario para los bosques aleatorios
    else:
        y_final_predicton = y_predicted   # esta línea solo es necesaria para los bosques aleatorios

    print(model.y_test)
    print(model.y_test.shape)

    print(y_predicted)
    print(y_predicted.shape)

    # Matriz de confusión
    # Obtener todas las clases posibles desde 0 hasta N-1
    num_classes = len(class_names_total)  # Asegurar que contiene todas las clases esperadas
    all_classes = np.arange(num_classes)  # Crear array con todas las clases (0, 1, 2, ..., N-1)

    # Crear la matriz de confusión asegurando que todas las clases están representadas
    cm = confusion_matrix(model.y_test, y_final_predicton, labels=all_classes)

    # Graficar la matriz de confusión
    confusion_matrix_test_path = os.path.join(case_id_folder, "confusion_matrix_test.png")

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_total, yticklabels=class_names_total)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix Test')
    plt.savefig(confusion_matrix_test_path, bbox_inches='tight')

    # MÉTRICAS DE TEST GLOBALES
    print("-------------------------------------------------\n")
    acc_score = accuracy_score(model.y_test, y_final_predicton)
    print("Global accuracy score = "+str(round(acc_score*100,2))+" [%]")

    F1_score = f1_score(model.y_test, y_final_predicton, average='macro')    # revisar las opciones de average
    print("Global F1 score = "+str(round(F1_score*100,2))+" [%]")

    # Save to a file
    clasification_global_report_path = os.path.join(case_id_folder, "clasification_global_report.txt")
    with open(clasification_global_report_path, "w") as f:
        f.write(f"Global F1 Score: {F1_score:.4f}\n")
        f.write(f"Global accuracy score: {acc_score:.4f}\n")

    # Obtener todas las clases posibles desde 0 hasta N-1
    num_classes = len(class_names_total)  # Asegúrate de que contiene TODAS las clases, incluso si no están en y_test
    all_classes = np.arange(num_classes)  # Crea un array con todas las clases (0, 1, 2, ..., N-1)

    # Tabla de métricas para cada clase
    # Save to a file
    classification_per_class_report = classification_report(model.y_test, y_final_predicton, labels=all_classes, target_names=class_names_total, zero_division=0)
    print(classification_per_class_report)        

    clasification_per_class_report_path = os.path.join(case_id_folder, "clasification_per_class_report.txt")
    with open(clasification_per_class_report_path, "w") as f:        
        f.write(classification_per_class_report)