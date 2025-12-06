from enum import Enum
import json
from wearablepermed_ml.data import DataReader
from wearablepermed_ml.models.model_generator import modelGenerator
from wearablepermed_ml.basic_functions.address import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import joblib

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'
    
def tester(case_id_folder, model_id, create_superclasses, create_superclasses_CPA_METs, create_9_superclasses_CAPTURE24, training_percent, validation_percent, run_index):
    # Cargar el LabelEncoder
    # Ver las clases asociadas a cada número
    test_label_encoder_path = os.path.join(case_id_folder, "label_encoder.pkl")
    label_encoder = joblib.load(test_label_encoder_path)

    print(label_encoder.classes_)

    # class_names_total = ['CAMINAR CON LA COMPRA', 'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR USUAL SPEED',
    # 'CAMINAR ZIGZAG', 'DE PIE BARRIENDO', 'DE PIE DOBLANDO TOALLAS',
    # 'DE PIE MOVIENDO LIBROS', 'DE PIE USANDO PC', 'FASE REPOSO CON K5',
    # 'INCREMENTAL CICLOERGOMETRO', 'SENTADO LEYENDO', 'SENTADO USANDO PC',
    # 'SENTADO VIENDO LA TV', 'SIT TO STAND 30 s', 'SUBIR Y BAJAR ESCALERAS',
    # 'TAPIZ RODANTE', 'TROTAR', 'YOGA']
    
    # class_names_total = ['CAMINAR CON LA COMPRA', 'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR USUAL SPEED',
    # 'CAMINAR ZIGZAG', 'DE PIE BARRIENDO', 'DE PIE DOBLANDO TOALLAS',
    # 'DE PIE MOVIENDO LIBROS', 'DE PIE USANDO PC', 'FASE REPOSO CON K5',
    # 'INCREMENTAL CICLOERGOMETRO', 'SENTADO LEYENDO', 'SENTADO USANDO PC',
    # 'SENTADO VIENDO LA TV', 'SUBIR Y BAJAR ESCALERAS',
    # 'TAPIZ RODANTE', 'TROTAR', 'YOGA']
    
    class_names_total = label_encoder.classes_

    print(len(class_names_total))

    # Obtener el mapeo de cada etiqueta a su número asignado
    mapeo = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Mapeo de etiquetas:", mapeo)

    # Lectura de hiperparámetros óptimos de cada modelo, previamente buscados
    # ------------------------------------------------------------------------------------
    if (model_id == ML_Model.ESANN.value):
        test_dataset_path = os.path.join(case_id_folder, "data_all.npz")
        # Ruta al archivo de hiperparámetros guardados
        hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_ESANN.json")
        # Cargar hiperparámetros desde el archivo JSON
        with open(hp_json_path, "r") as f:
            best_hp_values = json.load(f)  # Diccionario: {param: valor}
    
    elif (model_id == ML_Model.CAPTURE24.value):
        test_dataset_path = os.path.join(case_id_folder, "data_all.npz")
        # Ruta al archivo de hiperparámetros guardados
        hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_CAPTURE24.json")
        # Cargar hiperparámetros desde el archivo JSON
        with open(hp_json_path, "r") as f:
            best_hp_values = json.load(f)  # Diccionario: {param: valor}
    
    elif (model_id == ML_Model.RANDOM_FOREST.value):
        test_dataset_path = os.path.join(case_id_folder, "data_feature_all.npz")
        # Ruta al archivo de hiperparámetros guardados
        hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_BRF.json")
        # Cargar hiperparámetros desde el archivo JSON
        with open(hp_json_path, "r") as f:
            best_hp_values = json.load(f)  # Diccionario: {param: valor}

    elif (model_id == ML_Model.XGBOOST.value):
        test_dataset_path = os.path.join(case_id_folder, "data_feature_all.npz")
        # Ruta al archivo de hiperparámetros guardados
        hp_json_path = os.path.join(case_id_folder, "mejores_hiperparametros_XGB.json")
        # Cargar hiperparámetros desde el archivo JSON
        with open(hp_json_path, "r") as f:
            best_hp_values = json.load(f)  # Diccionario: {param: valor}
        
    # Testeamos el rendimiento del modelo de clasificación con los DATOS TOTALES
    data = DataReader(modelID=model_id, 
                      create_superclasses=create_superclasses,
                      create_superclasses_CPA_METs = create_superclasses_CPA_METs, 
                      p_train = training_percent, 
                      p_validation=validation_percent, 
                      file_path=test_dataset_path, 
                      label_encoder_path=test_label_encoder_path,
                      create_9_superclasses_CAPTURE24=create_9_superclasses_CAPTURE24)

    # Construir modelo usando modelGenerator y los hiperparámetros
    model = modelGenerator(
        modelID=model_id,
        data=data,
        params=best_hp_values,  # Pasamos directamente el diccionario de hiperparámetros óptimos
        debug=False
    )

    model.load(model_id, case_id_folder)

    # print train/test sizes
    print(model.X_test.shape)
    print(model.X_train.shape)

    # testing the model
    y_predicted_train = model.predict(model.X_train)
    y_predicted_test = model.predict(model.X_test)

    # get the class with the highest probability
    if (model_id == ML_Model.ESANN.value or model_id == ML_Model.CAPTURE24.value):
        y_predicted_validation = model.predict(model.X_validation)
        y_final_prediction_train = np.argmax(y_predicted_train, axis=1)
        y_final_prediction_validation = np.argmax(y_predicted_validation, axis=1)
        y_final_prediction_test = np.argmax(y_predicted_test, axis=1)  # Trabajamos con clasificación multicategoría, no necesario para los bosques aleatorios
        
        acc_score_validation = accuracy_score(model.y_validation, y_final_prediction_validation)
        print("Global accuracy score (validation) = "+str(round(acc_score_validation*100,2))+" [%]")
        F1_score_validation = f1_score(model.y_validation, y_final_prediction_validation, average='macro')    # revisar las opciones de average
        print("Global F1 score (validation) = "+str(round(F1_score_validation*100,2))+" [%]")

    else: # random forest, xgboost
        y_final_prediction_train = y_predicted_train
        y_final_prediction_test = y_predicted_test   # esta línea solo es necesaria para los bosques aleatorios y XGBoost


    print(model.y_test)
    print(model.y_test.shape)

    print(y_predicted_test)
    print(y_predicted_test.shape)

    # Matriz de confusión
    # Obtener todas las clases posibles desde 0 hasta N-1
    num_classes = len(class_names_total)  # Asegurar que contiene todas las clases esperadas
    all_classes = np.arange(num_classes)  # Crear array con todas las clases (0, 1, 2, ..., N-1)

    # Crear la matriz de confusión asegurando que todas las clases están representadas
    cm = confusion_matrix(model.y_test, y_final_prediction_test, labels=all_classes)

    # Graficar la matriz de confusión
    confusion_matrix_test_path = os.path.join(case_id_folder, "confusion_matrix_test_"+str(run_index)+".png")

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_total, yticklabels=class_names_total)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix Test')
    plt.savefig(confusion_matrix_test_path, bbox_inches='tight')

    # -------------------------------------------------
    # MÉTRICAS DE TEST GLOBALES
    print("-------------------------------------------------\n")

    # Accuracy
    acc_score_train = accuracy_score(model.y_train, y_final_prediction_train)
    print("Global accuracy score (train) = "+str(round(acc_score_train*100,2))+" [%]")    
    
    acc_score_test = accuracy_score(model.y_test, y_final_prediction_test)
    print("Global accuracy score (test) = "+str(round(acc_score_test*100,2))+" [%]")

    # F1 Score
    F1_score_train = f1_score(model.y_train, y_final_prediction_train, average='macro')    # revisar las opciones de average
    print("Global F1 score (train) = "+str(round(F1_score_train*100,2))+" [%]")
    
    F1_score_test = f1_score(model.y_test, y_final_prediction_test, average='macro')    # revisar las opciones de average
    print("Global F1 score (test) = "+str(round(F1_score_test*100,2))+" [%]")

    # Recall global
    # recall_score_global = recall_score(model.y_test, y_final_predicton, average='macro')
    # print("Global recall score = "+str(round(recall_score_global*100,2))+" [%]")

    # Save to a file
    clasification_global_report_path = os.path.join(case_id_folder, "clasification_global_report_"+str(run_index)+".txt")
    with open(clasification_global_report_path, "w") as f:
        f.write(f"Global F1 Score (train): {F1_score_train:.4f}\n")
        f.write(f"Global accuracy score (train): {acc_score_train:.4f}\n")
        if model_id in [ML_Model.ESANN, ML_Model.CAPTURE24, ML_Model.XGBOOST]:
            f.write(f"Global F1 Score (validation): {F1_score_validation:.4f}\n")
            f.write(f"Global accuracy score (validation): {acc_score_validation:.4f}\n")
        f.write(f"Global F1 Score (test): {F1_score_test:.4f}\n")
        f.write(f"Global accuracy score (test): {acc_score_test:.4f}\n")
        # f.write(f"Global recall score: {recall_score_global:.4f}\n")

    # -------------------------------------------------
    # Obtener todas las clases posibles desde 0 hasta N-1
    num_classes = len(class_names_total)  # Número total de clases
    all_classes = np.arange(num_classes)  # Crea un array con todas las clases (0, 1, 2, ..., N-1)

    # Tabla de métricas para cada clase
    classification_per_class_report = classification_report(
        model.y_test,
        y_final_prediction_test,
        labels=all_classes,
        target_names=class_names_total,
        zero_division=0
    )
    print(classification_per_class_report)        

    # Save per-class report to a file
    clasification_per_class_report_path = os.path.join(case_id_folder, "clasification_per_class_report_"+str(run_index)+".txt")
    with open(clasification_per_class_report_path, "w") as f:        
        f.write(classification_per_class_report)