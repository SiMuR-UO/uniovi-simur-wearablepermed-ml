# En este script se preprocesan los datos.

# Se normaliza, limpian , filtran, etc.

# El resultado puede ser una clase o un dictionario que contenga:

#         data.X_train
#         data.y_train
#         data.X_test
#         data.y_test


from enum import Enum
import os
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import defaultdict

_DEF_WINDOWS_REBALANCED_MEAN = 50 # for all tasks (training + test)

class ML_Model(Enum):
    ESANN = 'ESANN'
    CAPTURE24 = 'CAPTURE24'
    RANDOM_FOREST = 'RandomForest'
    XGBOOST = 'XGBoost'

WINDOW_CONCATENATED_DATA = "arr_0"
WINDOW_ALL_LABELS = "arr_1"
WINDOW_ALL_METADATA = "arr_2"

# Jittering
def jitter(X, sigma=0.5):
    # Añadir ruido gaussiano a los datos
    return X + np.random.normal(loc=0, scale=sigma, size=X.shape)


# Magnitude Warping
def magnitude_warp(X, sigma=0.2):
    """
    Aplica una distorsión en la magnitud de un vector 1D o matriz 2D.
    
    Parámetros:
    - X: np.array de 1D (shape (n,)) o 2D (shape (n_samples, n_features))
    - sigma: Desviación estándar del ruido gaussiano aplicado.
    
    Retorna:
    - X modificado con la distorsión aplicada.
    """
    factor = np.random.normal(1, sigma, X.shape)  # Genera un factor de escala aleatorio para cada elemento
    return X * factor


def shift(X, shift_max=2):
    """
    Aplica un desplazamiento aleatorio a un vector 1D.

    Parámetros:
    - X: np.array de 1D (shape (n,))
    - shift_max: Máximo número de posiciones a desplazar (positivo o negativo).

    Retorna:
    - np.array con los valores desplazados aleatoriamente.
    """
    shift = np.random.randint(-shift_max, shift_max + 1)  # Generar shift aleatorio
    return np.roll(X, shift)  # Aplicar desplazamiento


def time_warp(X, sigma=0.2):
    """
    Aplica un time warping sobre un vector 1D, distorsionando su temporalidad.

    Parámetros:
    - X: np.array de 1D (shape (n,))
    - sigma: Desviación estándar del ruido gaussiano aplicado a las distorsiones.

    Retorna:
    - np.array con la serie temporal distorsionada.
    """
    n = len(X)
    # Creamos un desplazamiento para cada índice, que sigue una distribución normal.
    time_warp = np.cumsum(np.random.normal(1, sigma, n))  # Cumsum para obtener una curva suave

    # Normalizamos para que el tiempo total no cambie (para que no se expanda ni se contraiga el vector)
    time_warp -= time_warp[0]
    time_warp /= time_warp[-1]

    # Interpolamos el vector original según la distRorsión
    new_indices = np.interp(np.linspace(0, 1, n), time_warp, np.linspace(0, 1, n))
    X_new = np.interp(new_indices, np.linspace(0, 1, n), X)

    return X_new

def config_participants(config_path, metadata_keys_train, metadata_keys_validation, metadata_keys_test):
    with open(config_path, "r") as f:
        lines = f.readlines()

    # Replace content from line 5 onward (i.e. index 4)
    new_lines = lines[:5]  # Keep first 4 lines (up to line 4)
    new_lines += [
        "\nTraining participants: " + ",".join(metadata_keys_train)+"\n\n",
        "Validation participants: " + ",".join(metadata_keys_validation)+"\n\n",
        "Testing participants: " + ",".join(metadata_keys_test)+"\n\n"
    ]
                
    with open(config_path, "w") as f:
        f.writelines(new_lines)

def aggregate_superclasses(etiquetas_output):
    etiquetas_superclase_1 = ['CAMINAR CON LA COMPRA', 'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR USUAL SPEED', 'CAMINAR ZIGZAG']
    etiquetas_superclase_2 = ['DE PIE BARRIENDO', 'DE PIE DOBLANDO TOALLAS', 'DE PIE MOVIENDO LIBROS', 'DE PIE USANDO PC', 'YOGA', 'SUBIR Y BAJAR ESCALERAS']
    etiquetas_superclase_3 = ['FASE REPOSO CON K5', 'SENTADO LEYENDO', 'SENTADO USANDO PC', 'SENTADO VIENDO LA TV']
    etiquetas_superclase_4 = ['TAPIZ RODANTE', 'TROTAR', 'INCREMENTAL CICLOERGOMETRO']

    for i in range(len(etiquetas_output)):
        if etiquetas_output[i] in etiquetas_superclase_1:
            etiquetas_output[i] = 'CAMINAR'
        elif etiquetas_output[i] in etiquetas_superclase_2:
            etiquetas_output[i] = 'DE PIE + ACTIVIDAD'
        elif etiquetas_output[i] in etiquetas_superclase_3:
            etiquetas_output[i] = 'SENTADO/REPOSO'
        elif etiquetas_output[i] in etiquetas_superclase_4:
            etiquetas_output[i] = 'CORRER/PEDALEAR'
    
    return etiquetas_output

def aggregate_superclasses_CPA_METs(etiquetas_output):
    etiquetas_superclase_1 = ['FASE REPOSO CON K5', 'SENTADO LEYENDO', 'SENTADO USANDO PC', 'SENTADO VIENDO LA TV']
    etiquetas_superclase_2 = ['DE PIE DOBLANDO TOALLAS', 'DE PIE USANDO PC', 'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR ZIGZAG']
    etiquetas_superclase_3 = ['DE PIE BARRIENDO', 'DE PIE MOVIENDO LIBROS', 'CAMINAR CON LA COMPRA', 'CAMINAR USUAL SPEED', 'SUBIR Y BAJAR ESCALERAS']
    etiquetas_superclase_4 = ['INCREMENTAL CICLOERGOMETRO', 'TROTAR']

    for i in range(len(etiquetas_output)):
        if etiquetas_output[i] in etiquetas_superclase_1:
            etiquetas_output[i] = 'SEDENTARY'
        elif etiquetas_output[i] in etiquetas_superclase_2:
            etiquetas_output[i] = 'LIGHT-INTENSITY'
        elif etiquetas_output[i] in etiquetas_superclase_3:
            etiquetas_output[i] = 'MODERATE-INTENSITY'
        elif etiquetas_output[i] in etiquetas_superclase_4:
            etiquetas_output[i] = 'VIGOROUS-INTENSITY'
    
    return etiquetas_output

  
   
def aggregate_9_superclasses_CAPTURE24(etiquetas_output):
    etiquetas_superclase_1 = ['CAMINAR CON LA COMPRA', 'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR USUAL SPEED', 'CAMINAR ZIGZAG']
    etiquetas_superclase_2 = ['DE PIE BARRIENDO', 'DE PIE MOVIENDO LIBROS', 'DE PIE DOBLANDO TOALLAS']
    # etiquetas_superclase_3 = ['DE PIE DOBLANDO TOALLAS']
    etiquetas_superclase_4 = ['DE PIE USANDO PC']
    etiquetas_superclase_5 = ['FASE REPOSO CON K5']
    etiquetas_superclase_6 = ['INCREMENTAL CICLOERGOMETRO']
    etiquetas_superclase_7 = ['SENTADO LEYENDO', 'SENTADO USANDO PC', 'SENTADO VIENDO LA TV']
    etiquetas_superclase_8 = ['SUBIR Y BAJAR ESCALERAS']
    etiquetas_superclase_9 = ['TROTAR']
    

    for i in range(len(etiquetas_output)):
        if etiquetas_output[i] in etiquetas_superclase_1:
            etiquetas_output[i] = 'WALKING'
        elif etiquetas_output[i] in etiquetas_superclase_2:
            etiquetas_output[i] = 'HOUSEHOLD-CHORES'
        # elif etiquetas_output[i] in etiquetas_superclase_3:
        #     etiquetas_output[i] = 'MANUAL-WORK'
        elif etiquetas_output[i] in etiquetas_superclase_4:
            etiquetas_output[i] = 'STANDING'
        elif etiquetas_output[i] in etiquetas_superclase_5:
            etiquetas_output[i] = 'SLEEP'
        elif etiquetas_output[i] in etiquetas_superclase_6:
            etiquetas_output[i] = 'BICYCLING'
        elif etiquetas_output[i] in etiquetas_superclase_7:
            etiquetas_output[i] = 'SITTING'
        elif etiquetas_output[i] in etiquetas_superclase_8:
            etiquetas_output[i] = 'MIXED-ACTIVITY'
        elif etiquetas_output[i] in etiquetas_superclase_9:
            etiquetas_output[i] = 'SPORTS'
    
    return etiquetas_output


def rebalanced(data, labels, metadata):
    # flat three datasets in one dictionary
    grouped = defaultdict(lambda: defaultdict(list))

    for xi, yi, mi in zip(data, labels, metadata):
        grouped[mi][yi].append(xi)

    participants = {mi: dict(classes) for mi, classes in grouped.items()}

    # rebalanced
    for participant_key in participants:
        for activity_key in participants[str(participant_key)]:
            try:            
                random_windows = random.sample(participants[str(participant_key)][str(activity_key)], _DEF_WINDOWS_REBALANCED_MEAN)
                participants[str(participant_key)][str(activity_key)] = random_windows
            except:
                print("This activity can't be balanced (in a downsampling way)")

    # return to three datasets from dictionary
    data_reconstructed = []
    labels_reconstructed = []
    metadata_reconstructed = []

    for metadata, class_participant in participants.items():
        for label, windows in class_participant.items():
            for window in windows:
                data_reconstructed.append(window)
                labels_reconstructed.append(label)
                metadata_reconstructed.append(metadata)
                
    data_reconstructed_stack=np.stack(data_reconstructed, axis=0)
    
    return data_reconstructed_stack, labels_reconstructed, metadata_reconstructed   

    
class DataReader(object):
    def __init__(self, modelID, create_superclasses, create_superclasses_CPA_METs, p_train, p_validation, file_path, label_encoder_path, config_path=None, add_sintetic_data=False, create_9_superclasses_CAPTURE24=False):        
        self.p_train = p_train / 100

        if (p_validation is not None):
            self.p_validation = p_validation / 100
            self.p_test = 1 - (self.p_train + self.p_validation )
        else:
            self.p_test = 1 - ( self.p_train )

        stack_de_datos_y_etiquetas_PMP_tot = np.load(file_path)
        datos_input = stack_de_datos_y_etiquetas_PMP_tot[WINDOW_CONCATENATED_DATA]
        etiquetas_output = stack_de_datos_y_etiquetas_PMP_tot[WINDOW_ALL_LABELS]
        metadata_output = stack_de_datos_y_etiquetas_PMP_tot[WINDOW_ALL_METADATA]
        
        idx = np.random.permutation(datos_input.shape[0])
        datos_input = datos_input[idx]
        etiquetas_output = etiquetas_output[idx]
        metadata_output = metadata_output[idx]

        # X data
        # X = datos_input
        
        # Creation of Activity Superclasses
        if create_superclasses == True:
            etiquetas_output = aggregate_superclasses(etiquetas_output)
            datos_input, etiquetas_output, metadata_output = rebalanced(datos_input, etiquetas_output, metadata_output)
            
        if create_superclasses_CPA_METs == True:
            etiquetas_output = aggregate_superclasses_CPA_METs(etiquetas_output)
            datos_input, etiquetas_output, metadata_output = rebalanced(datos_input, etiquetas_output, metadata_output)
            
        if create_9_superclasses_CAPTURE24 == True:
            etiquetas_output = aggregate_9_superclasses_CAPTURE24(etiquetas_output)
            datos_input, etiquetas_output, metadata_output = rebalanced(datos_input, etiquetas_output, metadata_output)
        
        # y data
        # Codificación numérica de las etiquetas para cada muestra de datos
        # Crear el codificador de etiquetas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(etiquetas_output)
        
        # Split train and test datasets
        grouped = defaultdict(list)
        for s in metadata_output:
            grouped[s].append(s)
        metadata_grouped = dict(grouped)

        metadata_keys = list(metadata_grouped.keys())
        metadata_keys_len = len(metadata_keys)
        
        number_of_keys_train = round(metadata_keys_len * self.p_train)
        metadata_keys_train = metadata_keys[0:number_of_keys_train]
        
        number_of_keys_validation = round(metadata_keys_len * self.p_validation)
        metadata_keys_validation = metadata_keys[number_of_keys_train:(number_of_keys_train+number_of_keys_validation)]
        
        number_of_keys_test = round(metadata_keys_len * self.p_test)
        metadata_keys_test = metadata_keys[(number_of_keys_train+number_of_keys_validation):(number_of_keys_train+number_of_keys_validation+number_of_keys_test)]
        
        if modelID == ML_Model.RANDOM_FOREST.value or modelID == ML_Model.XGBOOST.value:
            X_train = np.empty((0, datos_input.shape[1]))  # Inicializar vacío con n columnas
            X_validation = np.empty((0, datos_input.shape[1]))  # Inicializar vacío con n columnas
            X_test = np.empty((0, datos_input.shape[1]))
        elif modelID == ML_Model.ESANN.value or modelID == ML_Model.CAPTURE24.value:
            X_train_list = []
            X_validation_list = []
            X_test_list = []
            
        y_train = np.empty((0, 1))
        y_validation = np.empty((0, 1))
        y_test = np.empty((0, 1))
        

        # Save training, validation and test participants in the config file only in training step
        if (config_path is not None):
            config_participants(config_path, metadata_keys_train, metadata_keys_validation, metadata_keys_test)

        # Split train, validation and test datasets by participant
        for i in range(datos_input.shape[0]):
            participant_id_i = metadata_output[i]
            if participant_id_i in metadata_keys_train:
                if modelID == ML_Model.RANDOM_FOREST.value or modelID == ML_Model.XGBOOST.value:
                    fila_data = datos_input[i, :].reshape(1, -1)  # Asegura forma (1, n)
                    X_train = np.vstack([X_train, fila_data])
                elif modelID == ML_Model.ESANN.value or modelID == ML_Model.CAPTURE24.value:
                    window_data = datos_input[i, :, :]
                    X_train_list.append(window_data)
                
                label_i = y_encoded[i]
                label_i = np.array([[label_i]])
                y_train = np.vstack([y_train, label_i])

            if participant_id_i in metadata_keys_validation:
                if modelID == ML_Model.RANDOM_FOREST.value or modelID == ML_Model.XGBOOST.value:
                    fila_data = datos_input[i, :].reshape(1, -1)  # Asegura forma (1, n)
                    X_validation = np.vstack([X_validation, fila_data])
                elif modelID == ML_Model.ESANN.value or modelID == ML_Model.CAPTURE24.value:
                    window_data = datos_input[i, :, :]
                    X_validation_list.append(window_data)
                
                label_i = y_encoded[i]
                label_i = np.array([[label_i]])
                y_validation = np.vstack([y_validation, label_i])
                
            if participant_id_i in metadata_keys_test:
                if modelID == ML_Model.RANDOM_FOREST.value or modelID == ML_Model.XGBOOST.value:
                    fila_data = datos_input[i, :].reshape(1, -1)  # Asegura forma (1, n)
                    X_test = np.vstack([X_test, fila_data])
                elif modelID == ML_Model.ESANN.value or modelID == ML_Model.CAPTURE24.value:
                    window_data = datos_input[i, :, :]
                    X_test_list.append(window_data)
                
                label_i = y_encoded[i]
                label_i = np.array([[label_i]])
                y_test = np.vstack([y_test, label_i])
        
        try:      
            if X_train_list:
                X_train = np.stack(X_train_list)
            if X_validation_list:
                X_validation = np.stack(X_validation_list)
            if X_test_list:
                X_test = np.stack(X_test_list)
        except:
            print("Training a non-convolutional model.")

        # --------------------------------------------------------------------------------------------------
        # Realizamos el aumento de datos en el conjunto de entrenamiento. En el conjunto de test mantenemos
        # los datos origifile_pathnales:
        num_filas = X_train.shape[0]
        num_columnas = X_train.shape[1]

        if ((modelID == ML_Model.ESANN or modelID == ML_Model.CAPTURE24) and add_sintetic_data == True):
            profundidad = X_train.shape[2]
        
            # 1.- Jittering
            # ---------------------------
            # Generar nuevas series con jitter (una por cada serie original)
            datos_aumentados_jittering = np.zeros((num_filas, num_columnas, profundidad))
            etiquetas_aumentadas_jittering = np.zeros((num_filas,))
            
            for i in range(num_filas):
                for j in range(num_columnas):
                    # Extraemos la serie temporal de longitud 250
                    serie = X_train[i, j, :]
                    nueva_serie = jitter(serie, 0.01)          # Añadir ruido gaussiano a la serie temporal
                    datos_aumentados_jittering[i,j,:] = nueva_serie
                    etiquetas_aumentadas_jittering[i] = y_train[i]  # Mantener la misma etiqueta
            
            # X_train = np.concatenate((X_train, datos_aumentados_jittering), axis=0)      # X_train original + X_train aumentado
            # y_train = np.concatenate((y_train, etiquetas_aumentadas_jittering), axis=0)  # y_train original + y_train aumentado
            
            
            # 2.- Magnitude Warping
            # ---------------------------
            # Generar nuevas series con Magnitude Warping (una por cada serie original)
            datos_aumentados_magnitude_warping = np.zeros((num_filas, num_columnas, profundidad))
            etiquetas_aumentadas_magnitude_warping = np.zeros((num_filas,))
            for i in range(num_filas):
                for j in range(num_columnas):
                    # Extraemos la serie temporal de longitud 250
                    serie = X_train[i, j, :]
                    nueva_serie = magnitude_warp(serie, 0.03)          
                    datos_aumentados_magnitude_warping[i,j,:] = nueva_serie
                    etiquetas_aumentadas_magnitude_warping[i] = y_train[i]  # Mantener la misma etiqueta
            
            # X_train = np.concatenate((X_train, datos_aumentados_jittering, datos_aumentados_magnitude_warping), axis=0)          # X_train original + X_train aumentado
            # y_train = np.concatenate((y_train, etiquetas_aumentadas_jittering, etiquetas_aumentadas_magnitude_warping), axis=0)  # y_train original + y_train aumentado
            
            
            # 3.- Shifting
            # ---------------------------
            # Generar nuevas series con Shifting (una por cada serie original)
            datos_aumentados_shifting = np.zeros((num_filas, num_columnas, profundidad))
            etiquetas_aumentadas_shifting = np.zeros((num_filas,))
            for i in range(num_filas):
                for j in range(num_columnas):
                    # Extraemos la serie temporal de longitud 250
                    serie = X_train[i, j, :]
                    nueva_serie = shift(serie, 0.03)       
                    datos_aumentados_shifting[i,j,:] = nueva_serie
                    etiquetas_aumentadas_shifting[i] = y_train[i]  # Mantener la misma etiqueta
                    
            # X_train = np.concatenate((X_train, datos_aumentados_jittering, datos_aumentados_magnitude_warping, datos_aumentados_shifting), axis=0)              # X_train original + X_train aumentado
            # y_train = np.concatenate((y_train, etiquetas_aumentadas_jittering, etiquetas_aumentadas_magnitude_warping, etiquetas_aumentadas_shifting), axis=0)  # y_train original + y_train aumentado
            
            
            # 4.- Time Warping
            # ---------------------------
            # Generar nuevas series con Time Warping (una por cada serie original)
            datos_aumentados_time_warping = np.zeros((num_filas, num_columnas, profundidad))
            etiquetas_aumentadas_time_warping = np.zeros((num_filas,))
            for i in range(num_filas):
                for j in range(num_columnas):
                    # Extraemos la serie temporal de longitud 250
                    serie = X_train[i, j, :]
                    nueva_serie = shift(serie, 0.03)       
                    datos_aumentados_time_warping[i,j,:] = nueva_serie
                    etiquetas_aumentadas_time_warping[i] = y_train[i]  # Mantener la misma etiqueta
            
            X_train = np.concatenate((X_train, datos_aumentados_jittering, datos_aumentados_magnitude_warping, datos_aumentados_shifting, datos_aumentados_time_warping), axis=0)                  # X_train original + X_train aumentado
            y_train = np.concatenate((y_train, etiquetas_aumentadas_jittering, etiquetas_aumentadas_magnitude_warping, etiquetas_aumentadas_shifting, etiquetas_aumentadas_time_warping), axis=0)  # y_train original + y_train aumentado
        
        self.X_train = X_train
        self.y_train = y_train
        try:
            self.X_validation = X_validation
            self.y_validation = y_validation
            self.X_test = X_test
            self.y_test = y_test
        except:
            print("Not enough data for validation and/or test.")                                    
        
        # Guardar el LabelEncoder después de ajustarlo
        joblib.dump(label_encoder, label_encoder_path)
        