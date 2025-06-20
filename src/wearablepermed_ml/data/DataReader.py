# En este script se preprocesan los datos.

# Se normaliza, limpian , filtran, etc.

# El resultado puede ser una clase o un dictionario que contenga:

#         data.X_train
#         data.y_train
#         data.X_test
#         data.y_test


import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

WINDOW_CONCATENATED_DATA = "arr_0"
WINDOW_ALL_LABELS = "arr_1"

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

    # Interpolamos el vector original según la distorsión
    new_indices = np.interp(np.linspace(0, 1, n), time_warp, np.linspace(0, 1, n))
    X_new = np.interp(new_indices, np.linspace(0, 1, n), X)

    return X_new

class DataReader(object):
    def __init__(self, p_train, file_path, label_encoder_path):        
        self.p_train = p_train
        
        stack_de_datos_y_etiquetas_PMP_tot = np.load(file_path)
        datos_input = stack_de_datos_y_etiquetas_PMP_tot[WINDOW_CONCATENATED_DATA]
        etiquetas_output = stack_de_datos_y_etiquetas_PMP_tot[WINDOW_ALL_LABELS]

        # X data
        X = datos_input
        
        # y data
        # Codificación numérica de las etiquetas para cada muestra de datos
        # Crear el codificador de etiquetas
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(etiquetas_output)
        
        # Split train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=p_train, random_state=42)
        
        # --------------------------------------------------------------------------------------------------
        # Realizamos el aumento de datos en el conjunto de entrenamiento. En el conjunto de test mantenemos
        # los datos origifile_pathnales:
        num_filas = X_train.shape[0]     # 27190
        num_columnas = X_train.shape[1]  # 12
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
        self.X_test = X_test
        self.y_test = y_test
        
        # Guardar el LabelEncoder después de ajustarlo
        joblib.dump(label_encoder, label_encoder_path)
        