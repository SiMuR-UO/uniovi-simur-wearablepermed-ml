import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras
# from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import joblib # Librería empleada para guardar y cargar los modelos Random Forests

import funciones_caracteristicas_RandomForest

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

import math

# Librerías necesarias para implementar el algoritmo de fusión sensorial ahsr
# import ahrs
# from ahrs.filters import Madgwick


class SiMuRModel_ESANN(object):
    def __init__(self, data, params: dict, **kwargs) -> None:
        
        #############################################################################
        # Aquí se tratan los parámetros del modelo. Esto es necesario porque estos modelos contienen muchos hiperparámetros
        
        # - Hiperparámetros asociados a las opciones de entrenamiento de la CNN
        self.optimizador = params.get("optimizer", "rmsprop")             # especifica el optimizador a utilizar durante el entrenamiento
        self.tamanho_minilote = params.get("miniBatchSize", 10)           # especifica el tamaño del mini-lote
        self.tasa_aprendizaje = params.get("lr", 0.0004549)  # especifica el learning-rate empleado durante el entrenamiento
        
        # - Hiperparámetros asociados a la arquitectura de la red CNN
        self.N_capas = params.get("N_capas", 2)                           # especifica el número de capas ocultas de la red
        self.activacion_capas_ocultas = params.get("activation", "relu")  # especifica la función de activación asociada las neuronas de las capas ocultas
        self.numero_filtros = params.get("numFilters", 12)                # especifica el número de filtros utilizados en las capas ocultas de la red
        self.tamanho_filtro = params.get("filterSize", 7)                 # especifica el tamaño de los filtros de las capas ocultas
        # self.dimension_de_entrada = params.get()
        
        self.testMetrics = []
        self.metrics = [accuracy_score, f1_score]
        #############################################################################
        # Los datos de entrenamiento vienen en el parametro data:
        #     - Vienen pre-procesados.
        #     - data suele ser un objeto o diccionario con:
        #         data.X_Train
        #         data.Y_Train
        #         data.X_Test
        #         data.Y_Test
        # El formato del objeto Data puede variar de aplicación en aplicación
        
        self.X_train = data.X_train
        self.X_test  = data.X_test
        
        self.y_train = data.y_train
        self.y_test  = data.y_test

        #############################################################################

        # También se crea el modelo. Si es una red aquí se define el grafo. 
        # La creación del modelo se encapsula en la función "create_model"
        # Ejemplo de lectura de parámetros:
        #    param1 = params.get("N_capas", 3)

        self.model = self.create_model() 

        #############################################################################

    def create_model(self):
        # Aquí se define la red, SVC, árbol, etc.
        
        self.numFeatures = 12   # especifica el número de características
        self.numClasses = 18    # especifica el número de clases
        # self.filterSize = 5     # especifica el tamaño del filtro
        # self.numFilters = 16    # especifica el número de filtros
        
        # self.miniBatchSize = 27 # especifica el tamaño del lote mini
        # self.max_epochs = 20    # especifica el número máximo de épocas
        if (self.X_train).shape[1]==12:
            dimension_de_entrada = (12, 250)
        elif (self.X_train).shape[1]==6:
            dimension_de_entrada = (6, 250)
            
        model = models.Sequential([
            layers.InputLayer(input_shape=dimension_de_entrada),
            layers.Conv1D(self.numero_filtros, self.tamanho_filtro, padding="causal", activation=self.activacion_capas_ocultas),
            layers.LayerNormalization(),
            layers.Conv1D(2*self.numero_filtros, self.tamanho_filtro, padding="causal", activation=self.activacion_capas_ocultas),
            layers.LayerNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.3),
            layers.Dense(self.numClasses, activation='softmax')
        ])

        if self.optimizador == "adam":
            optimizer_hyperparameter = tf.keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje)
        elif self.optimizador == 'rmsprop':
            optimizer_hyperparameter = tf.keras.optimizers.RMSprop(learning_rate=self.tasa_aprendizaje)
        elif self.optimizador == 'SGD':
            optimizer_hyperparameter = tf.keras.optimizers.SGD(learning_rate=self.tasa_aprendizaje)
        else:
            raise
        
        model.compile(optimizer=optimizer_hyperparameter,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model
    
    def train(self):
        # Se lanza el entrenamiento de los modelos. El código para lanzar el entrenamiento depende mucho del modelo.  
        history = self.model.fit(self.X_train, self.y_train,
                                 batch_size=self.tamanho_minilote,
                                 epochs=100,
                                 verbose=1,
                                 validation_split = 0.3,  # porcentaje de los datos de train que se utilizarán como
                                                             # dataset de validación en cada época
                                 callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
                                 )
        
        # Cuando acaba el entrenamiento y obtenemos los pesos óptimos, las métricas de error para los datos de test son calculadas.
        self.y_test_est = self.predict(self.X_test)
        self.y_test_est = np.argmax(self.y_test_est, axis=1)  # Trabajamos con clasificación multicategoría
        
        # y_test_est_float_round = np.around(self.y_test_est)        # Redondear vector de tipo float (codificado en one_hot)
        # y_test_est_int_round = y_test_est_float_round.astype(int)  # Obtención de vector de tipo int
        # self.y_test_est = y_test_est_int_round                     # Asignación del atributo y_test_est

        self.testMetrics = [accuracy_score(self.y_test, self.y_test_est),
                            f1_score(self.y_test, self.y_test_est, average='micro')] # REVISAR la opción 'average'

    def predict(self,X):
        # Método para predecir una o varias muestras.
        # El código puede variar dependiendo del modelo
        return self.model.predict(X)
        
    def store(self, modelID, path):
        # Método para guardar los pesos en path
        # Serialize weights to HDF5
        self.model.save_weights(path + modelID + ".h5")
        print("Saved model to disk")
        return None
    
    def load(self, modelID, path):
        # Método para cargar los pesos desde el path indicado
        self.model.load_weights(path + modelID + ".h5")
        print("Loaded model from disk")
        # Evaluate loaded model on test data
        self.model.compile(loss='sparse_categorical_crossentropy', 
                           optimizer=self.optimizador, 
                           metrics=['accuracy'])
        return None

    ##########   MÉTODOS DE LAS CLASES    ##########
    # Estos métodos se pueden llamar sin instar un objeto de la clase
    # Ej.: import model; model.get_model_type()
    
    @classmethod
    def get_model_type(cls):
        return "CNN"   # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-learn, etc.
    
    @classmethod
    def get_model_name(cls):
        return "SiMuR" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def get_model_Obj(cls):
        return SiMuRModel_ESANN


class SiMuRModel_CAPTURE24(object):
    def __init__(self, data, params: dict, **kwargs) -> None:
        
        #############################################################################
        # - Hiperparámetros asociados a las opciones de entrenamiento de la CNN
        self.optimizador = params.get("optimizer", "rmsprop")             # especifica el optimizador a utilizar durante el entrenamiento
        self.tamanho_minilote = params.get("miniBatchSize", 10)           # especifica el tamaño del mini-lote
        self.tasa_aprendizaje = params.get("lr", 0.00045493796608069996)  # especifica el learning-rate empleado durante el entrenamiento
        
        # - Hiperparámetros asociados a la arquitectura de la red CNN
        self.N_capas = params.get("N_capas", 6)                           # especifica el número de capas ocultas de la red
        self.activacion_capas_ocultas = params.get("activation", "relu")  # especifica la función de activación asociada las neuronas de las capas ocultas
        self.numero_filtros = params.get("numFilters", 12)                # especifica el número de filtros utilizados en las capas ocultas de la red
        self.tamanho_filtro = params.get("filterSize", 7)                 # especifica el tamaño de los filtros de las capas ocultas
        
        
        self.testMetrics = []
        self.metrics = [accuracy_score, f1_score]
        #############################################################################
        # Los datos de entrenamiento vienen en el parámetro data:
        #     - Vienen pre-procesados.
        #     - data suele ser un objeto o diccionario con:
        #         data.X_Train
        #         data.Y_Train
        #         data.X_Test
        #         data.Y_Test
        # El formato del objeto Data puede variar de aplicación en aplicación
        
        self.X_train = data.X_train
        self.X_test  = data.X_test
        
        self.y_train = data.y_train
        self.y_test  = data.y_test

        #############################################################################

        # También se crea el modelo. Si es una red aquí se define el grafo.
        # La creación del modelo se encapsula en la función "create_model"
        # Ejemplo de lectura de parámetros:
        #    param1 = params.get("N_capas", 3)

        self.model = self.create_model() 

        #############################################################################

    # Definimos la función de bloque residual (ResBlock)
    def ResBlock(self, x, filters, kernel_size, strides=1):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, kernel_size, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, shortcut])  # Conexión residual
        x = layers.ReLU()(x)
        return x
    
    
    def create_model(self):
        self.numFeatures = 6      # especifica el número de características
        self.numClasses = 18      # especifica el número de clases
        # self.filterSize = 5     # especifica el tamaño del filtro
        # self.numFilters = 16    # especifica el número de filtros
        
        # self.miniBatchSize = 27 # especifica el tamaño del lote mini
        # self.max_epochs = 20    # especifica el número máximo de épocas
        
        if (self.X_train).shape[1]==12:
            dimension_de_entrada = (12, 250)
        elif (self.X_train).shape[1]==6:
            dimension_de_entrada = (6, 250)
        
        # Entrada
        inputs = layers.Input(shape=dimension_de_entrada)

        # Primer bloque Conv(3, 128)
        x = layers.Conv1D(128, 3, strides=2, padding="same", activation="relu")(inputs)

        # Segundo bloque Conv(3, 128) y 3 x ResBlock(3, 128) / 2
        x = layers.Conv1D(128, 3, strides=1, padding="same", activation="relu")(x)
        x = self.ResBlock(x, 128, 3)  # ResBlock 1
        x = self.ResBlock(x, 128, 3)  # ResBlock 2
        x = self.ResBlock(x, 128, 3)  # ResBlock 3

        # Tercer bloque Conv(3, 256) y 3 x ResBlock(3, 256) / 2
        x = layers.Conv1D(256, 3, strides=1, padding="same", activation="relu")(x)
        x = self.ResBlock(x, 256, 3)  # ResBlock 1
        x = self.ResBlock(x, 256, 3)  # ResBlock 2
        x = self.ResBlock(x, 256, 3)  # ResBlock 3

        # Cuarto bloque Conv(3, 256) y 3 x ResBlock(3, 256) / 5
        x = layers.Conv1D(256, 3, strides=1, padding="same", activation="relu")(x)
        x = self.ResBlock(x, 256, 3)  # ResBlock 1
        x = self.ResBlock(x, 256, 3)  # ResBlock 2
        x = self.ResBlock(x, 256, 3)  # ResBlock 3

        # Quinto bloque Conv(3, 512) y 3 x ResBlock(3, 512) / 5
        x = layers.Conv1D(512, 3, strides=1, padding="same", activation="relu")(x)
        x = self.ResBlock(x, 512, 3)  # ResBlock 1
        x = self.ResBlock(x, 512, 3)  # ResBlock 2
        x = self.ResBlock(x, 512, 3)  # ResBlock 3

        # Sexto bloque Conv(3, 512) y 3 x ResBlock(3, 512) / 5
        x = layers.Conv1D(512, 3, strides=1, padding="same", activation="relu")(x)
        x = self.ResBlock(x, 512, 3)  # ResBlock 1
        x = self.ResBlock(x, 512, 3)  # ResBlock 2
        x = self.ResBlock(x, 512, 3)  # ResBlock 3

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Capa de Dropout
        x = layers.Dropout(0.5)(x)

        # Capa totalmente conectada
        x = layers.Dense(1024, activation='relu')(x)

        # Capa de salida
        outputs = layers.Dense(18, activation='softmax')(x)   # 18 clases de actividad física inicialmente

        # Definimos el modelo
        model_CNN_CAPTURE24 = models.Model(inputs, outputs)

        if self.optimizador == "adam":
            optimizer_hyperparameter = tf.keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje)
        elif self.optimizador == 'rmsprop':
            optimizer_hyperparameter = tf.keras.optimizers.RMSprop(learning_rate=self.tasa_aprendizaje)
        elif self.optimizador == 'SGD':
            optimizer_hyperparameter = tf.keras.optimizers.SGD(learning_rate=self.tasa_aprendizaje)
        else:
            raise
        
        model_CNN_CAPTURE24.compile(optimizer=optimizer_hyperparameter,
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])
        model_CNN_CAPTURE24.summary()
        return model_CNN_CAPTURE24
    
    def train(self):
        # Se lanza el entrenamiento de los modelos. El código para lanzar el entrenamiento depende mucho del modelo.  
        history = self.model.fit(self.X_train, self.y_train,
                                 batch_size=self.tamanho_minilote,
                                 epochs=100,
                                 verbose=1,
                                 validation_split = 0.3,  # porcentaje de los datos de train que se utilizarán como
                                                             # dataset de validación en cada época
                                 callbacks = [keras.callbacks.EarlyStopping(monitor='loss', patience=5)]
                                 )
        
        # Cuando acaba el entrenamiento y obtenemos los pesos óptimos, las métricas de error para los datos de test son calculadas.
        self.y_test_est = self.predict(self.X_test)
        self.y_test_est = np.argmax(self.y_test_est, axis=1)  # Trabajamos con clasificación multicategoría
        
        # y_test_est_float_round = np.around(self.y_test_est)        # Redondear vector de tipo float (codificado en one_hot)
        # y_test_est_int_round = y_test_est_float_round.astype(int)  # Obtención de vector de tipo int
        # self.y_test_est = y_test_est_int_round                     # Asignación del atributo y_test_est

        self.testMetrics = [accuracy_score(self.y_test, self.y_test_est),
                            f1_score(self.y_test, self.y_test_est, average='micro')] # REVISAR la opción 'average'

    def predict(self,X):
        # Método para predecir una o varias muestras.
        # El código puede variar dependiendo del modelo
        return self.model.predict(X)
        
    def store(self, modelID, path):
        # Método para guardar los pesos en path
        # Serialize weights to HDF5
        self.model.save_weights(path + modelID + ".h5")
        print("Saved model to disk")
        return None
    
    def load(self, modelID, path):
        # Método para cargar los pesos desde el path indicado
        self.model.load_weights(path + modelID + ".h5")
        print("Loaded model from disk")
        # Evaluate loaded model on test data
        self.model.compile(loss='sparse_categorical_crossentropy', 
                           optimizer=self.optimizador, 
                           metrics=['accuracy'])
        return None

    ##########   MÉTODOS DE LAS CLASES    ##########
    # Estos métodos se pueden llamar sin instar un objeto de la clase
    # Ej.: import model; model.get_model_type()
    
    @classmethod
    def get_model_type(cls):
        return "CNN"   # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-learn, etc.
    
    @classmethod
    def get_model_name(cls):
        return "SiMuR" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def get_model_Obj(cls):
        return SiMuRModel_CAPTURE24



class SiMuRModel_RandomForest(object):
    def __init__(self, data, params: dict, **kwargs) -> None:
        
        #############################################################################
        # Aquí se tratan los parámetros del modelo. Esto es necesario porque estos modelos contienen muchos hiperparámetros
        self.optimizador = params.get("n_estimators", 3000)
        
        self.testMetrics = []
        self.metrics = [accuracy_score, f1_score]
        #############################################################################
        # Los datos de entrenamiento vienen en el parametro data:
        #     - Vienen pre-procesados.
        #     - data suele ser un objeto o diccionario con:
        #         data.X_Train
        #         data.Y_Train
        #         data.X_Test
        #         data.Y_Test
        
        # Obtención de características de X_train
        # -----------------------------------------
        
        # ***************
        # 1.- Cuantiles *
        # ***************
        # El vector de características empleado en el entrenamiento del Random-Forest será:
        # [Mín, Máx, Mediana, Percentil 25,Percentil 75] para Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Acc, Gyr.
        # self.X_train = data.X_train
        minimos_train = np.quantile(data.X_train, 0, axis=2, keepdims=True)
        maximos_train = np.quantile(data.X_train, 1, axis=2, keepdims=True)
        medianas_train = np.quantile(data.X_train, 0.5, axis=2, keepdims=True)
        Percentil_25_train = np.quantile(data.X_train, 0.25, axis=2, keepdims=True)
        Percentil_75_train = np.quantile(data.X_train, 0.75, axis=2, keepdims=True)
        Matriz_de_cuantiles_train = np.hstack((minimos_train, maximos_train, medianas_train, Percentil_25_train, Percentil_75_train))
        Matriz_de_cuantiles_train = np.squeeze(Matriz_de_cuantiles_train, axis=2)
        
        # *********************************
        # 2.- Características espectrales *
        # *********************************
        # Inicializamos las matrices de resultados
        num_filas = (data.X_train).shape[0]  # 27190
        num_columnas = (data.X_train).shape[1]  # 12
        # f1_mat      = np.zeros((num_filas, num_columnas))
        # p1_mat      = np.zeros((num_filas, num_columnas))
        # f2_mat      = np.zeros((num_filas, num_columnas))
        # p2_mat      = np.zeros((num_filas, num_columnas))
        # entropy_mat = np.zeros((num_filas, num_columnas))
        
        matriz_resultados_armonicos = np.zeros((num_filas,30))   # 1 IMU
        # matriz_resultados_armonicos = np.zeros((num_filas,60))   # 2 IMUs
        # # Recorremos cada serie temporal y calculamos las características
        for i in range(num_filas):   
            armonicos_totales = np.zeros((6,5))    # 1 IMU
            # armonicos_totales = np.zeros((12,5))    # 2 IMUs
            for j in range(num_columnas):
                # Extraemos la serie temporal de longitud 250
                serie = data.X_train[i, j, :]
                # Calculamos las características espectrales
                resultado_armonicos,_ = funciones_caracteristicas_RandomForest.obtener_caracteristicas_espectrales(serie,25)
                # Guardamos los resultados en las matrices correspondientes
                # f1_mat[i, j]      = f1
                # p1_mat[i, j]      = p1
                # f2_mat[i, j]      = f2
                # p2_mat[i, j]      = p2
                # entropy_mat[i, j] = entropy
                
                armonicos_totales[j, :] = resultado_armonicos
            armonicos_totales_2 = np.reshape(armonicos_totales,(1,-1))
            matriz_resultados_armonicos[i,:] = armonicos_totales_2
        # self.X_train = np.hstack(Matriz_de_cuantiles_train, f1_mat, p1_mat, f2_mat, p2_mat, entropy_mat)
        # aux = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos))
        
        # *****************************************
        # 3.- Número de picos y prominencia media *
        # *****************************************
        matriz_resultados_numero_picos = np.zeros((num_filas,12))   # 1 IMUs
        # matriz_resultados_numero_picos = np.zeros((num_filas,24))   # 2 IMUs
        # # Recorremos cada serie temporal y calculamos los picos
        for i in range(num_filas):  
            picos_totales = np.zeros(6)         # 1 IMU
            prominencias_totales = np.zeros(6)  # 1 IMUs
            # picos_totales = np.zeros(12)      # 2 IMUs
            # prominencias_totales = np.zeros(12) # 2 IMUs
            for j in range(num_columnas):
                # Extraemos la serie temporal de longitud 250
                serie = data.X_train[i, j, :]
                # Calculamos las características espectrales
                indices_picos, propiedades_picos = find_peaks(serie, prominence=True)
                numero_picos=len(indices_picos)
                if numero_picos > 0:
                    # Si se detectaron picos, podemos proceder con el cálculo
                    prominencias_picos = propiedades_picos['prominences']
                    # Por ejemplo, calcular la mediana de la prominencia de los picos
                    prominencia_media = np.median(prominencias_picos)
                    #print(f"Mediana de prominencia: {prominencia_media}")
                else:
                    # prominencia_media = np.NaN
                    prominencia_media = 0
                
                #prominencias_picos = propiedades_picos['prominences']
                # Guardamos los resultados en las matrices correspondientes
                # f1_mat[i, j]      = f1
                # p1_mat[i, j]      = p1
                # f2_mat[i, j]      = f2
                # p2_mat[i, j]      = p2
                # entropy_mat[i, j] = entropy
                
                picos_totales[j] = numero_picos
                prominencias_totales[j] = prominencia_media
                
            picos_totales_2 = np.reshape(picos_totales,(1,-1))
            prominencias_totales_2 = np.reshape(prominencias_totales,(1,-1))
            matriz_resultados_numero_picos[i,:] = np.hstack((picos_totales_2, prominencias_totales_2))
        
        # *******************
        # 4.- Correlaciones *
        # *******************
        matriz_correlaciones = np.zeros((num_filas,15))  # 1 IMU
        # matriz_correlaciones = np.zeros((num_filas,66))  # 2 IMUs
        for i in range(num_filas):
            # Calcular la matriz de correlación entre las filas
            correlacion = np.corrcoef(data.X_train[i,:,:], rowvar=True)
            # Extraer la parte superior de la matriz sin la diagonal principal
            upper_triangle_values = correlacion[np.triu_indices_from(correlacion, k=1)]
            # print(upper_triangle_values)
            
            matriz_correlaciones[i,:] = upper_triangle_values
        #self.X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones))
        #print(self.X_train)
        
        # **************************************
        # 5.- Autocorrelación del acelerómetro *
        # **************************************
        matriz_resultados_autocorrelacion = np.zeros((num_filas, 1))
        # matriz_resultados_autocorrelacion = np.zeros((num_filas, 2))
        # Recorremos cada serie temporal y calculamos los picos
        for i in range(num_filas):
            serie = np.linalg.norm(data.X_train[i,0:3,:], axis=0)
            # serie_desplazada = np.pad(serie[-25], (25,), mode='constant', constant_values=0)
            serie_desplazada = np.empty_like(serie)
            serie_desplazada[:25] = 0
            serie_desplazada[25:] = serie[:-25]
             
            autocorrelacion_acc_IMU1 = np.corrcoef(serie, serie_desplazada)

            serie = np.linalg.norm(data.X_train[i,6:9,:], axis=0)
            serie_desplazada = np.empty_like(serie)
            serie_desplazada[:25] = 0
            serie_desplazada[25:] = serie[:-25]
            # serie_desplazada = np.pad(serie[:,-25], (25,0), mode='constant', constant_values=0)
            autocorrelacion_acc_IMU2 = np.corrcoef(serie, serie_desplazada)
            
            # modulo_acc_IMU1 = np.linalg.norm(data.X_train[i,0:3,:], axis=0)
            # modulo_acc_IMU2 = np.linalg.norm(data.X_train[i,6:9,:], axis=0)
            # autocorrelacion_acc_IMU2 = np.corrcoef(modulo_acc_IMU2, nlags=25)
            
            matriz_resultados_autocorrelacion[i,0] = autocorrelacion_acc_IMU1[0,1]
            # matriz_resultados_autocorrelacion[i,1] = autocorrelacion_acc_IMU2[0,1]
        
        # self.X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion))      
        
        # **************************************************
        # 6.- Componentes roll, pitch y yaw del movimiento *
        # **************************************************
        dt = 1/25      # Período de muestreo en [s]
        rolls_promedio = np.zeros((num_filas, 1))
        pitches_promedio = np.zeros((num_filas, 1))
        yaws_promedio = np.zeros((num_filas, 1))
        for i in range(num_filas):
            rolls = []
            pitches = []
            yaws = []
            # Extraemos las series temporales de longitud 250 muestras (acelerómetro y giroscopio)
            serie_acc_x = data.X_train[i, 0, :]
            serie_acc_y = data.X_train[i, 1, :]
            serie_acc_z = data.X_train[i, 2, :]
            serie_gyr_x = data.X_train[i, 3, :]
            serie_gyr_y = data.X_train[i, 4, :]
            serie_gyr_z = data.X_train[i, 5, :]
            
            yaw_acumulado = 0
            for j in range(len(serie_acc_x)):
                acc_x = serie_acc_x[j]
                acc_y = serie_acc_y[j]
                acc_z = serie_acc_z[j]
                gyr_x = serie_gyr_x[j]
                gyr_y = serie_gyr_y[j]
                gyr_z = serie_gyr_z[j]

                roll = math.atan2(acc_y, acc_z)                             # Roll: rotación alrededor del eje X
                pitch = math.atan2(-acc_x, math.sqrt(acc_y**2 + acc_z**2))  # Pitch: rotación alrededor del eje Y
                yaw = gyr_z * dt                                            # Integración simple para obtener el cambio de yaw
                yaw_acumulado += yaw                                        # Efecto acumulativo de la acción integral
                rolls.append(roll)
                pitches.append(pitch)
            yaws.append(yaw_acumulado)
            yaw_acumulado = 0
            
            rolls_promedio[i] = np.mean(rolls)
            pitches_promedio[i] = np.mean(pitches)
            yaws_promedio[i] = np.mean(yaws)
        
        self.X_train = np.hstack((Matriz_de_cuantiles_train, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion, rolls_promedio, pitches_promedio, yaws_promedio))    
        
        # Obtención de características de X_test.
        # -----------------------------------------
        # ***************
        # 1.- Cuantiles *
        # ***************
        # [Mín, Máx, Mediana, Percentil 25,Percentil 75] para Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Acc, Gyr.
        minimos_test = np.quantile(data.X_test, 0, axis=2, keepdims=True)
        maximos_test = np.quantile(data.X_test, 1, axis=2, keepdims=True)
        medianas_test = np.quantile(data.X_test, 0.5, axis=2, keepdims=True)
        Percentil_25_test = np.quantile(data.X_test, 0.25, axis=2, keepdims=True)
        Percentil_75_test = np.quantile(data.X_test, 0.75, axis=2, keepdims=True)
        Matriz_de_cuantiles_test = np.hstack((minimos_test, maximos_test, medianas_test, Percentil_25_test, Percentil_75_test))
        Matriz_de_cuantiles_test = np.squeeze(Matriz_de_cuantiles_test, axis=2)
        
        # *********************************
        # 2.- Características espectrales *
        # *********************************
        # Inicializamos las matrices de resultados
        num_filas = (data.X_test).shape[0]  # m ejemplos
        num_columnas = (data.X_test).shape[1]  # 12
        
        matriz_resultados_armonicos = np.zeros((num_filas,30))    # 1 IMU
        # matriz_resultados_armonicos = np.zeros((num_filas,60))    # 2 IMUs
        # Recorremos cada serie temporal y calculamos las características
        for i in range(num_filas):
            armonicos_totales = np.zeros((6,5))      # 1 IMU  
            # armonicos_totales = np.zeros((12,5))   # 2 IMUs
            for j in range(num_columnas):
                # Extraemos la serie temporal de longitud 250
                serie = data.X_train[i, j, :]
                # Calculamos las características espectrales
                resultado_armonicos,_ = funciones_caracteristicas_RandomForest.obtener_caracteristicas_espectrales(serie,25)
                armonicos_totales[j, :] = resultado_armonicos
            armonicos_totales_2 = np.reshape(armonicos_totales,(1,-1))
            matriz_resultados_armonicos[i,:] = armonicos_totales_2
        
        # *****************************************
        # 3.- Número de picos y prominencia media *
        # *****************************************
        matriz_resultados_numero_picos = np.zeros((num_filas,12))   # 1 IMU
        # matriz_resultados_numero_picos = np.zeros((num_filas,24))   # 2 IMUs
        # Recorremos cada serie temporal y calculamos los picos
        for i in range(num_filas):  
            picos_totales = np.zeros(6) # 1 IMU
            prominencias_totales = np.zeros(6) # 1 IMU
            # picos_totales = np.zeros(12) # 2 IMUs
            # prominencias_totales = np.zeros(12) # 2 IMUs
            for j in range(num_columnas):
                # Extraemos la serie temporal de longitud 250
                serie = data.X_train[i, j, :]
                # Calculamos las características espectrales
                indices_picos, propiedades_picos = find_peaks(serie, prominence=True)
                numero_picos=len(indices_picos)
                if numero_picos > 0:
                    # Si se detectaron picos, podemos proceder con el cálculo
                    prominencias_picos = propiedades_picos['prominences']
                    # Por ejemplo, calcular la mediana de la prominencia de los picos
                    prominencia_media = np.median(prominencias_picos)
                else:
                    # prominencia_media = np.NaN
                    prominencia_media = 0
                
                picos_totales[j] = numero_picos
                prominencias_totales[j] = prominencia_media
                
            picos_totales_2 = np.reshape(picos_totales,(1,-1))
            prominencias_totales_2 = np.reshape(prominencias_totales,(1,-1))
            matriz_resultados_numero_picos[i,:] = np.hstack((picos_totales_2, prominencias_totales_2))
        
        # *******************
        # 4.- Correlaciones *
        # *******************
        matriz_correlaciones = np.zeros((num_filas,15))  # 1 IMU
        # matriz_correlaciones = np.zeros((num_filas,66))  # 2 IMUs
        for i in range(num_filas):
            # Calcular la matriz de correlación entre las filas
            correlacion = np.corrcoef(data.X_train[i,:,:], rowvar=True)
            # Extraer la parte superior de la matriz sin la diagonal principal
            upper_triangle_values = correlacion[np.triu_indices_from(correlacion, k=1)]  
            matriz_correlaciones[i,:] = upper_triangle_values
        
        # **************************************
        # 5.- Autocorrelación del acelerómetro *
        # **************************************
        matriz_resultados_autocorrelacion = np.zeros((num_filas, 1))
        # matriz_resultados_autocorrelacion = np.zeros((num_filas, 2))
        # Recorremos cada serie temporal y calculamos los picos
        for i in range(num_filas):
            serie = np.linalg.norm(data.X_train[i,0:3,:], axis=0)
            # serie_desplazada = np.pad(serie[-25], (25,), mode='constant', constant_values=0)
            serie_desplazada = np.empty_like(serie)
            serie_desplazada[:25] = 0
            serie_desplazada[25:] = serie[:-25]
            
            autocorrelacion_acc_IMU1 = np.corrcoef(serie, serie_desplazada)

            serie = np.linalg.norm(data.X_train[i,6:9,:], axis=0)
            serie_desplazada = np.empty_like(serie)
            serie_desplazada[:25] = 0
            serie_desplazada[25:] = serie[:-25]
            # serie_desplazada = np.pad(serie[:,-25], (25,0), mode='constant', constant_values=0)
            autocorrelacion_acc_IMU2 = np.corrcoef(serie, serie_desplazada)
            
            matriz_resultados_autocorrelacion[i,0] = autocorrelacion_acc_IMU1[0,1]
            # matriz_resultados_autocorrelacion[i,1] = autocorrelacion_acc_IMU2[0,1]
        
        # self.X_test = np.hstack((Matriz_de_cuantiles_test, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion))
                
        # **************************************************
        # 6.- Componentes roll, pitch y yaw del movimiento *
        # **************************************************
        dt = 1/25      # Período de muestreo en [s]
        rolls_promedio = np.zeros((num_filas, 1))
        pitches_promedio = np.zeros((num_filas, 1))
        yaws_promedio = np.zeros((num_filas, 1))
        for i in range(num_filas):
            rolls = []
            pitches = []
            yaws = []
            # Extraemos las series temporales de longitud 250 muestras (acelerómetro y giroscopio)
            serie_acc_x = data.X_test[i, 0, :]
            serie_acc_y = data.X_test[i, 1, :]
            serie_acc_z = data.X_test[i, 2, :]
            serie_gyr_x = data.X_test[i, 3, :]
            serie_gyr_y = data.X_test[i, 4, :]
            serie_gyr_z = data.X_test[i, 5, :]
            
            yaw_acumulado = 0
            for j in range(len(serie_acc_x)):
                acc_x = serie_acc_x[j]
                acc_y = serie_acc_y[j]
                acc_z = serie_acc_z[j]
                gyr_x = serie_gyr_x[j]
                gyr_y = serie_gyr_y[j]
                gyr_z = serie_gyr_z[j]

                roll = math.atan2(acc_y, acc_z)                             # Roll: rotación alrededor del eje X
                pitch = math.atan2(-acc_x, math.sqrt(acc_y**2 + acc_z**2))  # Pitch: rotación alrededor del eje Y
                yaw = gyr_z * dt                                            # Integración simple para obtener el cambio de yaw
                yaw_acumulado += yaw                                        # Efecto acumulativo de la acción integral
                rolls.append(roll)
                pitches.append(pitch)
            yaws.append(yaw_acumulado)
            yaw_acumulado = 0
            
            rolls_promedio[i] = np.mean(rolls)
            pitches_promedio[i] = np.mean(pitches)
            yaws_promedio[i] = np.mean(yaws)
        
        self.X_test = np.hstack((Matriz_de_cuantiles_test, matriz_resultados_armonicos, matriz_resultados_numero_picos, matriz_correlaciones, matriz_resultados_autocorrelacion, rolls_promedio, pitches_promedio, yaws_promedio))
        
        self.y_train = data.y_train
        self.y_test  = data.y_test

        #############################################################################

        # También se crea el modelo. Si es una red aquí se define el grafo. 
        # La creación del modelo se encapsula en la función "create_model"
        # Ejemplo de lectura de parámetros:
        #    param1 = params.get("N_capas", 3)

        self.model = self.create_model()

        #############################################################################

    def create_model(self):
        # Creamos el modelo de Random Forest con 3000 árboles
        model = BalancedRandomForestClassifier(n_estimators=3000, random_state=42, n_jobs=-1, verbose=1, max_features=None, max_depth=10)  # n_jobs=-1 utiliza todos los núcleos disponibles para acelerar el entrenamiento
        return model
    
    def train(self):
        
        # Se lanza el entrenamiento de los modelos. El código para lanzar el entrenamiento depende mucho del modelo.  
        history = self.model.fit(self.X_train, self.y_train)
                                                                                                                                                     
        # Cuando acaba el entrenamiento y obtenemos los pesos óptimos, las métricas de error para los datos de test son calculadas.
        self.y_test_est = self.predict(self.X_test)
        
        y_test_est_float_round = np.around(self.y_test_est)        # Redondear vector de tipo float (codificado en one_hot)
        y_test_est_int_round = y_test_est_float_round.astype(int)  # Obtención de vector de tipo int
        self.y_test_est = y_test_est_int_round                     # Asignación del atributo y_test_est

        self.testMetrics = [accuracy_score(self.y_test, self.y_test_est),
                            f1_score(self.y_test, self.y_test_est, average='micro')] # REVISAR la opción 'average'

    def predict(self,X):
        # Método para predecir una o varias muestras.
        # El código puede variar dependiendo del modelo
        return self.model.predict(X)
        
    def store(self, modelID, path):
        # Método para guardar el modelo Random Forest en formato '.pkl'
        joblib.dump(self.model, path + modelID + '.pkl')
        print("Saved model to disk")
        return None
    
    def load(self, modelID, path):
        # Método para cargar el modelo Random Forest desde el path indicado
        self.model = joblib.load(path + modelID + '.pkl')
        print("Loaded model from disk")
        return None

    ##########   MÉTODOS DE LAS CLASES    ##########
    # Estos métodos se pueden llamar sin instar un objeto de la clase
    # Ej.: import model; model.get_model_type()
    
    @classmethod
    def get_model_type(cls):
        return "Balanced Random Forest"   # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-learn, etc.
    
    @classmethod
    def get_model_name(cls):
        return "SiMuR" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def get_model_Obj(cls):
        return SiMuRModel_RandomForest

##########################################
# Unit testing
##########################################


if __name__ == "__main__":
    # Este código solo se ejecuta si el script de ejecución principal es BaseModel.py:
    #   run BaseModel.py
    
    # Aquí se puede escribir un código de prueba para probar por separado 
    
    pass