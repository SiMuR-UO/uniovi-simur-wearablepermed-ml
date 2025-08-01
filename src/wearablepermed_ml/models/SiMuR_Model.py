import math
import os
import joblib # Librería empleada para guardar y cargar los modelos Random Forests
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras
# from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

#import _spectral_features_calculator

# Librerías necesarias para implementar el algoritmo de fusión sensorial ahsr
# import ahrs 
# from ahrs.filters import Madgwick

class SiMuRModel_ESANN(object):
    def __init__(self, data, params: dict, **kwargs) -> None:
        
        #############################################################################
        # Aquí se tratan los parámetros del modelo. Esto es necesario porque estos modelos contienen muchos hiperparámetros
        
        # - Hiperparámetros asociados a las opciones de entrenamiento de la CNN
        self.optimizador = params.get("optimizer", "adam")                # especifica el optimizador a utilizar durante el entrenamiento
        self.tamanho_minilote = params.get("miniBatchSize", 10)           # especifica el tamaño del mini-lote
        self.tasa_aprendizaje = params.get("lr", 0.01)                     # especifica el learning-rate empleado durante el entrenamiento
        
        # - Hiperparámetros asociados a la arquitectura de la red CNN
        self.N_capas = params.get("N_capas", 2)                           # especifica el número de capas ocultas de la red
        self.activacion_capas_ocultas = params.get("activation", "relu")  # especifica la función de activación asociada las neuronas de las capas ocultas
        self.numero_filtros = params.get("numFilters", 12)                # especifica el número de filtros utilizados en las capas ocultas de la red
        self.tamanho_filtro = params.get("filterSize", 7)                 # especifica el tamaño de los filtros de las capas ocultas
        
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
        self.X_validation = data.X_validation        
        self.X_test  = data.X_test
        
        self.y_train = data.y_train
        self.y_validation  = data.y_validation
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
        
        # self.numFeatures = 18   # especifica el número de características
        # self.numClasses = 17    # especifica el número de clases
        self.numClasses = int((max(self.y_train)+1)[0])    # especifica el número de clases
        # self.filterSize = 5     # especifica el tamaño del filtro
        # self.numFilters = 16    # especifica el número de filtros
        # self.miniBatchSize = 27 # especifica el tamaño del lote mini
        # self.max_epochs = 20    # especifica el número máximo de épocas

        # if (self.X_train).shape[1]==12:
        #     dimension_de_entrada = (12, 250)
        # elif (self.X_train).shape[1]==6:
        dimension_de_entrada = (6, 250)
            
        model = models.Sequential([
            layers.InputLayer(input_shape=dimension_de_entrada),
            layers.Conv1D(self.numero_filtros, self.tamanho_filtro, padding="causal", activation=self.activacion_capas_ocultas),
            layers.LayerNormalization(),
            layers.Conv1D(2*self.numero_filtros, self.tamanho_filtro, padding="causal", activation=self.activacion_capas_ocultas),
            layers.LayerNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
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
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_validation, self.y_validation),
                                 batch_size=self.tamanho_minilote,
                                 epochs=100,
                                 verbose=1,
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
        
    def store(self, model_id, path):
        # Método para guardar los pesos en path
        # Serialize weights to HDF5
        path = os.path.join(path, model_id + ".weights.h5")
        
        self.model.save_weights(path)
        print("Saved model to disk")

        return None
    
    def load(self, model_id, path):
        # Método para cargar los pesos desde el path indicado
        path = os.path.join(path, model_id + ".weights.h5")

        self.model.load_weights(path)
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
        self.X_validation = data.X_validation        
        self.X_test  = data.X_test        

        self.y_train = data.y_train
        self.y_validation = data.y_validation
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
        #self.numFeatures = 6      # especifica el número de características
        # self.numClasses = 17      # especifica el número de clases
        self.numClasses = int((max(self.y_train)+1)[0])    # especifica el número de clases
        # self.filterSize = 5     # especifica el tamaño del filtro
        # self.numFilters = 16    # especifica el número de filtros
        # self.miniBatchSize = 27 # especifica el tamaño del lote mini
        # self.max_epochs = 20    # especifica el número máximo de épocas
        
        #if (self.X_train).shape[1]==12:
        #    dimension_de_entrada = (12, 250)
        #elif (self.X_train).shape[1]==6:
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
        outputs = layers.Dense(self.numClasses, activation='softmax')(x)   # 18 clases de actividad física inicialmente

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
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_validation, self.y_validation),
                                 batch_size=self.tamanho_minilote,
                                 epochs=100,
                                 verbose=1,
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

    def predict(self, X):
        # Método para predecir una o varias muestras.
        # El código puede variar dependiendo del modelo
        return self.model.predict(X)
        
    def store(self, model_id, path):
        # Método para guardar los pesos en path
        # Serialize weights to HDF5
        path = os.path.join(path, model_id + ".weights.h5")

        self.model.save_weights(path)
        print("Saved model to disk")

        return None
    
    def load(self, model_id, path):
        # Método para cargar los pesos desde el path indicado
        path = os.path.join(path, model_id + ".weights.h5")

        self.model.load_weights(path)
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
        #     - d".h5"ain = data.y_train
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

    def predict(self, X):
        # Método para predecir una o varias muestras.
        # El código puede variar dependiendo del modelo
        
        return self.model.predict(X)
        
    def store(self, model_id, path):
        path = os.path.join(path, model_id + ".pkl")

        # Método para guardar el modelo Random Forest en formato '.pkl'
        joblib.dump(self.model, path)
        print("Saved model to disk")
        
        return None
    
    def load(self, model_id, path):
        path = os.path.join(path, model_id + ".pkl")

        # Método para cargar el modelo Random Forest desde el path indicado
        self.model = joblib.load(path)
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