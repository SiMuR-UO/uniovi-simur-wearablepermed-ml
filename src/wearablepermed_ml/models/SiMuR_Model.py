import math
import os
import joblib # Librería empleada para guardar y cargar los modelos Random Forests
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import keras
# from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb

from tensorflow.keras import layers, models, optimizers


#import _spectral_features_calculator

# Librerías necesarias para implementar el algoritmo de fusión sensorial ahsr
# import ahrs 
# from ahrs.filters import Madgwick

class SiMuRModel_ESANN(object):
    def __init__(self, data, params: dict, **kwargs) -> None:
        
        #############################################################################
        # Aquí se tratan los parámetros del modelo. Esto es necesario porque estos modelos contienen muchos hiperparámetros
        
        # - Hiperparámetros asociados a las opciones de entrenamiento de la CNN
        self.optimizador = params.get("optimizador", "adam")                # especifica el optimizador a utilizar durante el entrenamiento
        self.tamanho_minilote = params.get("tamanho_minilote", 10)           # especifica el tamaño del mini-lote
        self.tasa_aprendizaje = params.get("tasa_aprendizaje", 0.01)                     # especifica el learning-rate empleado durante el entrenamiento
        
        # - Hiperparámetros asociados a la arquitectura de la red CNN
        self.N_capas = params.get("N_capas", 2)                           # especifica el número de capas ocultas de la red
        self.activacion_capas_ocultas = params.get("funcion_activacion", "relu")  # especifica la función de activación asociada las neuronas de las capas ocultas
        self.numero_filtros = params.get("numero_filtros", 12)                # especifica el número de filtros utilizados en las capas ocultas de la red
        self.tamanho_filtro = params.get("tamanho_filtro", 7)                 # especifica el tamaño de los filtros de las capas ocultas
        
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
        
        try:
            self.X_validation = data.X_validation 
        except:
            print("Not enough data for validation.")
            self.X_validation = None 
            
        try:      
            self.X_test = data.X_test
        except:
            print("Not enough data for test.")
            self.X_test = None
        
        self.y_train = data.y_train
        
        try:
            self.y_validation = data.y_validation
        except:
            print("Not enough data for validation.")
            self.y_validation = None
            
        try:
            self.y_test = data.y_test
        except:
            print("Not enough data for test.")
            self.y_test = None

        #############################################################################
        # También se crea el modelo. Si es una red aquí se define el grafo. 
        # La creación del modelo se encapsula en la función "create_model"
        # Ejemplo de lectura de parámetros:
        #    param1 = params.get("N_capas", 3)

        self.model = self.create_model() 

        #############################################################################

    def create_model(self):
        # Aquí se define la red, SVC, árbol, etc.
        self.numClasses = int((max(self.y_train)+1)[0])    # especifica el número de clases

        if (self.X_train).shape[1]==6:
            dimension_de_entrada = (6, 250)
        elif (self.X_train).shape[1]==3:
            dimension_de_entrada = (3, 250)
        
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=dimension_de_entrada))

        # Añadir capas convolucionales según N_capas
        for i in range(self.N_capas):
            filtros = self.numero_filtros
            model.add(layers.Conv1D(filtros, self.tamanho_filtro, padding="causal", activation=self.activacion_capas_ocultas))
            model.add(layers.LayerNormalization())
            model.add(layers.Dropout(0.6))

        # Capas finales fijas
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(0.7))
        model.add(layers.Dense(self.numClasses, activation='softmax'))

        # Optimizadores
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
    
    def train(self, epochs):
        # Se lanza el entrenamiento de los modelos. El código para lanzar el entrenamiento depende mucho del modelo.        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ]
        
        # Verifica si X_validation e y_validation existen
        if self.X_validation is not None and self.y_validation is not None:
            history = self.model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_validation, self.y_validation),
                batch_size=self.tamanho_minilote,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                self.X_train,
                self.y_train,
                batch_size=self.tamanho_minilote,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks
            )
        
        if self.X_test is not None and self.y_test is not None:
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
        print("Saved model to disk.")

        return None
    
    def load(self, model_id, path):
        # Método para cargar los pesos desde el path indicado
        path = os.path.join(path, model_id + ".weights.h5")

        self.model.load_weights(path)
        print("Loaded model from disk.")

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
    def __init__(self, data, params: dict):
        # -----------------------------
        # Hiperparámetros de entrenamiento
        self.optimizador = params.get("optimizador", "adam")
        self.tasa_aprendizaje = params.get("tasa_aprendizaje", 5e-3)  
        self.tamanho_minilote = params.get("tamanho_minilote", 4)  # batch pequeño

        # -----------------------------
        # Arquitectura CNN
        self.N_capas = params.get("N_capas", 3)  # usar 3–4 etapas
        self.funcion_activacion = params.get("funcion_activacion", "relu")
        self.numero_filtros = params.get("numero_filtros", 64)
        self.tamanho_filtro = params.get("tamanho_filtro", 3)
        self.num_resblocks = params.get("num_resblocks", 1)

        # -----------------------------
        # Datos
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_validation = getattr(data, "X_validation", None)
        self.y_validation = getattr(data, "y_validation", None)
        self.X_test = getattr(data, "X_test", None)
        self.y_test = getattr(data, "y_test", None)

        # -----------------------------
        # Input shape y clases
        if (self.X_train).shape[1]==6:
            self.input_shape = (6, 250)
        elif (self.X_train).shape[1]==3:
            self.input_shape = (3, 250)
        # self.input_shape = (6,250)  # (6, 250)
        self.num_classes = int(np.max(self.y_train) + 1)

        # -----------------------------
        # Crear modelo
        self.model = self.create_model()

    # -----------------------------
    # Bloque residual
    def ResBlock(self, x, filtros, kernel_size, activation='relu'):
        shortcut = x
        x = layers.Conv1D(filtros, kernel_size, padding='same', activation=activation)(x)
        x = layers.Conv1D(filtros, kernel_size, padding='same')(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation(activation)(x)
        x = layers.BatchNormalization()(x)
        return x

    # -----------------------------
    # Crear modelo
    def create_model(self):
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        filtros = self.numero_filtros  # inicializamos filtros
        for i in range(self.N_capas):
            # Reducir dimensionalidad temporal en cada etapa excepto la última
            stride = 2 if i < self.N_capas - 1 else 1
            x = layers.Conv1D(filtros, self.tamanho_filtro, strides=stride,
                            padding='same', activation=self.funcion_activacion)(x)
            
            # 1 ResBlock por etapa para no saturar memoria
            for _ in range(self.num_resblocks):
                x = self.ResBlock(x, filtros, self.tamanho_filtro, self.funcion_activacion)
            
            # Solo duplicar filtros si no es la última capa
            if i < self.N_capas - 1:
                filtros *= 2

        # Global average pooling para reducir dimensionalidad
        x = layers.GlobalAveragePooling1D()(x)

        # Dropout para regularización
        x = layers.Dropout(0.2)(x)  # menos agresivo que 0.5 para batch pequeño

        # Dense intermedio más pequeño para memoria limitada
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Capa de salida
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        # Optimizer
        if self.optimizador.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.tasa_aprendizaje)
        elif self.optimizador.lower() == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.tasa_aprendizaje)
        elif self.optimizador.lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.tasa_aprendizaje)
        else:
            raise ValueError(f"Optimizador {self.optimizador} no soportado")

        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        return model
    
    def train(self, epochs):
        # Se lanza el entrenamiento de los modelos. El código para lanzar el entrenamiento depende mucho del modelo.        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ]
        
        # Verifica si X_validation e y_validation existen
        if self.X_validation is not None and self.y_validation is not None:
            history = self.model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_validation, self.y_validation),
                batch_size=self.tamanho_minilote,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                self.X_train,
                self.y_train,
                batch_size=self.tamanho_minilote,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks
            )
        
        if self.X_test is not None and self.y_test is not None:
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
        print("Saved model to disk.")

        return None
    
    def load(self, model_id, path):
        # Método para cargar los pesos desde el path indicado
        path = os.path.join(path, model_id + ".weights.h5")

        self.model.load_weights(path)
        print("Loaded model from disk.")
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
        self.n_estimators = params.get("n_estimators", 1000)
        self.max_depth = params.get("max_depth", 10)
        self.min_samples_split = params.get("min_samples_split", 3)
        self.min_samples_leaf = params.get("min_samples_leaf", 2)
        self.max_features = params.get("max_features", "auto")
        
        
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
        model = BalancedRandomForestClassifier(n_estimators=self.n_estimators, 
                                               n_jobs=-1,                         # n_jobs=-1 utiliza todos los núcleos disponibles para acelerar el entrenamiento
                                               verbose=1, 
                                               max_features=self.max_features, 
                                               max_depth=self.max_depth,
                                               min_samples_split=self.min_samples_split,
                                               min_samples_leaf=self.min_samples_leaf)  
        
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
        
    def store(self, model_id, path, run_index):
        path = os.path.join(path, model_id + "_" + run_index + ".pkl")

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


# Implementación del modelo XGBoost
class SiMuRModel_XGBoost(object):
    def __init__(self, data, params: dict, **kwargs) -> None:
            
            #############################################################################
            # Aquí se tratan los parámetros del modelo. Esto es necesario porque estos modelos contienen muchos hiperparámetros
            self.num_boost_round = params.get("num_boost_round", 1000)    # Número de estimadores
            self.max_depth = params.get("max_depth", 10)                  # Profundidad máxima
            self.learning_rate = params.get("learning_rate", 0.05)        # Tasa de aprendizaje
            self.subsample = params.get("subsample", 0.70)                # Fracción de muestras por árbol
            self.colsample_bytree = params.get("colsample_bytree", 0.80)  # Fracción de columnas por árbol
            self.gamma = params.get("gamma", 4.2)                         # Regularización mínima de pérdida
            self.min_child_weight = params.get("min_child_weight", 2)     # Peso mínimo de hijos
            self.reg_alpha = params.get("reg_alpha", 0.6)                 # L1 regularization
            self.reg_lambda = params.get("reg_lambda", 0.04)              # L2 regularization
                        
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
            
            self.X_train = data.X_train
        
            try:
                self.X_validation = data.X_validation 
            except:
                print("Not enough data for validation.")
                self.X_validation = None 
                
            try:      
                self.X_test = data.X_test
            except:
                print("Not enough data for test.")
                self.X_test = None
            
            self.y_train = data.y_train
            
            try:
                self.y_validation = data.y_validation
            except:
                print("Not enough data for validation.")
                self.y_validation = None
                
            try:
                self.y_test = data.y_test
            except:
                print("Not enough data for test.")
                self.y_test = None
            
            #############################################################################

            # También se crea el modelo. Si es una red aquí se define el grafo. 
            # La creación del modelo se encapsula en la función "create_model"
            # Ejemplo de lectura de parámetros:
            #    param1 = params.get("N_capas", 3)

            self.model = self.create_model()

            #############################################################################   
    
    def create_model(self):
        # Creamos el modelo de XGBoost, el cual se entrenará con las mismas características definidas para el Random Forest
        model = xgb.XGBClassifier(use_label_encoder=False, 
                                  num_boost_round=self.num_boost_round,   # Como parámetros indicamos: el número de estimadores
                                  max_depth=self.max_depth,               # Profundidad máxima
                                  learning_rate=self.learning_rate,       # Tasa de aprendizaje
                                  subsample=self.subsample,               # Fracción de muestras por árbol
                                  colsample_bytree=self.colsample_bytree, # Fracción de columnas por árbol
                                  gamma=self.gamma,                       # Regularización mínima de pérdida
                                  min_child_weight=self.min_child_weight, # Peso mínimo de hijos
                                  reg_alpha=self.reg_alpha,               # L1 regularization
                                  reg_lambda=self.reg_lambda,             # L2 regularization
                                  eval_metric='mlogloss',
                                  tree_method="gpu_hist",     # GPU
                                  predictor="gpu_predictor",   # fuerza GPU
                                  random_state=None  # o simplemente no fijar semilla
                                  )  
        return model
    
    def train(self):
        # --- Conversión a DMatrix ---
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)            # DMatrix de entrenamiento
        dval = xgb.DMatrix(self.X_validation, label=self.y_validation)    # DMatrix de validación
        # --- Parámetros del modelo ---
        params = {
            "objective": "multi:softprob",                                # Salida probabilidades multiclase
            "num_class": len(np.unique(self.y_train)),                    # Número de clases
            "eval_metric": "mlogloss",                                    # Métrica log-loss multiclase
            "max_depth": self.max_depth,                                  # Profundidad máxima
            "learning_rate":self.learning_rate,                           # Tasa de aprendizaje
            "subsample": self.subsample,                                  # Fracción de muestras por árbol
            "colsample_bytree": self.colsample_bytree,                    # Fracción de columnas por árbol
            "gamma": self.gamma,                                          # Regularización mínima de pérdida
            "min_child_weight": self.min_child_weight,                    # Peso mínimo de hijos
            "reg_alpha": self.reg_alpha,                                  # L1 regularization
            "reg_lambda": self.reg_lambda,                                 # L2 regularization
            "seed": np.random.randint(0, 1000000)  # Semilla aleatoria en cada run
        }
        # --- Lista de evaluaciones ---
        evals = [(dtrain, "train"), (dval, "validation")]                 # Conjuntos de evaluación
        # --- Entrenamiento con early stopping ---
        self.model = xgb.train(
            params=params,                               # Parámetros
            dtrain=dtrain,                               # Datos de entrenamiento
            num_boost_round=self.num_boost_round,        # Número de estimadores (rondas) en XGBoost
            evals=evals,                                 # Conjuntos de validación
            early_stopping_rounds=20                     # Parada temprana (early-stopping)
        )
        # --- Predicciones ---
        y_predicted_probability = self.model.predict(dval)                     # Probabilidades predichas
        self.y_validation_est = np.argmax(y_predicted_probability, axis=1)     # Clase con mayor probabilidad
        # --- Métricas ---
        self.validationMetrics = [
            accuracy_score(self.y_validation, self.y_validation_est),             # Exactitud
            f1_score(self.y_validation, self.y_validation_est, average='micro')   # F1 micro
        ]
      
        
    def predict(self, X):
        dmatrix = xgb.DMatrix(X)               # Convierte numpy.ndarray a DMatrix
        y_pred_proba = self.model.predict(dmatrix)  # Obtiene probabilidades o scores

        # Para clasificación multiclase (softprob), devuelve la clase con mayor probabilidad
        return np.argmax(y_pred_proba, axis=1)

        
    def store(self, model_id, path):
        path = os.path.join(path, model_id + ".pkl")
        # Método para guardar el modelo Random Forest en formato '.pkl'
        joblib.dump(self.model, path)
        print("Saved model to disk.")
        
        return None
    
    
    def load(self, model_id, path):
        path = os.path.join(path, model_id + ".pkl")
        # Método para cargar el modelo Random Forest desde el path indicado
        self.model = joblib.load(path)
        print("Loaded model from disk.")
        
        return None
    
    
    ##########   MÉTODOS DE LAS CLASES    ##########
    # Estos métodos se pueden llamar sin instar un objeto de la clase
    # Ej.: import model; model.get_model_type()
    
    @classmethod
    def get_model_type(cls):
        return "XGBoost"   # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-learn, etc.
    
    @classmethod
    def get_model_name(cls):
        return "SiMuR" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def get_model_Obj(cls):
        return SiMuRModel_XGBoost


##########################################
# Unit testing
##########################################
if __name__ == "__main__":
    # Este código solo se ejecuta si el script de ejecución principal es BaseModel.py:
    #   run BaseModel.py
    
    # Aquí se puede escribir un código de prueba para probar por separado     
    pass