<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-ml.svg?branch=main)](https://cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-ml)
[![ReadTheDocs](https://readthedocs.org/projects/uniovi-simur-wearablepermed-ml/badge/?version=latest)](https://uniovi-simur-wearablepermed-ml.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/uniovi-simur-wearablepermed-ml/main.svg)](https://coveralls.io/r/<USER>/uniovi-simur-wearablepermed-ml)
[![PyPI-Server](https://img.shields.io/pypi/v/uniovi-simur-wearablepermed-ml.svg)](https://pypi.org/project/uniovi-simur-wearablepermed-ml/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/uniovi-simur-wearablepermed-ml.svg)](https://anaconda.org/conda-forge/uniovi-simur-wearablepermed-ml)
[![Monthly Downloads](https://pepy.tech/badge/uniovi-simur-wearablepermed-ml/month)](https://pepy.tech/project/uniovi-simur-wearablepermed-ml)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/uniovi-simur-wearablepermed-ml)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# Description

> Uniovi Simur WearablePerMed Machine Learning.

*********************************************************************
* Pasos para lanzar el entrenamiento de un modelo de clasificación: *
*********************************************************************

1. Abrir el archivo "train_automatizado.py".

2. Entre las líneas 27 y 37, descomentar el modelo de clasificación que se desea entrenar. Comentamos
   el resto de modelos. Para una misma clase de modelo tenemos 3 posibilidades, en función de si
   empleamos datos de muslo, muñeca o muslo+muñeca durante el entrenamiento.

3. Ejecutar el archivo "train_automatizado.py". Este código ya empleará el resto de dependencias auxiliares:
   Datareader, modelGenerator, etc.

Como resultado se guardará el modelo entrenado en formato ".h5" (para las CNNs) o ".pkl" (para RandomForest y XGBoost).
Estos modelos pueden cargarse para estudiar resultados en la fase de test mediante el fichero "scriptResults_v2.ipynb".

Fragmento del pipeline:

**********************************           **********************************
* Stack de datos/características *  -------> *    Modelos de clasificación    *
**********************************           **********************************


-------------------------------------------------------
- Repositorio Machine Learning (desarrollosPMP_SiMuR) -
-------------------------------------------------------

Árbol de directorios:

/Raíz (desarrollosPMP_SiMuR)
|
…
|
|-- modelGenerator.py
|-- train_automatizado.py (PROGRAMA PRINCIPAL)
|-- Models
|	|
|	|-- SiMuRModel.py  (Implementación de las clases, junto con sus métodos, asociada a cada modelo)
|	|-- *.h5, *.pkl    (Se generan tras entrenar cada modelo)
|	|-- __init__.py
|
|-- scriptResults_v2.ipynb (Resultados en la etapa de test para los modelos entrenados)
|-- train_hyperparameter_searching_v2.py (Búsqueda de hiperparámetros óptimos de cada modelo, empleando
					  el algoritmo ASHA).


*******************************************
*       Contenido de SiMuRModel.py        *
*******************************************

1 clase para cada modelo de clasificación:
	class SiMuRModel_ESANN        (Red Neuronal Convolucional, CNN)
	class SiMuRModel_CAPTURE24    (CNN de arquitectura más compleja)
	class SiMuRModel_RandomForest (Balanced Random Forest)
	class SiMuRModel_XGBoost <---- Tengo que actualizarlo en GitHub, lo tengo local en mi PC.

Cada modelo para datos de:
	* thigh (muslo).
	* wrist (muñeca).
	* thigh + wrist (muslo + muñeca).

En "train_automatizado.py" se crea un objeto de cada clase y se lanza el entrenamiento del modelo
de clasificación.

<!-- pyscaffold-notes -->

## Scaffold your project from scratch

- **STEP01**: Install PyScaffold and pyscaffoldext-markdown extension

     - You can install PyScaffold and extensions globally in your systems but ins recomendes use a virtual environment:

          Craate a temp folder and use a virtual environment to install PyScaffold tool and scaffold your project. Later will copy the results under the final git folder and remove the last temporal one:

          ```
          $ mkdir temp
          $ cd temp
          $ python3 -m venv .venv
          $ source .venv/bin/activate
          $ pip install pyscaffold
          $ pip install pyscaffoldext-markdown
          $ putup --markdown uniovi-simur-wearablepermed-ml -p wearablepermed_ml \
               -d "Uniovi Simur WearablePerMed ML." \
               -u https://github.com/Simur-project/uniovi-simur-wearablepermed-ml.git
          $ deactivate               
          ```

     - Also you can install **pyscaffold** and **pyscaffoldext-markdown** packages in your system and avoid the error from Python 3.11+: ```"error-externally-managed-environment" this environemnt is externally managed``` you can execute this command to force instalation:

          ```
          $ pip3 install pyscaffold --break-system-packages
          $ pip3 install pyscaffoldext-markdown --break-system-packages
          $ putup --markdown uniovi-simur-wearablepermed-ml -p wearablepermed_ml \
               -d "Uniovi Simur WearablePerMed ML." \
               -u https://github.com/Simur-project/uniovi-simur-wearablepermed-ml.git
          ```

          or permanent configure pip3 with this command to avoid the previous errors from 3.11+

          ```
          $ python3 -m pip config set global.break-system-packages true
          ```

- **STEP02**: creare your repo under SIMUR Organization with the name **uniovi-simur-wearablepermed-ml** and clone the previous scaffoled project

     ```
     $ cd git
     $ git clone https://github.com/Simur-project/uniovi-simur-wearablepermed-ml.git
     ```

- **STEP03**: copy PyScaffold project to your git folder without .venv folder

- **STEP04**: install tox project manager used by PyScaffold. Install project dependencies
     ```
     $ python3 -m venv .venv
     $ source .venv/bin/activate
     $ pip install tox
     $ pip install pandas
     $ pip install matplotlib
     $ pip install openpyxl
     $ tox list
     ```

     Installation your python pipeline packages in your virtual environment in development mode:

     ```
     $ pip freeze > requirements.txt
     ```
## Start develop your project
- **STEP01**: Clone your project
     ```
     $ git clone https://github.com/Simur-project/uniovi-simur-wearablepermed-ml.git
     ```

- **STEP01**: Build and Debug your project
     ```
     $ tox list
     default environments:
     default   -> Invoke pytest to run automated tests

     additional environments:
     build     -> Build the package in isolation according to PEP517, see https://github.com/pypa/build
     clean     -> Remove old distribution files and temporary build artifacts (./build and ./dist)
     docs      -> Invoke sphinx-build to build the docs
     doctests  -> Invoke sphinx-build to run doctests
     linkcheck -> Check for broken links in the documentation
     publish   -> Publish the package you have been developing to a package index server. By default, it uses testpypi. If you really want to publish your package to be publicly accessible in PyPI, use the `-- --repository pypi` option
     ```

     ```
     $ tox -e clean
     $ tox -e build
     $ tox -e docs
     $ tox -e publish -- --repository pypi
     ```

- **STEP02 Build service**
     ```
     $ docker build -t uniovi-simur-wearablepermed-ml:1.0.0 .
     ```

- **STEP03: Tag service**
     ```
     $ docker tag uniovi-simur-wearablepermed-ml:1.0.0 ofertoio/uniovi-simur-wearablepermed-ml:1.0.0
     ```

- **STEP04: Publish service**
     ```
     $ docker logout
     $ docker login
     $ docker push ofertoio/uniovi-simur-wearablepermed-ml:1.0.0
     ```

## Using GPU
Follow these steps to use GPU from your python script:

- **STEP01: Install NVIDIA Drivers**
     Install NVIDIA Drivers for your card, in our case: NVIDIA GeForce RTX 4060 Ti card

- **STEP02: Install CUDA Toolkit**
     ```
     $ sudo apt-get install -y nvidia-cuda-toolkit
     ```

     Get CUDA version installed:

     ```
     $ nvcc --version
     nvcc: NVIDIA (R) Cuda compiler driver
     Copyright (c) 2005-2023 NVIDIA Corporation
     Built on Fri_Jan__6_16:45:21_PST_2023
     Cuda compilation tools, release 12.0, V12.0.140
     Build cuda_12.0.r12.0/compiler.32267302_0
     ```

- **STEP03: Install cuDNN**

     Install cuDNN for DeepLearning from python compatible with your CUDA version:

     ```
	wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2404-9.10.2_1.0-1_amd64.deb
	sudo dpkg -i cudnn-local-repo-ubuntu2404-9.10.2_1.0-1_amd64.deb
	sudo cp /var/cudnn-local-repo-ubuntu2404-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
	sudo apt-get update
	sudo apt-get -y install cudnn
     ```

	Install CUDA 12 from aptitude, perform the above configuration but install the CUDA 12 specific package:

     ```
	sudo apt-get -y install cudnn-cuda-12
     ```

     Check cuDNN installed:
     ```
     $ dpkg -l | grep libcudnn
     ii  libcudnn9-cuda-12                             9.10.2.21-1                              amd64        cuDNN runtime libraries for CUDA 12.9
     ii  libcudnn9-dev-cuda-12                         9.10.2.21-1                              amd64        cuDNN development libraries for CUDA 12.9
     ii  libcudnn9-headers-cuda-12                     9.10.2.21-1                              amd64        cuDNN header files for CUDA 12.9
     ii  libcudnn9-static-cuda-12                      9.10.2.21-1                              amd64        cuDNN static libraries for CUDA 12.9
     ```

- **STEP04: Configure CUDA environment**
     Create a file called **cuda_env.sh** with these env variables configurations:

     ```
     export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
     export PATH=$CUDA_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_HOME/compute-sanitizer:$LD_LIBRARY_PATH
     export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
     ```

     make executable the **cuda_env.sh** file:

     ```
     $ chmod +x cuda_env.sh 
     ```

     Move the file to profile.d folder, to have these env variables globally, for any user server sessions     
     ```
     $ sudo mv cuda_env.sh /etc/profile.d/
     ```
     
- **STEP05: Relogin a session**
     You must logout and login again using your accout to have access to CUDA env variables. Check it:

     ```
     echo $CUDA_HOME
     echo $PATH
     echo $LD_LIBRARY_PATH
     echo $XLA_FLAGS
     ```

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
