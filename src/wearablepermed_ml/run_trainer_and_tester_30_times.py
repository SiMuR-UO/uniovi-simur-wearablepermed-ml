import os                                                    # Para manejar rutas de archivos
import subprocess                                            # Para ejecutar scripts Python como procesos separados
import argparse
import re
import sys                                                    # Para expresiones regulares, usado para extraer accuracy
import numpy as np                                           # Para operaciones numéricas y guardar resultados

N_RUNS = 30                                                  # Número de ejecuciones de train+test

# Ruta Windows
# case_id_folder = "D:\\DATA_PMP_File_Server\\output"          # Carpeta base de los datos
# Ruta Linux
case_id_folder = "/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/output"
# case_id_folder = "/mnt/simur-fileserver/data/wearablepermed/output"

case_id = "RECURADO_DE_DATOS_Paper_results/cases_dataset_C/case_C_CAPTURE24_acc_gyr_superclasses_CPA_METs/"                                         # Identificador del caso

# Argumentos para el script de entrenamiento
args_model = [
    # Ruta Linux                                             
    "src/wearablepermed_ml/trainer.py",                      # Script de entrenamiento
    "--case-id", case_id,                                    # ID del caso
    "--case-id-folder", case_id_folder,                      # Carpeta de datos
    "--ml-models", "CAPTURE24",                                  # Modelo ML a usar
    "--training-percent", "70",                              # Porcentaje de datos para entrenamiento
    "--validation-percent", "20",                            # Porcentaje de datos para validación
    # "--create-superclasses",                                  # Flag opcional para crear superclases
    "--create-superclasses-CPA-METs"
    # "--create-9-superclasses-CAPTURE24"
]

# Ruta del ejecutable de Python del entorno virtual (Windows)
# python_exe = os.path.join(".venv", "Scripts", "python.exe")
# En Linux, será:
python_exe = os.path.join(".venv", "bin", "python")

accuracies = []                                                     # Lista donde se guardarán las accuracy de cada ejecución
# recalls = []                                                        # Lista para los recall capturados
f1_scores = []                                                      # Lista para los f1-score capturados

def parse_args(args):
    """Parse command line parameters

    Args:hip
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="WareablePerMed Pipeline")

    parser.add_argument(
        "-file",
        "--file",
        dest="file",
        required=True,
        help="File: training or test file."
    )

    parser.add_argument(
        "-case-id",
        "--case-id",
        dest="case_id",
        required=True,
        help="Case ID."
    )

    parser.add_argument(
        "-case-id-folder",
        "--case-id-folder",
        dest="case_id_folder",
        required=True,
        help="Case ID Folder."
    )

    parser.add_argument(
        "-model-id",
        "--model-id",
        dest="model_id",
        required=True,
        help="Model ID."
    )

    parser.add_argument(
        "-training-percent",
        "--training-percent",
        dest="training_percent",
        required=True,
        type=int,
        help="Training Percent."
    )

    parser.add_argument(
        "-validation-percent",
        "--validation-percent",
        dest="validation_percent",
        type=int,
        help="Validation Percent."
    )

    parser.add_argument(
        "-create-superclasses-CPA-METs",
        "--create-superclasses-CPA-METs",
        dest="create_superclasses_CPA_METs",
        action='store_true',
        help="create superclasses CPA METs.")    
    
    return parser.parse_args(args)

args = parse_args(sys.argv[1:])

model_args = [
    args.file,                                       # Script de entrenamiento o test
    "--case-id", args.case_id,                       # Modelo ML a usar
    "--case-id-folder", args.case_id_folder,         # Carpeta de datos
    "--ml-models", args.ml_models,                   # Modelo ML a usar
    "--training-percent", args.training_percent,     # Porcentaje de datos para entrenamiento
    "--validation-percent", args.validation_percent  # Porcentaje de datos para validación   
]

if args.create_superclasses_CPA_METs == True:
    model_args.append("--create-superclasses-CPA-METs" )

for i in range(1, N_RUNS + 1):                                      # Bucle principal: repite N_RUNS veces
    print(f"\n=== EJECUCIÓN {i} ===")                               # Indica el número de ejecución actual

    # --- TRAIN ---
    print(f"\n--- TRAIN (ejecución {i}) ---")                       # Mensaje de inicio de entrenamiento
    subprocess.run([python_exe] + model_args, check=True)           # Ejecuta trainer.py con los argumentos definidos

    # --- TEST ---
    test_args_with_i = model_args + ["--run-index", str(i)]          # Agrega el índice de ejecución al comando de test
    print(f"\n--- TEST (ejecución {i}) ---")                        # Mensaje de inicio de test

    result = subprocess.run(                                        # Lanza tester.py y captura su salida
        [python_exe] + test_args_with_i,                            # Comando completo (python + tester.py + args)
        check=True,                                                 # Si hay error, lanza excepción
        capture_output=True,                                        # Captura stdout y stderr
        text=True                                                   # Interpreta la salida como texto (no bytes)
    )

    print(result.stdout)                                            # Muestra la salida completa del tester.py

    # --- Extraer métricas ---
    acc_match = re.search(r"Global\s*accuracy\s*score\s*\(test\)\s*=\s*([0-9.]+)\s*\[%\]", result.stdout)  # Busca el accuracy en la salida
    # recall_match = re.search(r"Global recall score\s*=\s*([0-9.]+)", result.stdout)  # Busca el recall
    f1_match = re.search(r"Global\s*F1\s*score\s*\(test\)\s*=\s*([0-9.]+)\s*\[%\]", result.stdout)     # Busca el F1-score (permite F1-score o F1 score)

    if acc_match:                                                   # Si se encontró el accuracy
        acc = float(acc_match.group(1))                             # Convierte el valor capturado a float
        accuracies.append(acc)                                      # Lo guarda en la lista
        print(f"Accuracy capturado en la ejecución {i}: {acc} [%]") # Muestra el valor capturado
    else:
        print("No se encontró 'Global accuracy score' en la salida de tester.py")  # Aviso si no se encontró

    # if recall_match:                                                # Si se encontró el recall
    #     rec = float(recall_match.group(1))                          # Convierte a float
    #     recalls.append(rec)                                         # Guarda el valor
    #     print(f"Recall capturado en la ejecución {i}: {rec} [%]")   # Muestra el valor capturado
    # else:
    #     print("No se encontró 'Global recall score' en la salida de tester.py")     # Aviso si falta el dato

    if f1_match:                                                    # Si se encontró el F1-score
        f1 = float(f1_match.group(1))                               # Convierte a float
        f1_scores.append(f1)                                        # Guarda el valor
        print(f"F1-score capturado en la ejecución {i}: {f1} [%]")  # Muestra el valor capturado
    else:
        print("No se encontró 'Global F1-score' en la salida de tester.py")         # Aviso si no se encontró

# --- RESUMEN FINAL ---
print("\n=== RESUMEN FINAL ===")                                    # Título del resumen
print("Accuracies:", accuracies)                                    # Muestra lista completa de accuracies
# print("Recalls:", recalls)                                          # Muestra lista de recalls
print("F1-scores:", f1_scores)                                      # Muestra lista de F1-scores

if accuracies:                                                      # Si hay valores de accuracy
    print(f"Accuracy mean: {np.mean(accuracies):.4f} | std: {np.std(accuracies):.4f}")  # Calcula y muestra media y std
# if recalls:                                                         # Si hay valores de recall
#     print(f"Recall mean: {np.mean(recalls):.4f} | std: {np.std(recalls):.4f}")          # Calcula y muestra media y std
if f1_scores:                                                       # Si hay valores de f1
    print(f"F1 mean: {np.mean(f1_scores):.4f} | std: {np.std(f1_scores):.4f}")          # Calcula y muestra media y std


# --- GUARDAR EN .npz ---
accuracies_test_path = os.path.join(case_id_folder, case_id, "metrics_test.npz")    # Ruta final donde guardar el archivo

np.savez(                                                     # Guarda los datos en un archivo comprimido .npz
    accuracies_test_path,                                     # Nombre/ruta del archivo de salida
    accuracies=np.array(accuracies),                          # Lista de accuracies
    # recalls=np.array(recalls),                                # Lista de recalls
    f1_scores=np.array(f1_scores),                            # Lista de F1-scores
    acc_mean=np.mean(accuracies),                             # Media de accuracy
    acc_std=np.std(accuracies),                               # Desviación estándar de accuracy
    # rec_mean=np.mean(recalls),                                # Media de recall
    # rec_std=np.std(recalls),                                  # Desviación estándar de recall
    f1_mean=np.mean(f1_scores),                               # Media de F1-score
    f1_std=np.std(f1_scores)                                  # Desviación estándar de F1-score
)

print(f"\nResultados guardados en {accuracies_test_path}")         # Mensaje final de confirmación
