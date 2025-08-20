import os                                                    # Para manejar rutas de archivos
import subprocess                                            # Para ejecutar scripts Python como procesos separados
import re                                                    # Para expresiones regulares, usado para extraer accuracy
import numpy as np                                           # Para operaciones numéricas y guardar resultados

N_RUNS = 30                                                  # Número de ejecuciones de train+test

# Ruta Windows
# case_id_folder = "D:\\DATA_PMP_File_Server\\output"          # Carpeta base de los datos
# Ruta Linux
case_id_folder = "/mnt/nvme1n2/git/uniovi-simur-wearablepermed-data/output"

case_id = "case_19"                                          # Identificador del caso

# Argumentos para el script de entrenamiento
train_args = [
    # Ruta Windows
    # "src\\wearablepermed_ml\\trainer.py",                  # Script de entrenamiento
    # Ruta Linux                                             
    "src/wearablepermed_ml/trainer.py",                      # Script de entrenamiento
    "--case-id", case_id,                                    # ID del caso
    "--case-id-folder", case_id_folder,                      # Carpeta de datos
    "--ml-models", "ESANN",                                  # Modelo ML a usar
    "--training-percent", "70",                              # Porcentaje de datos para entrenamiento
    "--validation-percent", "20",                            # Porcentaje de datos para validación
    "--create-superclasses"                                  # Flag opcional para crear superclases
]

# Argumentos para el script de test
test_args = [
    # Ruta Windows
    # "src\\wearablepermed_ml\\tester.py",                     # Script de test
    # Ruta Linux 
    "src/wearablepermed_ml/tester.py",                       # Script de test
    "--case-id", case_id,                                    # ID del caso
    "--case-id-folder", case_id_folder,                      # Carpeta de datos
    "--model-id", "ESANN",                                   # Modelo ML usado para test
    "--training-percent", "70",                              # Porcentaje usado en entrenamiento
    "--validation-percent", "20",                            # Porcentaje de datos para validacións
    "--create-superclasses"                                  # Flag opcional
]

# Ruta del ejecutable de Python del entorno virtual (Windows)
# python_exe = os.path.join(".venv", "Scripts", "python.exe")
# En Linux, será:
python_exe = os.path.join(".venv", "bin", "python")

accuracies = []                                              # Lista para almacenar los accuracy de cada ejecución

for i in range(1, N_RUNS + 1):                               # Bucle principal para N_RUNS ejecuciones
    print(f"\n=== EJECUCIÓN {i} ===")

    # --- TRAIN ---
    print(f"\n--- TRAIN (ejecución {i}) ---")
    subprocess.run([python_exe] + train_args, check=True)    # Ejecuta trainer.py y lanza excepción si falla

    # --- TEST ---
    test_args_with_i = test_args + ["--run-index", str(i)]   # Añadimos índice de ejecución al test
    print(f"\n--- TEST (ejecución {i}) ---")

    result = subprocess.run(
        [python_exe] + test_args_with_i,                     # Ejecuta tester.py
        check=True,                                          # Lanza excepción si falla
        capture_output=True,                                 # Captura stdout y stderr
        text=True                                            # Interpreta la salida como string
    )

    print(result.stdout)                                     # Imprime la salida completa del test

    # Extraer el valor de accuracy usando regex
    match = re.search(r"Global accuracy score\s*=\s*([0-9.]+)", result.stdout)
    if match:
        acc = float(match.group(1))                                 # Convertimos a float
        accuracies.append(acc)                                      # Guardamos en la lista
        print(f"Accuracy capturado en la ejecución {i}: {acc} [%]")
    else:
        print("No se encontró 'Global accuracy score' en la salida de tester.py")

# --- RESUMEN FINAL ---
print("\n=== RESUMEN FINAL ===")
print("Accuracies:", accuracies)              # Lista completa de accuracies
if accuracies:
    print("Media:", np.mean(accuracies))      # Media de los accuracies
    print("Std:", np.std(accuracies))         # Desviación estándar de los accuracies

# --- GUARDAR EN .npz ---
accuracies_test_path = os.path.join(case_id_folder, case_id, "accuracies_test.npz")  # Ruta final del archivo
np.savez(
    accuracies_test_path,                               # Guardar en archivo .npz
    accuracies=accuracies,                              # Lista de accuracies
    mean=np.mean(accuracies),                           # Media
    std=np.std(accuracies)                              # Desviación estándar
)
print("\nResultados guardados en accuracies_test.npz")  # Mensaje final
