import matplotlib.pyplot as plt

# Ejemplo de vectores de datos
F1_ESANN_acc_gyr_PI = [44.76, 43.92, 43.86, 44.42, 44.06, 44.38, 43.13, 46.56, 42.08, 43.28, 43.35, 39.27, 44.27, 42.75, 43.33, 44.31, 43.46, 42.81, 42.72, 45.4, 44.81, 43.4, 42.57, 45.41, 45.2, 46.43, 41.69, 41.39, 40.44, 45.87]
F1_ESANN_acc_gyr_M = [32.16, 31.48, 32.52, 34.67, 31.19, 32.38, 31.47, 32.86, 32.97, 32.43, 34.43, 31.61, 33.17, 31.3, 30.8, 34.66, 33.31, 33.35, 34.29, 34.82, 30.85, 34.39, 31.88, 31.85, 32.66, 34.06, 33.15, 31.41, 33.29, 31.34]
F1_ESANN_acc_gyr_C = [36.87, 35.01, 36.23, 36.2, 34.67, 34.9, 35.01, 33.73, 35.92, 36.14, 34.35, 37.35, 33.52, 33.63, 33.59, 34.82, 34.26, 34.11, 35.93, 36.17, 34.81, 36.36, 34.41, 35.83, 33.73, 33.05, 35.07, 35.38, 33.3, 35.83]

# Lista de vectores
data = [F1_ESANN_acc_gyr_PI, F1_ESANN_acc_gyr_M, F1_ESANN_acc_gyr_C]

plt.figure(figsize=(8, 5))
plt.boxplot(data)
plt.xticks([1, 2, 3], ['F1_ESANN_acc_gyr_PI', 'F1_ESANN_acc_gyr_M', 'F1_ESANN_acc_gyr_C'])
plt.ylabel('F1-score')
plt.title('Comparativa F1-scores. Modelo ESANN')
plt.show()
