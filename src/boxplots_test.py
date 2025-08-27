import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Crear 7 vectores aleatorios
# -----------------------------
np.random.seed(42)  # Para reproducibilidad
vectores = [np.random.randint(1, 101, size=15) for _ in range(7)]

# Etiquetas y colores para cada caja
labels = ['Thigh', 'Wrist', 'Hip', 'Thigh + Wrist', 'Thigh + Hip', 'Wrist + Hip', 'Thigh + Wrist + Hip']
colores = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 
           'lightcoral', 'lightgray', 'lightskyblue']

# -----------------------------
# Configurar boxplot
# -----------------------------
plt.boxplot(
    vectores,
    labels=labels,
    notch=True,                # Activar muescas
    patch_artist=True,         # Permite colorear cajas
    boxprops=dict(facecolor='lightblue', color='blue'),
    medianprops=dict(color='red'),
    whiskerprops=dict(color='green'),
    capprops=dict(color='black'),
    flierprops=dict(marker='o', color='purple', markersize=8)  # Outliers
)

# Cambiar color de cada caja individualmente
for patch, color in zip(plt.gca().artists, colores):
    patch.set_facecolor(color)

# Título y mostrar gráfico
plt.title("Boxplot de 7 vectores aleatorios con colores y notch")
plt.ylabel("Valores")
# Rotar etiquetas del eje X
plt.xticks(rotation=25)
plt.show()
