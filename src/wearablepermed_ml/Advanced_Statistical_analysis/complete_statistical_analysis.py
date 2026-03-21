#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS ESTADÍSTICO COMPLETO PARA 48 EXPERIMENTOS × 30 REPETICIONES
Incluye:
- Construcción del dataset largo
- ANOVA factorial (typ=3)
- Residuos + Test de Shapiro–Wilk
- Test de homogeneidad: Levene/Brown–Forsythe
- EMMs modelo‑basados
- Efectos simples inferenciales
- Post‑hoc: Tukey
- Exportación de resultados
"""

import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, levene
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

# =========================================================
# 1. Cargar vectores F1 y archivo meta (48 filas)
# =========================================================

# Compute the REAL path of the .py script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path of the .npy file
f1_vectors_path = os.path.join(script_dir, "f1_vectors_48x30.npy")
meta_48_experiments_path = os.path.join(script_dir, "meta_48_experimentos.csv")

f1_vectors = np.load(f1_vectors_path, allow_pickle=True)  # matrix 48×30
meta = pd.read_csv(meta_48_experiments_path)

assert len(f1_vectors) == len(meta) == 48, "Los 48 vectores deben coincidir con las 48 filas de meta."

# =========================================================
# 2. Construir dataset de F1-scores (48 × 30 = 1440 filas)
# =========================================================
def build_long_df(f1_vectors, meta):
    rows = []
    for i, vec in enumerate(f1_vectors):
        for r, val in enumerate(vec, start=1):
            rows.append({
                'ExpID': i,
                'Repeticion': r,
                'F1': float(val),
                'Modelo': meta.loc[i, 'Modelo'],
                'Sensor': meta.loc[i, 'Sensor'],
                'Config': meta.loc[i, 'Config'],
                'NumClases': meta.loc[i, 'NumClases'],
            })
    return pd.DataFrame(rows)

df = build_long_df(f1_vectors, meta)

# Convertir factores a categoría
for col in ['Modelo', 'Sensor', 'Config', 'NumClases']:
    df[col] = df[col].astype('category')

# =========================================================
# 3. ANOVA factorial completo (typ=3)
# =========================================================
formula = 'F1 ~ C(Modelo) + C(Sensor) + C(Config) + C(NumClases) \
           + C(Modelo):C(Sensor) \
           + C(Modelo):C(Config) \
           + C(Modelo):C(NumClases) \
           + C(Sensor):C(Config) \
           + C(Sensor):C(NumClases) \
           + C(Config):C(NumClases)'
modelo = ols(formula, data=df).fit()
anova = sm.stats.anova_lm(modelo, typ=3)
anova.to_csv(os.path.join(script_dir,'ANOVA_typ3.csv'))
print("ANOVA completado. Guardado en ANOVA_typ3.csv")

# =========================================================
# 4. Residuos + Test de Shapiro-Wilk
# =========================================================
residuos = modelo.resid
W, p = shapiro(residuos)
print(f"Shapiro–Wilk: W={W:.4f}, p={p:.4g}")
with open(os.path.join(script_dir,'shapiro_residuos.txt'),'w') as f:
    f.write(f"Shapiro–Wilk W={W}, p={p}\n")

# QQ-Plot
fig = sm.qqplot(residuos, line='45', fit=True)
plt.title('Q–Q plot de residuos')
plt.tight_layout()
plt.savefig(os.path.join(script_dir,'QQplot_residuos.png'))
plt.close()

# =========================================================
# 5. Test de homogeneidad: Levene/Brown–Forsythe
# =========================================================
def levene_by(df, factor):
    groups = [grp['F1'].values for _, grp in df.groupby(factor)]
    return levene(*groups, center='median')   # Brown–Forsythe

homoced = {}
for fac in ['Modelo','Sensor','Config','NumClases']:
    stat, p = levene_by(df, fac)
    homoced[fac] = (stat, p)

pd.DataFrame.from_dict(homoced, orient='index', columns=['stat','p']).to_csv(os.path.join(script_dir,'homogeneidad_brown_forsythe.csv'))
print("Tests de homogeneidad guardados.")

# =========================================================
# 6. EMMs modelo‑basados (predicciones marginales)
# =========================================================
from patsy import dmatrix
import itertools

factors = ['Modelo','Sensor','Config','NumClases']

# Función para EMMs

def emms_for_factor(model, df, factor, other_factors):
    levels = df[factor].unique()
    others_levels = [df[f].unique() for f in other_factors]
    combos = list(itertools.product(*others_levels))
    rows = []

    for lvl in levels:
        preds = []
        for combo in combos:
            row = {factor: lvl}
            for f, val in zip(other_factors, combo):
                row[f] = val
            design = dmatrix(model.model.data.design_info.builder, pd.DataFrame([row]), return_type='dataframe')
            yhat = np.dot(design, model.params)
            preds.append(float(yhat))
        rows.append({factor: lvl, 'EMM': float(np.mean(preds))})
    return pd.DataFrame(rows)

# Calcular EMMs
emms = {}
for target in factors:
    others = [f for f in factors if f != target]
    emms[target] = emms_for_factor(modelo, df, target, others)
    emms[target].to_csv(os.path.join(script_dir,f'EMM_{target}.csv'), index=False)

print("EMMs calculados y guardados.")

# =========================================================
# 7. Efectos simples + post-hoc Tukey (inferencial)
# =========================================================
resultados_simple = []

for sensor_nivel, sub in df.groupby('Sensor'):
    tuk = pairwise_tukeyhsd(endog=sub['F1'], groups=sub['Modelo'], alpha=0.05)
    out = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    out['Sensor'] = sensor_nivel
    resultados_simple.append(out)

ef_simple_df = pd.concat(resultados_simple, ignore_index=True)
ef_simple_df.to_csv(os.path.join(script_dir,'EfectosSimples_Modelo_en_Sensor_Tukey.csv'), index=False)
print("Efectos simples (Modelo dentro de Sensor) guardados.")

print("\nPipeline completado con éxito.")


# ====================================================================================
# 8. Efectos simples + post-hoc Tukey (inferencial) PARA TODOS LOS FACTORES Y NIVELES
# ====================================================================================
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

factores = ["Modelo", "Sensor", "Config", "NumClases"]
resultados = []

# ---------------------------------------------
# 1) Tukey global por cada factor principal
# ---------------------------------------------
for fac in factores:
    tuk = pairwise_tukeyhsd(endog=df["F1"], groups=df[fac], alpha=0.05)
    out = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    out["Factor"] = fac
    out["Nivel"] = "GLOBAL"
    resultados.append(out)

# ---------------------------------------------
# 2) Tukey por efectos simples (todas interacciones 2-way)
# ---------------------------------------------
from itertools import combinations

for facA, facB in combinations(factores, 2):
    # facA: factor sobre el que comparamos
    # facB: factor dentro del cual comparamos
    for nivel in df[facB].unique():
        sub = df[df[facB] == nivel]
        if len(sub[facA].unique()) < 2:
            continue
        tuk = pairwise_tukeyhsd(endog=sub["F1"], groups=sub[facA], alpha=0.05)
        out = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
        out["Factor"] = facA
        out["Nivel"] = f"{facB}={nivel}"
        resultados.append(out)

# Unir todo y guardar
df_tukey = pd.concat(resultados, ignore_index=True)
df_tukey.to_csv(os.path.join(script_dir,"Tukey_completo_todos_factores.csv"), index=False)

print("Análisis Tukey completo guardado en Tukey_completo_todos_factores.csv")
