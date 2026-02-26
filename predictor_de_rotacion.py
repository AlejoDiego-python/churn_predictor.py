import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. SIMULACIÓN DE DATOS (Para demostración técnica)
# En un entorno real, estos datos vendrían de la base de datos SQL de la empresa
data = {
    'meses_cliente': [12, 24, 5, 45, 10, 2, 36, 48, 3, 15],
    'fallas_tecnicas': [0, 1, 4, 0, 3, 5, 1, 0, 6, 2],
    'latencia_ms': [20, 25, 80, 15, 60, 95, 22, 18, 110, 45],
    'pago_atrasado': [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    'churn': [0, 0, 1, 0, 1, 1, 0, 0, 1, 0] # 1 = Se fue a la competencia, 0 = Se quedó
}

df = pd.DataFrame(data)

# 2. PREPARACIÓN DE VARIABLES
X = df.drop('churn', axis=1) # Características (Features)
y = df['churn']              # Objetivo (Target)

# Dividimos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. IMPLEMENTACIÓN DEL MODELO (RANDOM FOREST)
# Usamos hiperparámetros básicos para demostrar conocimiento de la arquitectura
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 4. RESULTADOS E IMPORTANCIA DE VARIABLES
def mostrar_analisis():
    importancias = model.feature_importances_
    for feature, imp in zip(X.columns, importancias):
        print(f"Variable: {feature} | Impacto en la fuga: {imp:.4f}")

if __name__ == "__main__":
    print("--- MODELO PREDICTIVO DE RETENCIÓN DE CLIENTES ---")
    mostrar_analisis()
    # Aquí se demuestra que el modelo puede predecir nuevos casos
    prediccion = model.predict(X_test)
    print("\nAnálisis completado con éxito.")
