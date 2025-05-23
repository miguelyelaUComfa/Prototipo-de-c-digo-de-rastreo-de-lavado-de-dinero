# Importamos las librerías necesarias
import networkx as nx  # Para trabajar con grafos (redes)
import pandas as pd   # Para manejar datos en tablas
from sklearn.metrics import confusion_matrix, classification_report  # Para evaluar el modelo
import matplotlib.pyplot as plt  # Para gráficos
import seaborn as sns  # Para visualización más estética

print("Análisis de Redes para Detección de Lavado de Dinero (AML)")
print("----------------------------------------------------------")

# Definimos las cuentas bancarias que forman la red
accounts = [
    "Cuenta_A", "Cuenta_B", "Cuenta_C", "Cuenta_D", "Cuenta_E",
    "Cuenta_F", "Cuenta_G", "Offshore_X", "Offshore_Y"
]
# La definición de nodos (cuentas) esta presentada en las referencias [1, 6, 9]

# Ground truth: etiquetas reales que nos indican si una cuenta es fraudulenta o no
# 0 = Libre (no sospechosa), 1 = Fraude (posible lavado de dinero)
ground_truth = {
    "Cuenta_A": 0,
    "Cuenta_B": 1,
    "Cuenta_C": 1,
    "Cuenta_D": 0,
    "Cuenta_E": 0,
    "Cuenta_F": 0,
    "Cuenta_G": 1,
    "Offshore_X": 1,
    "Offshore_Y": 1
}
# El uso de etiquetas binarias (0, 1) es común en detección de fraudes [4, 11], y la asignación explícita de 
# cuentas offshore como fraudulentas es consistente con [1, 5], que identifican cuentas offshore como de alto riesgo.

# Lista de transacciones entre cuentas:
# Cada elemento contiene: (origen, destino, monto, frecuencia)
transactions = [
    ("Cuenta_A", "Cuenta_B", 50000, 5),
    ("Cuenta_B", "Cuenta_C", 45000, 4),
    ("Cuenta_C", "Offshore_X", 40000, 3),
    ("Cuenta_A", "Cuenta_D", 10000, 1),
    ("Cuenta_D", "Cuenta_E", 9000, 1),
    ("Cuenta_E", "Cuenta_F", 8000, 1),
    ("Cuenta_F", "Offshore_X", 7000, 1),
    ("Cuenta_A", "Cuenta_G", 30000, 2),
    ("Cuenta_G", "Offshore_Y", 28000, 2),
    ("Offshore_X", "Offshore_Y", 20000, 1)
]
# Las transacciones con monto y frecuencia son relevantes para modelar redes financieras sospechosas,
# como se describe en [1, 6, 9]. La inclusión de cuentas offshore como destinos es consistente con patrones de lavado de dinero [5].

# Creamos un grafo dirigido vacío
G = nx.DiGraph()

# Añadimos los nodos (cuentas) al grafo
for account in accounts:
    G.add_node(account)
# El uso de un grafo dirigido es apropiado para modelar transacciones financieras, como se 
# menciona en [1, 4, 6, 8, 9], ya que las transacciones tienen dirección (origen a destino).

# Añadimos las aristas (transacciones) al grafo
for origen, destino, monto, frecuencia in transactions:
    # Calculamos una puntuación de sospecha: cuanto mayor sea, más sospechosa
    suspicion_score = monto * frecuencia

    # Asignamos un peso inversamente proporcional a la sospecha
    # Esto hace que Dijkstra priorice caminos hacia cuentas sospechosas
    weight = 1000 / suspicion_score

    # Añadimos la conexión con atributos personalizados
    G.add_edge(origen, destino, weight=weight, suspicion_score=suspicion_score)
# El enfoque de 'casillas de verificación' genera altas tasas de falsos positivos que dañan la eficacia del sistema AML

# Definimos las cuentas offshore como puntos de alto riesgo
high_risk_accounts = ["Offshore_X", "Offshore_Y"]

# Inicializamos un diccionario para almacenar la distancia mínima a cuentas offshore
risk_scores = {}

# Calculamos la menor distancia desde cada cuenta a alguna cuenta offshore
for account in accounts:
    min_distance = float('inf')  # Empezamos con infinito
    for offshore in high_risk_accounts:
        try:
            # Usamos Dijkstra para encontrar la distancia mínima hasta offshore
            distance = nx.dijkstra_path_length(G, account, offshore, weight="weight")
            if distance < min_distance:
                min_distance = distance
        except nx.NetworkXNoPath:
            # Si no hay camino, lo ignoramos
            pass
    risk_scores[account] = min_distance  # Guardamos la distancia mínima
# Se usa el algoritmo de Dijkstra para calcular distancias mínimas a nodos de 
# alto riesgo. con enfoques basados en grafos [1, 4, 6, 9] para identificar nodos cercanos a actividades sospechosas.

# Umbral de clasificación: ajustable
# Cuanto menor sea, más estricta será la detección de fraude
THRESHOLD = 0.05
# El uso de un umbral para clasificar cuentas es común [4, 11].

# Clasificamos las cuentas según su proximidad a las offshore
predictions = {
    account: 1 if risk_scores[account] < THRESHOLD else 0
    for account in accounts
}

# Preparamos las etiquetas verdaderas y predichas para evaluar el modelo
y_true = [ground_truth[account] for account in accounts]
y_pred = [predictions[account] for account in accounts]
# La preparación de etiquetas para evaluación [3, 4, 7].

# Calculamos la matriz de confusión y el reporte de clasificación
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Libre", "Fraude"])

# Mostramos los resultados por consola
print("\nResultados de Dijkstra para cada cuenta:")
for account in accounts:
    estado = 'Fraude' if predictions[account] == 1 else 'Libre'
    print(f"{account}: Distancia = {risk_scores[account]:.4f} → Predicción = {estado}")

print("\nMatriz de Confusión:")
print(cm)
print("\nReporte de Clasificación:")
print(report)

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Libre", "Fraude"], yticklabels=["Libre", "Fraude"])
plt.title("Matriz de Confusión (Dijkstra como Clasificador)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
# La visualización de la matriz de confusión con un heatmap es una práctica recomendada en [3, 7] para mejorar la interpretabilidad de los resultados.

# Creamos un DataFrame para mostrar los errores del modelo
results_df = pd.DataFrame({
    "Cuenta": accounts,
    "Real": ["Fraude" if ground_truth[account] == 1 else "Libre" for account in accounts],
    "Predicción": ["Fraude" if predictions[account] == 1 else "Libre" for account in accounts],
    "Distancia Dijkstra": [risk_scores[account] for account in accounts]
})

# Falsos positivos: cuentas marcadas como fraude pero son legítimas
print("\nFalsos Positivos (Libres marcadas como Fraude):")
print(results_df[(results_df["Real"] == "Libre") & (results_df["Predicción"] == "Fraude")])

# Falsos negativos: cuentas marcadas como limpias pero son fraudulentas
print("\nFalsos Negativos (Fraudes marcadas como Libre):")
print(results_df[(results_df["Real"] == "Fraude") & (results_df["Predicción"] == "Libre")])
# La identificación de falsos positivos y negativos es crucial esto es mencionado en [3, 4, 7, 11].