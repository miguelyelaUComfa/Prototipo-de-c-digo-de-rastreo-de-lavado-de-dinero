import networkx as nx
import numpy as np
import pandas as pd

print("Análisis de Redes para Detección de Lavado de Dinero (AML)")
print("---------------------------------------------------------")

# CREACIÓN DE DATOS DE EJEMPLO
# ------------------------------------------------------------------------------

print("\n1. Creando datos de ejemplo...")

# Los artículos menciona que los sistemas AML requieren datos transaccionales realistas.
# Estas transacciones simuladas incluyen montos y frecuencias, factores clave según FATF.
# ------------------------------------------------------------------------------
accounts = ["Cuenta_A", "Cuenta_B", "Cuenta_C", "Cuenta_D", "Cuenta_E", 
            "Cuenta_F", "Cuenta_G", "Offshore_X", "Offshore_Y"]

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

df = pd.DataFrame(transactions, columns=["origen", "destino", "monto", "frecuencia"])
print("Transacciones de ejemplo:")
print(df)

# CREACIÓN DEL GRAFO
# ------------------------------------------------------------------------------
print("\n2. Construyendo el grafo de transacciones...")
G = nx.DiGraph()

for account in accounts:
    G.add_node(account)

# Validación de reducción de falsos positivos
# ------------------------------------------------------------------------------
for _, row in df.iterrows():
    suspicion_score = row["monto"] * row["frecuencia"]  # Métrica cuantitativa de riesgo
    
# El enfoque de 'casillas de verificación' genera altas tasas de falsos positivos que dañan la eficacia del sistema AML."
# ------------------------------------------------------------------------------
    weight = 1000 / suspicion_score  # Peso inverso: mayor sospecha = menor peso, Puntaje basado en datos
    
    G.add_edge(
        row["origen"], 
        row["destino"], 
        weight=weight,
        monto=row["monto"],
        frecuencia=row["frecuencia"],
        suspicion_score=suspicion_score
    )

print(f"Grafo creado con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas")

# ALGORITMO DE DIJKSTRA PARA RUTAS SOSPECHOSAS
# ------------------------------------------------------------------------------
print("\n3. Aplicando el algoritmo de Dijkstra para encontrar rutas sospechosas...")
source = "Cuenta_A"
target = "Offshore_X"

try:
    # Los expertos evalúan la racionalidad económica de una transacción para determinar si es sospechosa
    # ------------------------------------------------------------------------------
    path = nx.dijkstra_path(G, source, target, weight="weight") # Ruta con mayor sospecha
    # Proximidad a nodos de alto riesgo mediante métricas de distancia gráfica.
    length = nx.dijkstra_path_length(G, source, target, weight="weight")
    
    print(f"Camino más sospechoso de {source} a {target}:")
    print(" -> ".join(path))
    print(f"Puntuación total: {length:.2f}")
    
    print("\nDetalles de las transacciones en el camino:")
    total_amount = 0
    for i in range(len(path)-1):
        src = path[i]
        dst = path[i+1]
        edge_data = G.get_edge_data(src, dst)
        amount = edge_data["monto"]
        freq = edge_data["frecuencia"]
        susp = edge_data["suspicion_score"]
        total_amount += amount
        
        print(f"{src} -> {dst}: ${amount:.2f} (frecuencia: {freq:.1f}, sospecha: {susp:.1f})")
    
    print(f"Monto total transferido: ${total_amount:.2f}")

except nx.NetworkXNoPath:
    print(f"No se encontró un camino de {source} a {target}")

# IDENTIFICACIÓN DE CUENTAS "PUENTE"
# ------------------------------------------------------------------------------
print("\n4. Identificando cuentas puente (intermediarios)...")

# Los expertos deben identificar intermediarios (p. ej., capas de transacciones).
# La centralidad de intermediación cuantifica este rol, automatizando lo que humanos hacen subjetivamente.
# ------------------------------------------------------------------------------
betweenness = nx.betweenness_centrality(G, weight="weight") # Detecta intermediarios clave
betweenness_df = pd.DataFrame(list(betweenness.items()), columns=["cuenta", "centralidad"])
betweenness_df = betweenness_df.sort_values("centralidad", ascending=False)

print("Cuentas ordenadas por centralidad de intermediación:")
print(betweenness_df)

# PRIORIZACIÓN DE ALERTAS POR PROXIMIDAD A OFFSHORE
# ------------------------------------------------------------------------------
print("\n5. Priorizando alertas basadas en la proximidad a cuentas de alto riesgo...")

# Cuentas no residentes o en jurisdicciones de alto riesgo son factores de peligro
# ------------------------------------------------------------------------------
high_risk_accounts = ["Offshore_X", "Offshore_Y"] # Cuentas de alto riesgo
print(f"Cuentas de alto riesgo: {', '.join(high_risk_accounts)}")

risk_scores = {}
for account in accounts:
    if account in high_risk_accounts:
        continue
        
    min_distance = float("inf")
    closest_account = None
    
    for risk_account in high_risk_accounts:
        try:
            distance = nx.dijkstra_path_length(G, account, risk_account, weight="weight")
            if distance < min_distance:
                min_distance = distance
                closest_account = risk_account
        except nx.NetworkXNoPath:
            pass
    
    if min_distance != float("inf"):
        risk_scores[account] = {
            "puntaje_riesgo": 1000 / min_distance,  # Puntaje inverso a la distancia
            "cuenta_cercana": closest_account,
            "distancia": min_distance
        }
    else:
        risk_scores[account] = {
            "puntaje_riesgo": 0,
            "cuenta_cercana": None,
            "distancia": float("inf")
        }

risk_df = pd.DataFrame.from_dict(risk_scores, orient="index")
risk_df = risk_df.sort_values("puntaje_riesgo", ascending=False)
risk_df = risk_df.reset_index().rename(columns={"index": "cuenta"})

print("Cuentas ordenadas por puntaje de riesgo:")
print(risk_df)

# 6. CONCLUSIONES
# ------------------------------------------------------------------------------
print("\n6. Conclusiones del análisis:")

print("-----------------------------")
# a) Ruta sospechosa (usando los resultados de Dijkstra)
print("\na) Ruta más sospechosa identificada usando Dijkstra:")
print(f"   - Camino encontrado: {' -> '.join(path)}")
print(f"   - Puntuación total de sospecha: {length:.2f}")
print(f"   - Monto total transferido: ${total_amount:.2f}")

# b) Cuentas puente (usando betweenness centrality)
print("\nb) Cuentas puente identificadas usando centralidad de intermediación:")
print("   - Top 3 cuentas con mayor centralidad:")
top_betweenness = betweenness_df.head(3)
for idx, row in top_betweenness.iterrows():
    print(f"     {row['cuenta']}: {row['centralidad']:.4f}")

# c) Alertas priorizadas (usando risk_scores)
print("\nc) Alertas priorizadas basadas en proximidad a cuentas de alto riesgo:")
print("   - Top 3 cuentas con mayor puntaje de riesgo:")
top_risk = risk_df.head(3)
for idx, row in top_risk.iterrows():
    print(f"     {row['cuenta']}: Puntaje {row['puntaje_riesgo']:.2f} (Cercana a: {row['cuenta_cercana']})")

# Mensaje final
print("\nEste análisis demuestra cómo los algoritmos de grafos pueden")
print("detectar patrones sospechosos en redes financieras, identificando:")
print("- Rutas críticas de flujo de dinero")
print("- Nodos intermediarios clave")
print("- Cuentas de alto riesgo por asociación")