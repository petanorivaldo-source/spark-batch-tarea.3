"""
Tarea 3 - Big Data (UNAD) - Procesamiento en batch con Apache Spark
Dataset: NYC Yellow Taxi Trip Records (TLC, Nueva York)

Problema: ¿Cómo varían la demanda, la tarifa y la propina de los viajes en taxi
según la hora del día, el día de la semana y el tipo de pago?
"""
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")  # permite guardar gráficas sin interfaz gráfica
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# 1. Carga de datos desde la fuente original
# ---------------------------------------------------------------------------
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "yellow_tripdata_2024-01.parquet")
OUT_DIR = "resultados"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    print("Descargando dataset desde la fuente original (TLC NYC)...")
    urllib.request.urlretrieve(URL, DATA_FILE)

spark = (
    SparkSession.builder.appName("Tarea3-Batch-NYC-Taxi")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(DATA_FILE)
print(f"\nRegistros cargados: {df.count():,}")
print(f"Particiones: {df.rdd.getNumPartitions()}")
df.printSchema()

# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
total_inicial = df.count()

# 2.1 Valores nulos en las columnas clave
cols_clave = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "fare_amount", "tip_amount", "total_amount", "payment_type",
]
df = df.dropna(subset=cols_clave)

# 2.2 Duplicados
df = df.dropDuplicates()

# 2.3 Filtros de calidad: valores imposibles o atípicos
df = df.filter(
    (F.col("trip_distance") > 0) & (F.col("trip_distance") < 100)
    & (F.col("fare_amount") > 0) & (F.col("fare_amount") < 500)
    & (F.col("tip_amount") >= 0)
    & (F.col("passenger_count") > 0) & (F.col("passenger_count") <= 6)
    & (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
    # el archivo es de enero 2024: se descartan fechas fuera de rango
    & (F.col("tpep_pickup_datetime") >= "2024-01-01")
    & (F.col("tpep_pickup_datetime") < "2024-02-01")
)

total_limpio = df.count()
print(f"\nRegistros antes de limpiar : {total_inicial:,}")
print(f"Registros después de limpiar: {total_limpio:,}")
print(f"Registros eliminados        : {total_inicial - total_limpio:,} "
      f"({(total_inicial - total_limpio) / total_inicial:.2%})")

# ---------------------------------------------------------------------------
# 3. Transformación (columnas derivadas)
# ---------------------------------------------------------------------------
df = (
    df.withColumn("hora", F.hour("tpep_pickup_datetime"))
    # dayofweek: 1 = domingo ... 7 = sábado
    .withColumn("dia_num", F.dayofweek("tpep_pickup_datetime"))
    .withColumn(
        "dia_semana",
        F.expr(
            "CASE dayofweek(tpep_pickup_datetime) "
            "WHEN 1 THEN 'Dom' WHEN 2 THEN 'Lun' WHEN 3 THEN 'Mar' "
            "WHEN 4 THEN 'Mié' WHEN 5 THEN 'Jue' WHEN 6 THEN 'Vie' ELSE 'Sáb' END"
        ),
    )
    .withColumn(
        "duracion_min",
        (F.unix_timestamp("tpep_dropoff_datetime")
         - F.unix_timestamp("tpep_pickup_datetime")) / 60,
    )
    .withColumn("velocidad_kmh", (F.col("trip_distance") * 1.609) / (F.col("duracion_min") / 60))
    .withColumn("pct_propina", F.col("tip_amount") / F.col("fare_amount") * 100)
    .withColumn(
        "tipo_pago",
        F.expr(
            "CASE payment_type WHEN 1 THEN 'Tarjeta' WHEN 2 THEN 'Efectivo' "
            "WHEN 3 THEN 'Sin cargo' WHEN 4 THEN 'Disputa' ELSE 'Otro' END"
        ),
    )
)

# Se descartan viajes con duración o velocidad poco realistas
df = df.filter(
    (F.col("duracion_min") >= 1) & (F.col("duracion_min") <= 180)
    & (F.col("velocidad_kmh") < 120)
).cache()

print(f"Registros finales para el análisis: {df.count():,}")

# ---------------------------------------------------------------------------
# 4. Análisis exploratorio (EDA)
# ---------------------------------------------------------------------------
print("\n--- Estadísticas descriptivas ---")
df.select("trip_distance", "duracion_min", "fare_amount", "tip_amount", "total_amount") \
  .describe().show()

# 4.1 Demanda por hora
por_hora = (
    df.groupBy("hora")
    .agg(F.count("*").alias("viajes"),
         F.round(F.avg("fare_amount"), 2).alias("tarifa_prom"))
    .orderBy("hora")
)
print("--- Viajes por hora ---")
por_hora.show(24)

# 4.2 Demanda por día de la semana
por_dia = (
    df.groupBy("dia_num", "dia_semana")
    .agg(F.count("*").alias("viajes"),
         F.round(F.avg("total_amount"), 2).alias("total_prom"))
    .orderBy("dia_num")
)
print("--- Viajes por día de la semana ---")
por_dia.show()

# 4.3 Propina según tipo de pago
por_pago = (
    df.groupBy("tipo_pago")
    .agg(F.count("*").alias("viajes"),
         F.round(F.avg("tip_amount"), 2).alias("propina_prom"),
         F.round(F.avg("pct_propina"), 2).alias("pct_propina_prom"))
    .orderBy(F.desc("viajes"))
)
print("--- Propina por tipo de pago ---")
por_pago.show()

# 4.4 Mapa de calor hora x día (conteo)
heat = (
    df.groupBy("dia_num", "hora").agg(F.count("*").alias("viajes"))
    .orderBy("dia_num", "hora")
)

# 4.5 Top 10 zonas de recogida
top_zonas = (
    df.groupBy("PULocationID").agg(F.count("*").alias("viajes"))
    .orderBy(F.desc("viajes")).limit(10)
)
print("--- Top 10 zonas de recogida (PULocationID) ---")
top_zonas.show()

# 4.6 Ejemplo equivalente con RDDs (map / reduceByKey / take)
rdd_pasajeros = (
    df.select("passenger_count").rdd
    .map(lambda r: (int(r[0]), 1))
    .reduceByKey(lambda a, b: a + b)
    .sortByKey()
)
print("--- Viajes por número de pasajeros (RDD) ---")
for k, v in rdd_pasajeros.take(6):
    print(f"{k} pasajero(s): {v:,}")

# 4.7 Correlaciones
print("\n--- Correlaciones ---")
print("distancia vs tarifa :", round(df.stat.corr("trip_distance", "fare_amount"), 3))
print("distancia vs propina:", round(df.stat.corr("trip_distance", "tip_amount"), 3))
print("duración vs tarifa  :", round(df.stat.corr("duracion_min", "fare_amount"), 3))

# ---------------------------------------------------------------------------
# 5. Visualización (los resultados agregados son pequeños: se pasan a pandas)
# ---------------------------------------------------------------------------
ph = por_hora.toPandas()
pd_ = por_dia.toPandas()
pp = por_pago.toPandas()
hm = heat.toPandas().pivot(index="dia_num", columns="hora", values="viajes")
tz = top_zonas.toPandas()

# Gráfica 1: viajes por hora
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(ph["hora"], ph["viajes"], color="#1f77b4")
ax.set_title("Viajes por hora del día - Enero 2024")
ax.set_xlabel("Hora")
ax.set_ylabel("Número de viajes")
ax.set_xticks(range(0, 24))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_viajes_por_hora.png", dpi=150)
plt.close(fig)

# Gráfica 2: viajes por día de la semana
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(pd_["dia_semana"], pd_["viajes"], color="#ff7f0e")
ax.set_title("Viajes por día de la semana")
ax.set_ylabel("Número de viajes")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_viajes_por_dia.png", dpi=150)
plt.close(fig)

# Gráfica 3: propina promedio por tipo de pago
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(pp["tipo_pago"], pp["propina_prom"], color="#2ca02c")
ax.set_title("Propina promedio (USD) por tipo de pago")
ax.set_ylabel("USD")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_propina_por_pago.png", dpi=150)
plt.close(fig)

# Gráfica 4: mapa de calor día x hora
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
ax.set_yticks(range(7))
ax.set_yticklabels(["Dom", "Lun", "Mar", "Mié", "Jue", "Vie", "Sáb"])
ax.set_xticks(range(0, 24))
ax.set_xticklabels(range(0, 24))
ax.set_xlabel("Hora")
ax.set_title("Demanda por día y hora")
fig.colorbar(im, ax=ax, label="Viajes")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_mapa_calor.png", dpi=150)
plt.close(fig)

# Gráfica 5: top zonas
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(tz["PULocationID"].astype(str), tz["viajes"], color="#9467bd")
ax.invert_yaxis()
ax.set_title("Top 10 zonas de recogida (ID de zona)")
ax.set_xlabel("Número de viajes")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_top_zonas.png", dpi=150)
plt.close(fig)

print(f"\nListo. Gráficas guardadas en la carpeta '{OUT_DIR}/'.")
spark.stop()
