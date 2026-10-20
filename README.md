# Tarea 3 - Procesamiento en batch con Apache Spark

Curso Big Data (202016911) - UNAD

## Problema

Analizar la demanda de taxis amarillos de Nueva York y responder:
¿cómo varían la cantidad de viajes, la tarifa y la propina según la hora del día,
el día de la semana y el tipo de pago?

## Dataset

- **NYC Yellow Taxi Trip Records** (TLC, Nueva York), enero de 2024, formato Parquet.
- Fuente original: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- El script lo descarga automáticamente a la carpeta `data/` la primera vez que se ejecuta
  (no se sube al repositorio por su tamaño).

## Qué hace el código (`batch_taxi.py`)

1. **Carga** el dataset desde la fuente original con Spark.
2. **Limpieza:** elimina nulos en columnas clave, duplicados y valores imposibles
   (distancias, tarifas o fechas fuera de rango, duraciones y velocidades irreales).
3. **Transformación:** crea columnas derivadas (hora, día de la semana, duración en minutos,
   velocidad, porcentaje de propina, tipo de pago legible).
4. **EDA:** estadísticas descriptivas, viajes por hora y por día, propina por tipo de pago,
   top 10 zonas de recogida, correlaciones y un ejemplo con RDDs (`map`, `reduceByKey`, `take`).
5. **Visualización:** guarda 5 gráficas en la carpeta `resultados/`.

## Requisitos

- Python 3.9 a 3.11 (versiones compatibles con PySpark 3.5)
- Java 8, 11 o 17 (necesario para Spark)
- Dependencias: `pip install -r requirements.txt`

## Ejecución

```bash
git clone https://github.com/petanorivaldo-source/spark-batch-tarea.3.git
cd spark-batch-tarea.3
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
python batch_taxi.py
```

Los resultados (tablas) salen por consola y las gráficas quedan en `resultados/`.

## Autor

Rivaldo Andrés Palmett Petano - Ingeniería de Sistemas, UNAD
