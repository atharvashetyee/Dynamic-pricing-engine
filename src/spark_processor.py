from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
import xgboost as xgb
import redis
import json
import numpy as np
import pandas as pd
import os

# Define Schema corresponding to the CSV columns
schema = StructType([
    StructField("event_time", StringType()),
    StructField("event_type", StringType()),
    StructField("product_id", StringType()),
    StructField("category_code", StringType()),
    StructField("brand", StringType()),
    StructField("price", StringType()), # Reading as string to avoid parsing errors, then cast
    StructField("user_id", StringType()),
    StructField("user_session", StringType())
])

def process_batch(batch_df, batch_id):
    if batch_df.count() == 0:
        return
        
    print(f"Processing batch {batch_id}...")
    
    # Collect aggregated data to driver for model inference
    # Note: In high production, use UDFs or pandas_udf for distributed inference.
    # checking simplistic 'collect' for demo purposes as requested logic
    
    # For prediction we need: clicks_in_window, cart_adds, base_price
    # The aggregation below extracts 'demand_count'. 
    # We need to more closely match the features the model was trained on.
    # Model features: ['clicks_in_window', 'cart_adds', 'base_price']
    
    # But window aggregation usually gives us just counts per window.
    # Let's approximate: 
    # clicks_in_window = demand_count (assuming most events are views)
    # cart_adds = 0 (simplification for stream if we don't differentiate in agg)
    # base_price = average price in batch or lookup
        
    data = batch_df.toPandas()
    
    if data.empty:
        return
        
    # Load Model (Load once in reality, but for simplicity here loading or Global)
    # Ideally load model outside and broadcast, but xgboost model object not easily broadcasted sometimes.
    # We can load it here.
    model = xgb.XGBRegressor()
    try:
        model_path = os.path.join(os.getenv('MODEL_DIR', 'models'), 'pricing_model.json')
        model.load_model(model_path)
    except Exception as e:
        print(f"Model not found yet at {model_path}, skipping inference. Error: {e}")
        return

    # Prepare input for model
    # data has columns: window, product_id, demand_count, avg_price
    
    # Feature map
    data['clicks_in_window'] = data['demand_count']
    data['cart_adds'] = 0 # Placeholder if not streaming specific event types separately
    data['base_price'] = data['avg_price']
    
    X = data[['clicks_in_window', 'cart_adds', 'base_price']]
    
    # Predict
    # Result is the "Target: price", which we interpret as the Dynamic Price
    predictions = model.predict(X)
    data['dynamic_price'] = predictions
    
    # Write to Redis
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)
    pipe = r.pipeline()
    
    for index, row in data.iterrows():
        product_id = row['product_id']
        price = float(row['dynamic_price'])
        pipe.set(f"price_prod_{product_id}", price)
        
    pipe.execute()
    print(f"Updated prices for {len(data)} products in Redis.")

def main():
    spark = SparkSession.builder \
        .appName("DynamicPricingStream") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    
    kafka_broker = os.getenv('KAFKA_BROKER', 'localhost:9092')

    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_broker) \
        .option("subscribe", "clickstream") \
        .load()
        
    # Parse JSON
    parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    # Preprocessing
    # Cast event_time to Timestamp
    # Format in dataset: 2019-10-01 00:00:00 UTC
    parsed_df = parsed_df.withColumn("event_time", col("event_time").cast(TimestampType()))
    parsed_df = parsed_df.withColumn("price", col("price").cast(DoubleType()))
    
    # Watermark
    parsed_df = parsed_df.withWatermark("event_time", "1 minutes")
    
    # Aggregation
    # Group by product_id and Window
    agg_df = parsed_df.groupBy(
        window(col("event_time"), "1 minute"),
        col("product_id")
    ).agg(
        count("*").alias("demand_count"),
        expr("avg(price)").alias("avg_price")
    )
    
    # Write Stream
    query = agg_df.writeStream \
        .outputMode("update") \
        .foreachBatch(process_batch) \
        .start()
        
    query.awaitTermination()

if __name__ == "__main__":
    main()
