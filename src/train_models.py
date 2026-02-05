import os
import json
import redis
import pandas as pd
import numpy as np
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split

# Connect to Redis
r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)

def train_pricing_model(df):
    print("Training Pricing Model (XGBoost)...")
    
    # Feature Engineering (Simplified)
    # We will aggregate to simulate product demand features
    # In a real scenario, this would be more complex time-series features
    
    # Create dummy features for demonstration if original data doesn't support complex ones easily
    # or aggregations.
    # Group by product_id to get simpler dataset for pricing model
    
    # Let's create a derived dataset for pricing model
    # Features: base_price (price), demand (count of interactions), category_code (encoded)
    
    # For the purpose of this demo, we'll try to predict 'price' or a multiplier based on demand
    # But usually price is the input and demand is the target or vice versa. 
    # The prompt says: Target: price. Features: clicks_in_window, cart_adds, base_price.
    # Re-reading prompt: "Task A (XGBoost): Pre-process data to create features: clicks_in_window, cart_adds, base_price. Target: price."
    # Wait, predicting price based on base_price? Maybe "optimal price"? 
    # Or maybe predicting the *actual* price transaction happened at? 
    # Let's assume we are learning the relationship between demand signals and price.
    
    # Preprocessing
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Aggregate data to create features per product
    product_stats = df.groupby('product_id').agg({
        'price': 'mean',
        'event_type': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Extract features from event_type dict
    product_stats['clicks_in_window'] = product_stats['event_type'].apply(lambda x: x.get('view', 0))
    product_stats['cart_adds'] = product_stats['event_type'].apply(lambda x: x.get('cart', 0))
    product_stats['base_price'] = product_stats['price'] # Just using mean price as base price
    
    # Target: We'll just use the current price as the target to learn "market price" or similar
    # In a real dynamic pricing, we'd want to maximize revenue, but here we just follow instructions.
    
    features = ['clicks_in_window', 'cart_adds', 'base_price']
    X = product_stats[features]
    y = product_stats['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # Save model
    model.save_model(os.path.join(os.getenv('MODEL_DIR', 'models'), 'pricing_model.json'))
    print(f"Pricing model saved to {os.getenv('MODEL_DIR', 'models')}/pricing_model.json")

def train_recommender_system(spark_session, input_file):
    print("Training Recommender System (Spark ALS)...")
    
    # Load data with Spark
    df = spark_session.read.csv(input_file, header=True, inferSchema=True)
    
    # Filter for interactions (view, cart, purchase)
    # We need to convert user_id and product_id to integers for ALS
    # Assuming user_id and product_id are already integers in the dataset (they look like it in description)
    
    # Select relevant columns
    ratings_df = df.select('user_id', 'product_id', 'event_type')
    
    # Implicit ratings: view=1, cart=2, purchase=3
    from pyspark.sql.functions import when
    ratings_df = ratings_df.withColumn("rating", 
        when(col("event_type") == "view", 1)
        .when(col("event_type") == "cart", 2)
        .when(col("event_type") == "purchase", 3)
        .otherwise(0)
    )
    
    # Drop 0 ratings
    ratings_df = ratings_df.filter(col("rating") > 0)
    
    # Train ALS
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="product_id", ratingCol="rating",
              coldStartStrategy="drop", implicitPrefs=True)
    
    model = als.fit(ratings_df)
    
    # Generate top 5 recommendations for each user
    user_recs = model.recommendForAllUsers(5)
    
    # Save to Redis
    # Collecting to driver might be heavy for large data, but for 2019-Oct.csv (can be large), we might need to limit
    # For this demo, we'll take a subset or just process as is if memory allows. 
    # To be safe for the demo, let's limit to top 100 users output or write to redis in partitions.
    
    print("Saving recommendations to Redis...")
    
    def write_to_redis(row):
        r_conn = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)
        user_id = row.user_id
        recs = [row.recommendations[i].product_id for i in range(len(row.recommendations))]
        r_conn.set(f"rec_user_{user_id}", json.dumps(recs))
        
    # Using foreach to write directly from executors
    # Note: connect to redis inside the function
    try:
        user_recs.foreach(write_to_redis)
        print("Recommendations saved to Redis.")
    except Exception as e:
        print(f"Error saving to Redis: {e}")
        # Fallback for local run without redis container accessible or similar issues
        pass

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.getenv('DATA_DIR', 'dataset'), 'events.csv')
    
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please ensure file exists.")
        # Create dummy data for testing if file missing (optional, but good for robustness)
        exit(1)
        
    # pandas for XGBoost
    # Reading only first 100k rows to avoid OOM in this simple container setup if file is huge (it is ~5GB usually)
    print("Loading data for Pricing Model...")
    df_pd = pd.read_csv(DATA_PATH, nrows=100000) 
    train_pricing_model(df_pd)
    
    # Spark for Recommender
    print("Initializing Spark...")
    spark = SparkSession.builder \
        .appName("RecommenderTraining") \
        .getOrCreate()
        
    train_recommender_system(spark, DATA_PATH)
    
    spark.stop()
