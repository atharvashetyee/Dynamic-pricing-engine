import time
import json
import csv
import datetime
import os
from kafka import KafkaProducer

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

def main():
    topic_name = 'clickstream'
    input_file = os.path.join(os.getenv('DATA_DIR', 'dataset'), 'events.csv')
    kafka_broker = os.getenv('KAFKA_BROKER', 'localhost:9092')
    
    # Wait for Kafka to be ready
    producer = None
    for i in range(10):
        try:
            print(f"Connecting to Kafka at {kafka_broker} (Attempt {i+1})...")
            producer = KafkaProducer(
                bootstrap_servers=[kafka_broker],
                value_serializer=json_serializer
            )
            print("Connected to Kafka!")
            break
        except Exception as e:
            print(f"Connection failed: {e}")
            time.sleep(5)
            
    if not producer:
        print("Failed to connect to Kafka after multiple attempts.")
        return

    print(f"Starting to stream data from {input_file}...")
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ensure serialization of known types
            # row is already a dict of strings from csv
            # We assume event_time is a string in the CSV, so it passes as string directly
            
            producer.send(topic_name, row)
            
            # Simulate delay
            time.sleep(0.1) 

if __name__ == "__main__":
    main()
