import streamlit as st
import redis
import json
import pandas as pd
import time
import numpy as np
import os

# Connect to Redis
# Using 'redis' as hostname defined in docker-compose, or localhost
r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0, decode_responses=True)

st.set_page_config(page_title="Dynamic Pricing & Recommender", layout="wide")

st.title("Real-Time Dynamic Pricing & Recommender Engine")

# Placeholder for auto-refresh
placeholder = st.empty()

# Sidebar for User Interaction
st.sidebar.header("User Personalization")
user_input = st.sidebar.text_input("Enter User ID", "541312140") # Default ID from dataset example

def get_product_price(product_id):
    price = r.get(f"price_prod_{product_id}")
    return float(price) if price else None

def get_user_recommendations(user_id):
    recs = r.get(f"rec_user_{user_id}")
    if recs:
        return json.loads(recs)
    return []

def main():
    while True:
        with placeholder.container():
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Live Pricing Monitor")
                # Fetch random keys or known keys to display
                # Since we don't have a list of all keys easily, we scan for some
                keys = r.keys("price_prod_*")
                
                if keys:
                    # Pick 5 random keys
                    selected_keys = np.random.choice(keys, min(5, len(keys)), replace=False)
                    data = []
                    
                    for key in selected_keys:
                        prod_id = key.replace("price_prod_", "")
                        price = float(r.get(key))
                        # Simulate a base price for comparison (random variation or static if stored)
                        # Here assuming base price is slightly lower/different to show "dynamic" effect
                        base_price = price * 0.95 
                        
                        data.append({
                            "Product ID": prod_id,
                            "Dynamic Price": price,
                            "Base Price": base_price
                        })
                        
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    st.line_chart(df.set_index("Product ID")[["Dynamic Price", "Base Price"]])
                else:
                    st.info("Waiting for pricing data in Redis...")

            with col2:
                st.subheader("User Recommendations")
                if user_input:
                    st.write(f"Recommendations for User: **{user_input}**")
                    recs = get_user_recommendations(user_input)
                    
                    if recs:
                        st.success(f"Top 5 Products: {recs}")
                        # Could add fake images/details here
                    else:
                        st.warning("No recommendations found for this user (or model not trained yet).")
                else:
                    st.info("Enter a User ID in sidebar.")
                    
            time.sleep(2)

if __name__ == "__main__":
    main()
