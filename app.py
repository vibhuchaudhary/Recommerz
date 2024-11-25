import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

# Load dataset
data = pd.read_csv('data/grocery_data.csv')  # Replace with your dataset
rest_data = pd.read_csv('data/restocking_data.csv')
# Example columns: UserID, ProductID, Quantity, PurchaseDate, OrderID

# --- Clustering Functionality ---
def perform_clustering(data, n_clusters=5):
    user_product_matrix = data.pivot_table(index='UserID', columns='ProductID', values='Quantity', fill_value=0)
    
    # Standardize data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(user_product_matrix)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(standardized_data)
    
    # Add cluster labels to users
    user_clusters = pd.DataFrame({'UserID': user_product_matrix.index, 'Cluster': clusters})
    return user_clusters

def prepare_prediction_model(data):
    # Group data to extract features
    cart_data = data.groupby(['OrderID', 'UserID'])['ProductID'].apply(list).reset_index()
    
    # Encode products
    all_products = list(set(data['ProductID']))
    product_encoder = {product: idx for idx, product in enumerate(all_products)}

    def encode_products(products):
        return [product_encoder.get(p, -1) for p in products if p in product_encoder]

    # Prepare features and target
    X, y = [], []
    for _, row in cart_data.iterrows():
        products = row['ProductID']
        if len(products) > 1:
            # Use first products as features, last product as target
            features = encode_products(products[:-1])
            target = product_encoder.get(products[-1], -1)
            
            # Ensure features are not empty
            if features and target != -1:
                # Pad or truncate features to a fixed length
                max_features = 10  # Adjust based on your data
                features = (features + [0] * max_features)[:max_features]
                X.append(features)
                y.append(target)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Train model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    return clf, product_encoder


# --- Restocking Functionality ---
def calculate_restocking(data):
    # Ensure PurchaseDate column exists and is datetime
    if 'PurchaseDate' not in data.columns:
        st.error("PurchaseDate column is missing from the dataset!")
        return pd.DataFrame()  # Return empty dataframe if column missing

    # Convert PurchaseDate to datetime if not already
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'], errors='coerce')

    # Drop rows with invalid dates
    data = data.dropna(subset=['PurchaseDate'])

    # Check if required columns exist
    required_columns = ['UserID', 'ProductID', 'PurchaseDate']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        return pd.DataFrame()

    # Sort and calculate restocking
    try:
        data = data.sort_values(by=['UserID', 'ProductID', 'PurchaseDate'])
        
        # Calculate average interval for restocking
        avg_intervals = (
            data.groupby('ProductID')['PurchaseDate']
            .apply(lambda x: x.diff().dt.days.dropna().mean())
            .reset_index(name='AvgInterval')
        )
        
        # Calculate next restocking date
        data = data.merge(avg_intervals, on='ProductID', how='left')
        data['NextRestock'] = data.apply(
            lambda row: (
                row['PurchaseDate'] + pd.to_timedelta(max(0, row['AvgInterval']), unit='days') 
                if pd.notna(row['AvgInterval']) else None
            ),
            axis=1
        )
        
        restocking_data = data.groupby('ProductID').agg({
            'NextRestock': 'max'
        }).reset_index()
        
        return restocking_data

    except Exception as e:
        st.error(f"Error in restocking calculation: {e}")
        return pd.DataFrame()

# Initialize clustering and prediction models
st.title("Grocery Recommendation System")

# Perform clustering
st.subheader("User Clustering")
user_clusters = perform_clustering(data)
st.write("Clustered Users:")
st.dataframe(user_clusters)

# Prepare prediction model
clf, product_encoder = prepare_prediction_model(data)
reverse_encoder = {v: k for k, v in product_encoder.items()}  # Reverse mapping for predictions

# Calculate restocking data
restocking_data = calculate_restocking(rest_data)

# Cart management: Initialize session state for cart if not already initialized
if 'cart' not in st.session_state:
    st.session_state.cart = []

def get_dynamic_recommendations(cart, clf, product_encoder, reverse_encoder, max_recommendations=5):
    cart_product_ids = [item['ProductID'] for item in cart]
    
    # Encode current cart products
    encoded_cart = []
    for pid in cart_product_ids:
        if pid in product_encoder:
            encoded_cart.append(product_encoder[pid])
    
    recommendations = []
    if encoded_cart:
        # Pad or truncate encoded cart
        max_features = 10
        encoded_cart_padded = (encoded_cart + [0] * max_features)[:max_features]
        
        try:
            # Get probability predictions for all possible products
            prob_predictions = clf.predict_proba([encoded_cart_padded])[0]
            
            # Get top N recommendations
            top_indices = prob_predictions.argsort()[-max_recommendations:][::-1]
            recommendations = [
                {
                    'ProductID': reverse_encoder.get(idx, "Unknown"),
                    'Probability': prob_predictions[idx]
                } 
                for idx in top_indices
            ]
        except Exception as e:
            st.error(f"Recommendation error: {e}")
    
    return recommendations

# --- Add to Cart ---
def add_to_cart(product_id):
    st.session_state.cart.append({'ProductID': product_id, 'AddedTime': datetime.now()})
    st.success(f"Added Product ID {product_id} to cart!")

# Display products
st.subheader("Product Catalog")
categories = data.groupby("Category")["ProductID"].apply(list).to_dict()

for category, products in categories.items():
    with st.expander(category.title()):
        cols = st.columns(3)
        for idx, product in enumerate(products):
            with cols[idx % 3]:
                st.markdown(f"**Product ID:** {product}")
                # Use category, product, and index for a unique key
                unique_key = f"add_{category}_{product}_{idx}"
                if st.button(f"Add {product}", key=unique_key):
                    add_to_cart(product)

# --- Display Cart ---
st.subheader("Your Cart")
if st.session_state.cart:
    cart_df = pd.DataFrame(st.session_state.cart)
    st.table(cart_df)

    # Dynamic Recommendations
    st.subheader("Recommendations")
    recommendations = get_dynamic_recommendations(
        st.session_state.cart, 
        clf, 
        product_encoder, 
        reverse_encoder
    )
    
    if recommendations:
        recommendation_df = pd.DataFrame(recommendations)
        st.table(recommendation_df)
        
        # Add to cart buttons for recommendations
        cols = st.columns(len(recommendations))
        for i, rec in enumerate(recommendations):
            with cols[i]:
                if st.button(f"Add {rec['ProductID']}", key=f"rec_{rec['ProductID']}"):
                    add_to_cart(rec['ProductID'])
    else:
        st.warning("No recommendations available")

def get_user_restocking_data(data, user_id):
    """
    Filter restocking data for a specific user and calculate restocking recommendations.
    """
    # Ensure PurchaseDate is in datetime format
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'], errors='coerce')

    # Filter for specific user
    user_data = data[data['UserID'] == user_id]
    if user_data.empty:
        st.warning(f"No data found for User ID: {user_id}")
        return pd.DataFrame()

    # Calculate restocking recommendations
    try:
        user_restocking_data = calculate_restocking(user_data)
        return user_restocking_data
    except Exception as e:
        st.error(f"Error in restocking calculation: {e}")
        return pd.DataFrame()

# Streamlit UI for Personalized Restocking
st.title("Personalized Restocking Recommendations")

# User ID input
user_id = st.text_input("Enter your User ID (e.g., U001, U002)")

if user_id:
    try:
        # Get user-specific restocking data
        user_restocking_data = get_user_restocking_data(rest_data, user_id)

        if not user_restocking_data.empty:
            st.subheader(f"Restocking Recommendations for User ID: {user_id}")
            st.dataframe(user_restocking_data[['ProductID', 'NextRestock']])
        else:
            st.info("No restocking recommendations available for this user.")
    except Exception as e:
        st.error(f"Error generating restocking data: {e}")

# Streamlit UI for Personalized Restocking
st.title("Personalized Restocking Recommendations")

# User ID input with a unique key
user_id = st.text_input("Enter your User ID (e.g., U001, U002)", key="restocking_user_input")

if user_id:
    try:
        # Get user-specific restocking data
        user_restocking_data = get_user_restocking_data(rest_data, user_id)

        if not user_restocking_data.empty:
            st.subheader(f"Restocking Recommendations for User ID: {user_id}")
            st.dataframe(user_restocking_data[['ProductID', 'NextRestock']])
        else:
            st.info("No restocking recommendations available for this user.")
    except Exception as e:
        st.error(f"Error generating restocking data: {e}")


# --- Restocking ---
st.subheader("Restocking Data")
if st.button("Show Restocking Data"):
    st.table(restocking_data)
