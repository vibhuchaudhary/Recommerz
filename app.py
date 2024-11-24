import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Load the data
data = pd.read_csv('data/grocery_data.csv')

purchase_data = pd.read_csv('data/restocking_data.csv')

# Preprocess data
def preprocess_data(data):
    # Pivot table for user-product interactions
    product_user_matrix = data.pivot_table(index='UserID', columns='ProductID', values='Quantity', fill_value=0)

    # One-hot encode the product categories
    categories = data[['ProductID', 'Category']].drop_duplicates()
    encoder = OneHotEncoder()
    category_encoded = encoder.fit_transform(categories[['Category']]).toarray()
    category_df = pd.DataFrame(category_encoded, index=categories['ProductID'], columns=encoder.get_feature_names_out())

    # Combine product-user matrix with category encoding
    product_user_matrix_T = product_user_matrix.T
    combined_matrix = product_user_matrix_T.merge(category_df, left_index=True, right_index=True, how='left').fillna(0)

    return product_user_matrix, combined_matrix

# Preprocess the dataset
product_user_matrix, combined_matrix = preprocess_data(data)

# Generate similarity matrix
similarity_matrix = cosine_similarity(combined_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=combined_matrix.index, columns=combined_matrix.index)

# Get unique products for dropdown
unique_products = sorted(data['ProductID'].unique())

# Calculate restocking patterns
import pandas as pd
import numpy as np

def calculate_restocking(data):
    # Convert PurchaseDate to datetime
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])
    
    # Calculate AvgInterval for each ProductID
    avg_intervals = (
        data.groupby('ProductID')['PurchaseDate']
        .apply(lambda x: (x.diff().dt.days).mean() if len(x) > 1 else None)  # Calculate mean difference if more than one purchase
        .reset_index(name='AvgInterval')
    )
    
    # Merge AvgInterval into the original data
    data = data.merge(avg_intervals, on='ProductID', how='left')
    
    # Calculate NextRestock
    data['NextRestock'] = data.apply(
        lambda row: row['PurchaseDate'] + pd.to_timedelta(row['AvgInterval'], unit='days') 
        if pd.notna(row['AvgInterval']) else "Insufficient Data", axis=1
    )
    
    # Handle products with only one purchase (set AvgInterval and NextRestock to "Insufficient Data")
    data['AvgInterval'] = data['AvgInterval'].fillna("Insufficient Data")
    data['NextRestock'] = data['NextRestock'].fillna("Insufficient Data")
    
    # Now, we create the restocking data in the desired format
    restocking_data = []
    for (user_id, product_id), group in data.groupby(['UserID', 'ProductID']):
        last_purchase = group['PurchaseDate'].max()
        avg_interval = group['AvgInterval'].iloc[0]  # Take the first value (same for the entire group)
        next_restock = group['NextRestock'].iloc[0]  # Take the first value (same for the entire group)

        restocking_data.append({
            'UserID': user_id,
            'ProductID': product_id,
            'LastPurchase': last_purchase,
            'AvgInterval': avg_interval,
            'NextRestock': next_restock
        })
    
    # Return as a DataFrame
    return pd.DataFrame(restocking_data)

# Load the existing data (CSV file)
purchase_data = pd.read_csv('data/restocking_data.csv')

# Calculate the restocking data
restocking_data = calculate_restocking(purchase_data)

# Save the updated data back to a CSV file
restocking_data.to_csv('data/updated_restocking_data.csv', index=False)

print("Restocking data has been updated and saved successfully.")

# Recommendation logic
def recommend_products(product_id, top_n=3):
    if product_id not in similarity_df.index:
        return []
    similar_products = similarity_df[product_id].sort_values(ascending=False).iloc[1:top_n + 1]
    return list(similar_products.index)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations')
def rerecommendation():
    return render_template('recommendation.html', products=unique_products)

# Recommendation API
@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({'error': 'Product ID is required'}), 400

    recommendations = recommend_products(product_id)
    return jsonify({'recommendations': recommendations})

# Define route to show restocking data
@app.route('/restocking')
def show_restocking():
    # Load the updated CSV data
    restocking_data = pd.read_csv('data/updated_restocking_data.csv')
    
    # Convert DataFrame to HTML table format
    
    # Render the HTML template and pass the table data
    return render_template('restocking.html', data=restocking_data)


# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
