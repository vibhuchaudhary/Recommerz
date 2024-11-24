import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Load the data
data = pd.read_csv('data/grocery_data.csv')

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

# Recommendation logic
def recommend_products(product_id, top_n=3):
    if product_id not in similarity_df.index:
        return []
    similar_products = similarity_df[product_id].sort_values(ascending=False).iloc[1:top_n + 1]
    return list(similar_products.index)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html', products=unique_products)

# Recommendation API
@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({'error': 'Product ID is required'}), 400

    recommendations = recommend_products(product_id)
    return jsonify({'recommendations': recommendations})

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
