from sklearn.metrics.pairwise import cosine_similarity

# Calculate item similarity
item_similarity = cosine_similarity(user_product_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_product_matrix.columns, columns=user_product_matrix.columns)

# Recommend top N products based on a specific product
def recommend_products(product_id, n=5):
    similar_products = item_similarity_df[product_id].sort_values(ascending=False).head(n+1)
    return similar_products.index[1:]

# Example usage
print(recommend_products(product_id='Milk', n=5))
