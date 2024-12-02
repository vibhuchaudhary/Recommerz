import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('grocery_data.csv')  

user_product_matrix = data.pivot_table(index='UserID', columns='ProductID', values='Quantity', fill_value=0)
x

scaler = StandardScaler()
standardized_data = scaler.fit_transform(user_product_matrix)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(standardized_data)

user_clusters = pd.DataFrame({'UserID': user_product_matrix.index, 'Cluster': clusters})
print(user_clusters.head())
