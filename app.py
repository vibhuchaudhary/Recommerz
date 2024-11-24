from faker import Faker
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from flask import Flask, render_template, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Initialize Faker for Dummy Data
faker = Faker()

# Global Variables
NUM_USERS = 100
NUM_PRODUCTS = 50
NUM_TRANSACTIONS = 500

# Step 1: Generate Dummy Data
def generate_dummy_data():
    # Generate Users, Products, and Transactions
    users = [{"user_id": i, "age": random.randint(18, 60), "location": faker.city()} for i in range(NUM_USERS)]
    products = [{"product_id": i, "name": faker.word(), "category": faker.word()} for i in range(NUM_PRODUCTS)]
    transactions = [
        {
            "user_id": random.choice(users)["user_id"],
            "product_id": random.choice(products)["product_id"],
            "timestamp": faker.date_time_this_year()
        }
        for _ in range(NUM_TRANSACTIONS)
    ]

    # Save as CSV
    pd.DataFrame(users).to_csv("users.csv", index=False)
    pd.DataFrame(products).to_csv("products.csv", index=False)
    pd.DataFrame(transactions).to_csv("transactions.csv", index=False)
    print("Dummy data generated and saved.")

# Step 2: Train Smart Suggestions Model
def train_smart_suggestions(data_path="transactions.csv"):
    transactions = pd.read_csv(data_path)
    basket = transactions.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(basket)
    return model, basket

def recommend_smart(model, basket, user_id, num_recommendations=5):
    user_vector = basket.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=num_recommendations + 1)
    return basket.index[indices.flatten()].tolist()[1:]  # Exclude the user themselves

# Step 3: Train Association Rules Model
def train_association_rules(data_path="transactions.csv"):
    transactions = pd.read_csv(data_path)
    basket = transactions.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
    basket = basket.applymap(lambda x: x > 0)  # Convert to True/False
    
    # Generate frequent itemsets using apriori
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    
    # Generate association rules with a minimum lift threshold of 1 and a number of itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=10)
    return rules


def recommend_together(rules, product_id, num_recommendations=5):
    suggestions = rules[rules['antecedents'].apply(lambda x: product_id in x)]
    return suggestions.sort_values('lift', ascending=False).head(num_recommendations)

# Step 4: Train Recurring Orders Model
def train_recurring_orders(data_path="transactions.csv"):
    transactions = pd.read_csv(data_path)
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    product_time_series = transactions.groupby(
        [transactions['timestamp'].dt.date, 'product_id']
    ).size().unstack(fill_value=0)
    models = {}
    for product_id in product_time_series.columns:
        models[product_id] = ExponentialSmoothing(
            product_time_series[product_id], seasonal='add', seasonal_periods=7
        ).fit()
    return models

def predict_recurring_orders(models, product_id, future_periods=30):
    if product_id not in models:
        return []
    return models[product_id].forecast(steps=future_periods).tolist()

# Step 5: Initialize models
def initialize_models():
    print("Generating dummy data...")
    generate_dummy_data()
    print("Training models...")
    smart_model, smart_basket = train_smart_suggestions()
    association_rules_model = train_association_rules()
    recurring_models = train_recurring_orders()
    return smart_model, smart_basket, association_rules_model, recurring_models

smart_model, smart_basket, association_rules_model, recurring_models = initialize_models()

# Step 6: API Endpoint to handle user requests
@app.route("/", methods=["GET", "POST"])
def recommend():
    if request.method == "POST":
        user_id = int(request.form.get("user_id"))
        recommendations = recommend_smart(smart_model, smart_basket, user_id)
        return render_template("index.html", user_id=user_id, recommendations=recommendations)
    return render_template("index.html", user_id=None, recommendations=None)

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
