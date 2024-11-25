import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

categories = {
    'fresh fruits': ['Fresh_Apples', 'Fresh_Bananas', 'Fresh_Oranges', 'Fresh_Mangoes', 'Fresh_Grapes', 'Fresh_Strawberries', 'Fresh_Blueberries', 'Fresh_Peaches'],
    'fresh vegetables': ['Fresh_Carrots', 'Fresh_Spinach', 'Fresh_Tomatoes', 'Fresh_Broccoli', 'Fresh_Peppers', 'Fresh_Cucumbers', 'Fresh_Lettuce', 'Fresh_Kale'],
    'milk': ['Whole_Milk', 'Skim_Milk', 'Almond_Milk', 'Soy_Milk', 'Oat_Milk', 'Coconut_Milk', 'Lactose_Free_Milk'],
    'packaged cheese': ['Cheddar_Cheese', 'Mozzarella_Cheese', 'Swiss_Cheese', 'Parmesan_Cheese', 'Gouda_Cheese', 'Blue_Cheese', 'Feta_Cheese'],
    'bread': ['White_Bread', 'Wheat_Bread', 'Sourdough_Bread', 'Rye_Bread', 'Multigrain_Bread', 'Ciabatta', 'Baguette'],
    'meat counter': ['Ground_Beef', 'Ribeye_Steak', 'Pork_Chops', 'Chicken_Breast', 'Lamb_Chops', 'Turkey_Breast', 'Bacon'],
    'poultry counter': ['Chicken_Breast', 'Turkey_Breast', 'Duck_Breast', 'Whole_Chicken'],
    'yogurt': ['Greek_Yogurt', 'Plain_Yogurt', 'Flavored_Yogurt', 'Low_Fat_Yogurt', 'Organic_Yogurt'],
    'frozen foods': ['Frozen_Pizza', 'Frozen_Vegetables', 'Ice_Cream', 'Frozen_Chicken', 'Frozen_Fish'],
    'pantry staples': ['Rice', 'Pasta', 'Olive_Oil', 'Salt', 'Sugar', 'Flour', 'Canned_Tomatoes']
}

def generate_complex_order_data(num_orders=200, max_items_per_order=5):
    data = []
    users = [f'U{str(i).zfill(3)}' for i in range(1, 31)]  # 30 unique users
    order_counter = 1000
    order_index_tracker = {}

    for _ in range(num_orders):
        user_id = random.choice(users)
        order_id = f'ORD{order_counter}'
        order_date = datetime(2024, random.randint(9, 11), random.randint(1, 30))
        
        # Ensure unique items in this order
        unique_categories = random.sample(list(categories.keys()), 
                                          random.randint(1, min(len(categories), max_items_per_order)))
        
        for order_index, category in enumerate(unique_categories, 1):
            product_id = random.choice(categories[category])
            quantity = random.randint(1, 4)

            data.append([
                order_id,
                user_id,
                product_id,
                quantity,
                category,
                order_index
            ])

        order_counter += 1

    columns = ['OrderID', 'UserID', 'ProductID', 'Quantity', 'Category', 'OrderIndex']
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate dataset
df = generate_complex_order_data()

# Save to CSV
df.to_csv('expanded_grocery_orders.csv', index=False)
print(df.head(15))
print(f"\nTotal entries: {len(df)}")

# Additional analysis
print("\nOrder Distribution:")
print(df['OrderID'].nunique(), "unique orders")
print(df['UserID'].nunique(), "unique users")
print(df.groupby('Category')['ProductID'].nunique(), "unique products per category")