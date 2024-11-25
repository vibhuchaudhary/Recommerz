import random
import csv
from datetime import datetime, timedelta

# Categories and products
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

# Generate random data
def generate_random_data():
    user_id = f"U{random.randint(1, 33):03d}"  # User IDs: U001, U002, ...
    category = random.choice(list(categories.keys()))
    product = random.choice(categories[category])
    product_id = f"{product}"  # ProductID + UserID
    quantity = random.randint(1, 50)  # Random quantity between 1 and 50
    order_id = f"O{random.randint(1000, 9999)}"  # Order IDs: O1000, O1001, ...
    purchase_date = datetime.now() - timedelta(days=random.randint(0, 365))  # Random date within the past year
    purchase_date_str = purchase_date.strftime('%Y-%m-%d')
    return [user_id, product_id, quantity, order_id, category, purchase_date_str]

# Generate 200 entries
entries = [generate_random_data() for _ in range(200)]

# Write to CSV
output_file = "restocking_entries.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['UserID', 'ProductID', 'Quantity', 'OrderID', 'Category', 'PurchaseDate'])
    writer.writerows(entries)

print(f"200 restocking entries have been saved to {output_file}.")
