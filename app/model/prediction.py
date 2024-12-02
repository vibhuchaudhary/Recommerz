from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cart_data = data.groupby(['OrderID', 'UserID'])['ProductID'].apply(list).reset_index()
cart_data['TargetProduct'] = cart_data['ProductID'].apply(lambda x: x[-1])  # Last product as target
cart_data['ProductFeatures'] = cart_data['ProductID'].apply(lambda x: x[:-1])  # All except last

all_products = list(set(data['ProductID']))
product_encoder = {product: idx for idx, product in enumerate(all_products)}

def encode_products(products):
    return [product_encoder[p] for p in products]

cart_data['EncodedFeatures'] = cart_data['ProductFeatures'].apply(encode_products)
cart_data['EncodedTarget'] = cart_data['TargetProduct'].map(product_encoder)

X = np.array(cart_data['EncodedFeatures'].tolist())
y = np.array(cart_data['EncodedTarget'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

def predict_cart_items(features):
    encoded_features = encode_products(features)
    return clf.predict([encoded_features])

print(predict_cart_items(['Milk', 'Bread']))
