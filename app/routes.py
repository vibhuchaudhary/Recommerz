from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Home page route
@app.route('/')
def index():
    return render_template('index.html')  # Ensure templates/index.html exists

# API for recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({'error': 'Product ID is required'}), 400
    
    # Replace with your recommendation logic
    recommendations = ['Example Product 1', 'Example Product 2']
    return jsonify({'recommendations': recommendations})
