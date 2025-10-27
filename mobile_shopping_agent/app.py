from flask import Flask, render_template, request, jsonify, send_file, abort
from agent import ShoppingChatAgent
from data_loader import DataLoader
from io import BytesIO
import os
from functools import wraps

app = Flask(__name__)

# Initialize data loader with CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_mobiles.csv')
data_loader = DataLoader(csv_path)

# Initialize the shopping agent
agent = ShoppingChatAgent()

# Error handling
def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred'}), 500
    return wrapper

@app.route('/')
def home():
    return render_template('index.html')

# Chat API
@app.route('/api/chat', methods=['POST'])
@handle_errors
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({
            'response': "I didn't catch that. Could you please rephrase?",
            'suggestions': [],
            'products': []
        })
    
    # Process the message with our agent
    response = agent.process_message(user_message)
    return jsonify(response)

# Phone endpoints
@app.route('/api/phones', methods=['GET', 'POST'])
@handle_errors
def handle_phones():
    if request.method == 'GET':
        # Get query parameters
        brand = request.args.get('brand')
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)
        min_ram = request.args.get('min_ram', type=int)
        min_storage = request.args.get('min_storage', type=int)
        os_type = request.args.get('os_type')
        
        # Build filters
        filters = {}
        if brand: filters['brand'] = brand
        if min_price is not None: filters['min_price'] = min_price
        if max_price is not None: filters['max_price'] = max_price
        if min_ram is not None: filters['min_ram'] = min_ram
        if min_storage is not None: filters['min_storage'] = min_storage
        if os_type: filters['os_type'] = os_type
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        skip = (page - 1) * per_page
        
        # Get phones with pagination
        phones = data_loader.search_phones(filters, limit=per_page, skip=skip)
        total = data_loader.db.phones.count_documents(filters) if filters else data_loader.db.phones.count_documents({})
        
        return jsonify({
            'data': phones,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
    
    elif request.method == 'POST':
        # Create a new phone
        phone_data = request.get_json()
        if not phone_data or 'brand' not in phone_data or 'model' not in phone_data:
            return jsonify({'error': 'Brand and model are required'}), 400
            
        phone = data_loader.add_phone(phone_data)
        return jsonify(phone), 201

@app.route('/api/phones/<phone_id>', methods=['GET', 'PUT', 'DELETE'])
@handle_errors
def handle_phone(phone_id):
    if request.method == 'GET':
        phone = data_loader.get_phone_details(phone_id=phone_id)
        if not phone:
            return jsonify({'error': 'Phone not found'}), 404
        return jsonify(phone)
    
    elif request.method == 'PUT':
        update_data = request.get_json()
        if not update_data:
            return jsonify({'error': 'No update data provided'}), 400
            
        phone = data_loader.update_phone(phone_id, update_data)
        if not phone:
            return jsonify({'error': 'Phone not found'}), 404
            
        return jsonify(phone)
    
    elif request.method == 'DELETE':
        if data_loader.delete_phone(phone_id):
            return '', 204
        return jsonify({'error': 'Phone not found'}), 404

# Image endpoints
@app.route('/api/phones/<phone_id>/images', methods=['GET', 'POST'])
@handle_errors
def handle_phone_images(phone_id):
    if request.method == 'GET':
        # Get list of image IDs for the phone
        phone = data_loader.get_phone_details(phone_id=phone_id)
        if not phone:
            return jsonify({'error': 'Phone not found'}), 404
            
        return jsonify({
            'images': phone.get('images', [])
        })
    
    elif request.method == 'POST':
        # Upload a new image
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Read image data
        image_data = image_file.read()
        content_type = image_file.content_type or 'image/jpeg'
        
        # Add image to phone
        image_id = data_loader.add_phone_image(phone_id, image_data, content_type)
        if not image_id:
            return jsonify({'error': 'Failed to add image'}), 500
            
        return jsonify({
            'id': image_id,
            'url': f'/api/images/{image_id}'
        }), 201

@app.route('/api/images/<image_id>')
@handle_errors
def get_image(image_id):
    # Get the phone ID from query params
    phone_id = request.args.get('phone_id')
    if not phone_id:
        return jsonify({'error': 'Phone ID is required'}), 400
    
    # Get thumbnail if requested
    thumbnail = request.args.get('thumbnail', '').lower() in ('true', '1', 't')
    
    # Get image data
    image_data = data_loader.get_phone_image(phone_id, image_id, thumbnail=thumbnail)
    if not image_data:
        return jsonify({'error': 'Image not found'}), 404
        
    return send_file(
        BytesIO(image_data['data']),
        mimetype=image_data['content_type']
    )

@app.route('/api/images/<image_id>', methods=['DELETE'])
@handle_errors
def delete_image(image_id):
    # Get the phone ID from query params
    phone_id = request.args.get('phone_id')
    if not phone_id:
        return jsonify({'error': 'Phone ID is required'}), 400
    
    if data_loader.delete_phone_image(phone_id, image_id):
        return '', 204
    return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
