import os
import uuid
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import img_proc
from color_database import color_db

app = Flask(__name__, static_folder='public', template_folder='templates')

# Configure upload folders
UPLOAD_FOLDER = os.path.join('public', 'images')
EDITED_FOLDER = os.path.join('public', 'edited')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EDITED_FOLDER'] = EDITED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EDITED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # List available sample images
    images = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    images.sort()
    
    # List available patterns
    patterns_dir = os.path.join('public', 'patterns')
    os.makedirs(patterns_dir, exist_ok=True)
    patterns = [f for f in os.listdir(patterns_dir) if allowed_file(f)]
    patterns.sort()
    
    return render_template('index.html', images=images, patterns=patterns)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # Generate unique filename to avoid caching issues
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"upload_{uuid.uuid4().hex[:8]}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image parameter'}), 400
        
    image_name = data['image']
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    
    if not os.path.exists(input_path):
        return jsonify({'error': 'Image not found'}), 404
        
    # Generate output filename
    output_name = f"edited_{uuid.uuid4().hex[:8]}_{image_name}"
    output_path = os.path.join(app.config['EDITED_FOLDER'], output_name)
    
    try:
        if 'color' in data and data['color']:
            # Expecting "R,G,B" string
            color = [int(c.strip()) for c in data['color'].split(',')]
            img_proc.changeColor(input_path, output_path, new_color=color)
        elif 'pattern' in data and data['pattern']:
            pattern_name = data['pattern']
            pattern_path = os.path.join('public', 'patterns', pattern_name)
            img_proc.changeColor(input_path, output_path, pattern_path=pattern_path)
        else:
            return jsonify({'error': 'No color or pattern provided'}), 400
            
        return jsonify({'success': True, 'edited_image': output_name})
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

@app.route('/search_colors', methods=['GET'])
def search_colors():
    """
    Поиск цветов по коду
    Query params:
        q - поисковый запрос (например "F00")
        limit - максимальное количество результатов (по умолчанию 20)
    """
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))

    try:
        results = color_db.search(query, limit=limit)
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'total': color_db.get_count()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_color/<code>', methods=['GET'])
def get_color(code):
    """Получить конкретный цвет по коду"""
    try:
        color = color_db.get_by_code(code)
        if color:
            return jsonify({'success': True, 'color': color})
        else:
            return jsonify({'success': False, 'error': 'Color not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=======================================")
    print("Starting AI Wall Paint Visualizer Server")
    print("=======================================")

    # Run warmup synchronously so the server only starts accepting requests
    # when the model is 100% loaded and warmed up into GPU/CPU memory
    print("Step 1/3: Loading color database...")
    try:
        color_db.load()
        print(f"Successfully loaded {color_db.get_count()} colors from database")
    except Exception as e:
        print(f"Warning: Could not load color database: {e}")
        print("Color search will not be available")

    print("\nStep 2/3: Loading and warming up SegFormer AI Model...")
    img_proc.warmup_model()

    print("\nStep 3/3: Starting Web Server...")
    print("Server ready! Open http://localhost:8000 in your browser.")
    app.run(host='0.0.0.0', port=8000, debug=False)
