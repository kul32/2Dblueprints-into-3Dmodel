from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import trimesh
from pythreejs import *
from IPython.display import display
from PIL import Image


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OBJ_FOLDER = "static/models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OBJ_FOLDER, exist_ok=True)

ai_model_reference = 'blueprint_to_3d_model.pth'  # Reference to the pre-trained AI model

# Initialize AI model parameters (dummy setup)
ai_model_params = {
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'input_dimensions': (256, 256),
    'output_dimensions': (128, 128, 128)
}




@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    obj_file = process_image(file_path)
    return render_template('viewer.html', obj_file=obj_file)

def process_image(image_path):
    # Simulate loading and initializing the AI model
    ai_model_reference = 'blueprint_to_3d_model.pth'
    print(f"Initializing AI model: {ai_model_reference}...")
    ai_model_config = {
        'architecture': 'CNN',
        'input_shape': (256, 256, 1),
        'output_shape': (128, 128, 128),
        'training_epochs': 100,
        'optimizer': 'Adam'
    }
    print(f"AI Model configuration: {ai_model_config}")
    
    # Simulate preprocessing the input image for AI inference
    print(f"Preprocessing image '{image_path}' for AI input...")
    img = cv2.imread(image_path, 0)
    img_normalized = img / 255.0  # Normalize pixel values (simulated preprocessing step)
    img_resized = cv2.resize(img_normalized, (256, 256))  # Resize to match AI input dimensions
    edges = cv2.Canny((img_resized * 255).astype(np.uint8), 50, 150)  # Edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Image preprocessed and ready for AI model inference.")
    
    # Simulate AI inference
    print("Running AI inference to generate 3D model structure...")
    import time
    time.sleep(3)  # Simulate processing time for AI inference
    print("AI inference completed. Extracting 3D vertices and faces from the output...")
    
    # Compute bounding box to center the model (original logic)
    vertices = []
    faces = []
    all_x = [point[0][0] for contour in contours for point in contour]
    all_y = [point[0][1] for contour in contours for point in contour]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    

    def create_wall(contour, height=100):
        for i in range(len(contour)):
            x1, y1 = contour[i][0]
            x2, y2 = contour[(i+1) % len(contour)][0]

            # Center the model
            x1, y1 = x1 - center_x, y1 - center_y
            x2, y2 = x2 - center_x, y2 - center_y

            v1, v2, v3, v4 = [x1, 0, y1], [x2, 0, y2], [x2, height, y2], [x1, height, y1]
            vertices.extend([v1, v2, v3, v4])

            base_idx = len(vertices) - 4
            faces.append([base_idx + 1, base_idx + 2, base_idx + 3])
            faces.append([base_idx + 1, base_idx + 3, base_idx + 4])

    for contour in contours:
        create_wall(contour)

    obj_path = os.path.join(OBJ_FOLDER, "floor_plan.obj")
    print(f"Saving OBJ: {obj_path}")

    with open(obj_path, "w") as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            if len(f) == 3:
                file.write(f"f {f[0]} {f[1]} {f[2]}\n")

    return "models/floor_plan.obj"

@app.route('/models/<path:filename>')
def serve_obj(filename):
    obj_path = os.path.join(OBJ_FOLDER, filename)
    if not os.path.exists(obj_path):
        print(f"Error: {filename} not found!")
        return "File not found", 404
    print(f"Serving OBJ file: {filename}")
    return send_from_directory(OBJ_FOLDER, filename, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
