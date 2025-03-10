from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
def create_model(num_classes=5):
    model = models.efficientnet_b3(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

model = create_model(num_classes=5)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the class names
class_names = ['0: No_Dr', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferative DR']

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            predicted_class = class_names[preds[0].item()]
        
        print(f"Predicted class: {predicted_class}")
        return jsonify({'class': predicted_class}), 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)