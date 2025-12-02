from flask import Flask, request, jsonify
from PIL import Image
import torch

from models import (get_classification_model, get_detection_model, get_classes, get_coco_names,
                   detect_ai_generated_content, detect_ai_text, detect_ai_video, detect_objects_in_video_file)
from utils import preprocess_image, draw_bounding_boxes

app = Flask(__name__)

@app.route("/predict/classify", methods=["POST"])
def predict_classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        
        model = get_classification_model()
        classes = get_classes()
        
        processed_image = preprocess_image(image)
        with torch.no_grad():
            outputs = model(processed_image)
        
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        
        results = []
        for i in range(top5_probs.size(0)):
            class_name = classes[top5_indices[i]]
            probability = top5_probs[i].item()
            results.append({"class": class_name, "probability": probability})
            
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/detect", methods=["POST"])
def predict_detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        
        model = get_detection_model()
        coco_names = get_coco_names()
        
        img_tensor = preprocess_image(image, detection=True)
        
        with torch.no_grad():
            predictions = model(img_tensor)

        # The output of the detection model is complex and includes bounding boxes.
        # For an API, you might want to return the bounding box coordinates and labels as JSON.
        
        results = []
        for i in range(len(predictions[0]["boxes"])):
            confidence = predictions[0]["scores"][i].item()
            if confidence > 0.5: # Default confidence threshold
                box = predictions[0]["boxes"][i].detach().numpy().tolist()
                label_index = predictions[0]["labels"][i].item()
                label = coco_names[label_index]
                results.append({"label": label, "confidence": confidence, "box": box})

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect/ai-image", methods=["POST"])
def detect_ai_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        result = detect_ai_generated_content(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect/ai-text", methods=["POST"])
def detect_ai_text_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        result = detect_ai_text(data['text'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect/ai-video", methods=["POST"])
def detect_ai_video_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        result = detect_ai_video(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect/video-objects", methods=["POST"])
def detect_video_objects_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        result = detect_objects_in_video_file(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
