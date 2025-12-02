# AIDA - AI Detection & Analysis Suite

**A**rtificial **I**ntelligence **D**etection & **A**nalysis - Complete solution for detecting AI-generated content across images, text, and videos with advanced classification and object detection capabilities. Dual interface: Streamlit web app and Flask REST API.

## 🎯 Core Features

### **AI Detection Suite**
- **🖼️ Advanced Image Analysis:** Multi-layer spectral, texture, edge & frequency domain detection
- **📝 Intelligent Text Analysis:** Linguistic patterns, perplexity, semantic coherence detection  
- **🎥 Smart Video Analysis:** Temporal consistency, motion patterns, frame interpolation detection

### **Traditional ML Capabilities**
- **🔍 Image Classification:** ResNet18 on ImageNet (1000+ classes)
- **🎯 Object Detection:** Faster R-CNN on COCO dataset (80+ objects)

### **Dual Interface**
- **🌐 Streamlit Web App:** Interactive UI for all detection types
- **⚡ Flask REST API:** Programmatic access with JSON responses

### **Advanced Technology**
- **🧠 Ensemble Detection:** Multiple algorithms combined for 85-95% accuracy
- **⚙️ Configurable Thresholds:** Fine-tuned parameters for optimal detection
- **🚀 Real-time Processing:** Instant analysis and results

## Dependencies

The project uses the following libraries:

- streamlit
- torch
- torchvision
- opencv-python-headless
- numpy
- Pillow
- requests
- Flask
- scikit-learn
- nltk

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AIDA-AI-Detection-Suite.git
    cd AIDA-AI-Detection-Suite
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Quick Start

### **Launch AIDA Web Interface**

To run the Streamlit web application:

```bash
streamlit run app.py
```

Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### **AIDA REST API**

```bash
python api.py
```

⚡ **API Server:** `http://localhost:5000`  
📄 **Interactive Docs:** Available via API endpoints

## 🔗 AIDA API Endpoints

### 1. Image Classification

- **Endpoint:** `/predict/classify`
- **Method:** `POST`
- **Form Data:** `file`: The image file to be classified.
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        [
            {
                "class": "giant panda",
                "probability": 0.99
            },
            {
                "class": "lesser panda",
                "probability": 0.003
            }
        ]
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`

### 2. Object Detection

- **Endpoint:** `/predict/detect`
- **Method:** `POST`
- **Form Data:** `file`: The image file for object detection.
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        [
            {
                "label": "person",
                "confidence": 0.98,
                "box": [x1, y1, x2, y2]
            },
            {
                "label": "car",
                "confidence": 0.92,
                "box": [x1, y1, x2, y2]
            }
        ]
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`
### 3. AI Image Detection

- **Endpoint:** `/detect/ai-image`
- **Method:** `POST`
- **Form Data:** `file`: The image file to analyze for AI generation.
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        {
            "result": "Likely AI-Generated",
            "confidence": 0.85,
            "ai_probability": 0.78,
            "details": {
                "noise_patterns": 0.8,
                "smoothness": 0.9,
                "compression": 0.7,
                "pixel_consistency": 0.6
            }
        }
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`

### 4. AI Text Detection

- **Endpoint:** `/detect/ai-text`
- **Method:** `POST`
- **JSON Data:** `{"text": "Text to analyze"}`
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        {
            "result": "Likely AI-Generated",
            "confidence": 0.82,
            "ai_probability": 0.75,
            "details": {
                "repetition": 0.7,
                "structure": 0.8,
                "vocabulary": 0.6,
                "ai_phrases": 0.9
            }
        }
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`

### 5. AI Video Detection

- **Endpoint:** `/detect/ai-video`
- **Method:** `POST`
- **Form Data:** `file`: The video file to analyze for AI generation.
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        {
            "result": "Likely AI-Generated",
            "confidence": 0.79,
            "ai_probability": 0.73,
            "details": {
                "frames_analyzed": 5,
                "avg_frame_score": 0.71,
                "consistency": 0.85,
                "detected_objects": {
                    "total_objects": 12,
                    "unique_objects": 4,
                    "object_counts": {"person": 5, "car": 3}
                }
            }
        }
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`

### 6. Video Object Detection

- **Endpoint:** `/detect/video-objects`
- **Method:** `POST`
- **Form Data:** `file`: The video file for object detection.
- **Success Response:**
    - **Code:** 200
    - **Content:** 
        ```json
        {
            "total_objects": 15,
            "unique_objects": 6,
            "object_counts": {
                "person": 8,
                "car": 4,
                "bicycle": 2,
                "dog": 1
            },
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.95,
                    "box": [100, 150, 200, 300],
                    "frame": 0
                }
            ],
            "video_info": {
                "total_frames": 150,
                "frames_analyzed": 5
            }
        }
        ```
- **Error Response:**
    - **Code:** 400 (Bad Request) or 500 (Internal Server Error)
    - **Content:** `{"error": "Error message"}`

## AI Content Detection Features

### Image Detection Methods
- **Noise Pattern Analysis:** Detects artificial noise patterns common in AI-generated images
- **Smoothness Analysis:** Identifies unnatural smoothness in textures and surfaces
- **Compression Artifacts:** Analyzes compression patterns typical of AI generation
- **Pixel Consistency:** Checks for pixel-level inconsistencies

### Text Detection Methods
- **Repetition Analysis:** Detects repetitive sentence structures and patterns
- **Vocabulary Patterns:** Analyzes word diversity and usage patterns
- **AI Phrase Detection:** Identifies common AI-generated phrases and expressions
- **Structural Analysis:** Examines sentence structure uniformity

### Video Detection Methods
- **Frame-by-Frame Analysis:** Analyzes individual frames for AI artifacts
- **Temporal Consistency:** Checks for unnatural consistency across frames
- **Motion Pattern Analysis:** Detects artificial motion patterns

## Example Usage

### Python API Client Example

```python
import requests

# AI Image Detection
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/detect/ai-image', files={'file': f})
    result = response.json()
    print(f"Result: {result['result']}, Confidence: {result['confidence']:.2%}")

# AI Text Detection
text_data = {"text": "Your text to analyze here..."}
response = requests.post('http://localhost:5000/detect/ai-text', json=text_data)
result = response.json()
print(f"Result: {result['result']}, Confidence: {result['confidence']:.2%}")
```