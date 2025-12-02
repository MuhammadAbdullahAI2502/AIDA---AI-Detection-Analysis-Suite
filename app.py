import streamlit as st
import numpy as np
from PIL import Image
import time
import torch
import json
from datetime import datetime
from io import BytesIO

from models import get_classification_model, get_detection_model, get_classes, get_coco_names, detect_ai_generated_content, detect_ai_text, detect_ai_video
from utils import preprocess_image, draw_bounding_boxes
try:
    from webcam_utils import enhance_webcam_image, apply_webcam_filters, analyze_webcam_image_quality
except ImportError:
    enhance_webcam_image = None
    apply_webcam_filters = None
    analyze_webcam_image_quality = None

# --- Constants ---
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MAX_HISTORY_ITEMS = 10

# --- Helper Functions ---
def setup_page():
    """Configure page settings and styles."""
    st.set_page_config(
        page_title="AIDA - AI Detection & Analysis Suite",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Modern Black & Blue CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            font-family: 'Orbitron', monospace;
        }
        
        .stSidebar {
            background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
            border-right: 2px solid #00d4ff;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #00d4ff, #0066ff);
            border: none;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            font-family: 'Orbitron', monospace;
            transition: all 0.4s ease;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.6);
            background: linear-gradient(45deg, #0066ff, #00d4ff);
        }
        
        .result-card {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 102, 255, 0.1));
            border-radius: 20px;
            padding: 25px;
            margin: 15px 0;
            backdrop-filter: blur(15px);
            border: 2px solid rgba(0, 212, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
            to { box-shadow: 0 8px 32px rgba(0, 212, 255, 0.4); }
        }
        
        .stSelectbox>div>div {
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid #00d4ff;
            border-radius: 10px;
        }
        
        .stSlider>div>div>div {
            background: linear-gradient(90deg, #00d4ff, #0066ff);
        }
        
        h1, h2, h3 {
            font-family: 'Orbitron', monospace;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        .stProgress .st-bo {
            background: linear-gradient(90deg, #00d4ff, #0066ff);
        }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the app header."""
    st.markdown("""
    <div class="text-center py-8">
        <h1 style="font-size: 4rem; font-weight: 900; color: #00d4ff; margin-bottom: 20px; 
                   text-shadow: 0 0 20px rgba(0, 212, 255, 0.8); 
                   animation: pulse 2s infinite;" 
           class="animate__animated animate__bounceIn">🎯 AIDA</h1>
        <p style="font-size: 1.5rem; font-weight: bold; color: #ffffff; 
                  text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);" 
           class="animate__animated animate__fadeIn animate__delay-1s">AI Detection & Analysis Suite</p>
        <p style="font-size: 1.2rem; color: #b3d9ff;" 
           class="animate__animated animate__fadeIn animate__delay-2s">Advanced AI content detection for images, text & videos</p>
    </div>
    
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

def add_to_history(mode, results):
    """Add analysis results to session history."""
    if 'history' not in st.session_state:
        st.session_state.history = []

    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "results": results
    }
    st.session_state.history.append(history_entry)

    # Limit history size
    if len(st.session_state.history) > MAX_HISTORY_ITEMS:
        st.session_state.history = st.session_state.history[-MAX_HISTORY_ITEMS:]

# --- Initialize App ---
setup_page()
display_header()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color: #00d4ff; text-shadow: 0 0 10px rgba(0, 212, 255, 0.8); font-family: Orbitron;'>🎯 AIDA Control Panel</h2>", unsafe_allow_html=True)
    app_mode = st.selectbox("Choose the App Mode", ["Image Classification", "Object Detection", "AI Image Detection", "AI Text Detection", "AI Video Detection", "Video Object Detection"])
    st.markdown("---")

    if app_mode == "Image Classification":
        st.subheader("Classification Settings")
        top_k = st.slider("Top-K Predictions", 1, 10, DEFAULT_TOP_K)

    if app_mode == "Object Detection":
        st.subheader("Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)
    
    if app_mode == "AI Image Detection":
        st.subheader("AI Image Detection")
        st.info("🔍 Analyzes images for AI-generated content using multiple detection techniques.")
    
    if app_mode == "AI Text Detection":
        st.subheader("AI Text Detection")
        st.info("📝 Analyzes text for AI-generated content using linguistic patterns.")
    
    if app_mode == "AI Video Detection":
        st.subheader("AI Video Detection")
        st.info("🎥 Analyzes video frames for AI-generated content.")
    
    if app_mode == "Video Object Detection":
        st.subheader("Video Object Detection")
        st.info("🎯 Detects objects in video frames using advanced AI models.")

    st.markdown("---")
    st.subheader("Image Source")
    image_source = st.radio("Select Image Source", ["Upload an Image", "Use Webcam", "Real-time Webcam"])
    
    if image_source == "Real-time Webcam":
        st.info("🎥 Enhanced real-time webcam with advanced processing")
        auto_analyze = st.checkbox("Auto-analyze captured images", value=True)
        webcam_quality = st.selectbox("Camera Quality", ["High (640x480)", "Medium (320x240)", "Low (160x120)"])
        image_enhancement = st.checkbox("Enable image enhancement", value=True)
        quality_monitoring = st.checkbox("Show quality metrics", value=True)

# --- Input Selection ---
image_file = None
text_input = None
video_file = None

if app_mode in ["Image Classification", "Object Detection", "AI Image Detection"]:
    if image_source == "Upload an Image":
        image_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
    elif image_source == "Use Webcam":
        webcam_image = st.camera_input("📸 Take a picture")
        if webcam_image:
            image_file = webcam_image
    elif image_source == "Real-time Webcam":
        st.markdown("### 🎥 Enhanced Webcam Interface")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            webcam_image = st.camera_input("📹 Live Camera Feed", key="realtime_cam")
        
        with col2:
            if st.button("🔄 Refresh Camera", key="refresh_cam"):
                st.rerun()
        
        with col3:
            if st.button("💾 Save Image", key="save_cam") and webcam_image:
                st.success("Image captured!")
        
        if webcam_image:
            image_file = webcam_image
            
            # Auto-analyze if enabled
            if auto_analyze and 'last_webcam_analysis' not in st.session_state:
                st.session_state.last_webcam_analysis = True
                st.info("🔄 Auto-analyzing captured image...")
            
            # Show webcam controls
            st.markdown("#### 🎛️ Webcam Controls:")
            webcam_col1, webcam_col2 = st.columns(2)
            
            with webcam_col1:
                if st.button("🎯 Analyze Current Frame", key="analyze_frame"):
                    st.session_state.force_analysis = True
            
            with webcam_col2:
                if st.button("🗑️ Clear Results", key="clear_results"):
                    if 'results' in st.session_state:
                        del st.session_state.results
                    st.success("Results cleared!")
elif app_mode == "AI Text Detection":
    text_input = st.text_area("Enter text to analyze:", height=200, placeholder="Paste or type the text you want to analyze for AI generation...")
elif app_mode in ["AI Video Detection", "Video Object Detection"]:
    st.markdown("**Note:** Streamlit has a built-in 200MB file limit. For larger videos, use the Flask API directly.")
    video_file = st.file_uploader("Upload your video (Max 200MB due to Streamlit limit)", type=["mp4", "avi", "mov", "mkv"])

# --- Enhanced Error Handling ---
if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        
        # Enhanced image processing for webcam
        if image_source in ["Use Webcam", "Real-time Webcam"]:
            # Apply advanced webcam enhancements
            if enhance_webcam_image:
                image = enhance_webcam_image(image)
            
            if apply_webcam_filters:
                image = apply_webcam_filters(image, "enhance")
            
            # Show enhanced image info
            st.sidebar.markdown("#### 📊 Image Quality:")
            st.sidebar.text(f"Size: {image.size[0]}x{image.size[1]}")
            st.sidebar.text(f"Mode: {image.mode}")
            
            # Quality analysis
            if analyze_webcam_image_quality:
                quality_info = analyze_webcam_image_quality(image)
                st.sidebar.markdown(f"**Quality:** {quality_info['quality_status']}")
                st.sidebar.progress(quality_info['quality_score'] / 100)
                
                if quality_info['recommendations']:
                    st.sidebar.markdown("**Tips:**")
                    for tip in quality_info['recommendations'][:2]:
                        st.sidebar.text(f"• {tip}")
            
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        image = None
else:
    image = None

# --- Main Content ---
if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        icon = "⚡" if "AI" not in app_mode else "🔍"
        st.markdown(f"<h3 style='font-size: 2.5rem; color: #00d4ff; text-align: center; margin-bottom: 30px; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);'>{icon} Results</h3>", unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Analyzing the image..."):
            start_time = time.time()

            try:
                if app_mode == "Image Classification":
                    status_text.text("Loading classification model...")
                    progress_bar.progress(20)
                    model = get_classification_model()
                    classes = get_classes()

                    status_text.text("Preprocessing image...")
                    progress_bar.progress(40)
                    processed_image = preprocess_image(image)

                    status_text.text("Running inference...")
                    progress_bar.progress(60)
                    with torch.no_grad():
                        outputs = model(processed_image)

                    status_text.text("Processing results...")
                    progress_bar.progress(80)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                    results = []
                    st.markdown("#### Top Predictions:")
                    for i in range(top_k):
                        class_name = classes[top_k_indices[i]]
                        probability = top_k_probs[i].item()
                        results.append({"rank": i+1, "class": class_name, "probability": probability})
                        st.markdown(f"""
                        <div class="result-card">
                            <p style="font-size: 1.2rem; color: #00d4ff; font-weight: bold; margin: 0;">
                                #{i+1} {class_name}
                            </p>
                            <p style="font-size: 1rem; color: #ffffff; margin: 5px 0 0 0;">
                                Confidence: {probability:.2%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.session_state.results = results
                    add_to_history(app_mode, results)

                elif app_mode == "Object Detection":
                    status_text.text("Loading detection model...")
                    progress_bar.progress(20)
                    model = get_detection_model()
                    coco_names = get_coco_names()

                    status_text.text("Preprocessing image...")
                    progress_bar.progress(40)
                    img_tensor = preprocess_image(image, detection=True)

                    status_text.text("Running inference...")
                    progress_bar.progress(60)
                    with torch.no_grad():
                        predictions = model(img_tensor)

                    status_text.text("Drawing bounding boxes...")
                    progress_bar.progress(80)
                    img_with_boxes = draw_bounding_boxes(image, predictions, coco_names, confidence_threshold)

                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    st.image(img_with_boxes, caption="Image with Detected Objects", use_container_width=True)

                    detection_results = []
                    for i in range(len(predictions[0]["boxes"])):
                        confidence = predictions[0]["scores"][i].item()
                        if confidence > confidence_threshold:
                            box = predictions[0]["boxes"][i].detach().cpu().numpy()
                            label_index = predictions[0]["labels"][i].item()
                            label = coco_names[label_index]
                            detection_results.append({
                                "label": label,
                                "confidence": confidence,
                                "bbox": box.tolist()
                            })
                    st.session_state.results = detection_results
                    st.session_state.processed_image = img_with_boxes
                    add_to_history(app_mode, detection_results)

                elif app_mode == "AI Image Detection":
                    status_text.text("Analyzing image for AI generation...")
                    progress_bar.progress(30)
                    
                    ai_result = detect_ai_generated_content(image)
                    
                    progress_bar.progress(100)
                    status_text.text("AI analysis complete!")
                    
                    if "error" not in ai_result:
                        result_color = "#ff4444" if "AI-Generated" in ai_result["result"] else "#44ff44"
                        st.markdown(f"""
                        <div class="result-card">
                            <h4 style="color: {result_color}; font-size: 1.5rem; margin: 0;">
                                🔍 {ai_result["result"]}
                            </h4>
                            <p style="color: #ffffff; font-size: 1.2rem; margin: 10px 0;">
                                Confidence: {ai_result["confidence"]:.1%}
                            </p>
                            <p style="color: #00d4ff; font-size: 1rem;">
                                AI Probability: {ai_result["ai_probability"]:.1%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📊 Detailed Analysis:")
                        details = ai_result["details"]
                        
                        for metric, score in details.items():
                            metric_name = metric.replace("_", " ").title()
                            color = "#ff6b6b" if score > 0.6 else "#ffd93d" if score > 0.3 else "#6bcf7f"
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <p style="color: #00d4ff; font-size: 1.1rem; margin: 0;">
                                    {metric_name}
                                </p>
                                <p style="color: {color}; font-size: 1rem; margin: 5px 0 0 0;">
                                    Score: {score:.2f}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.session_state.results = [ai_result]
                        add_to_history(app_mode, [ai_result])
                    else:
                        st.error(f"Analysis failed: {ai_result['error']}")

                end_time = time.time()
                inference_time = end_time - start_time
                st.markdown(f"""
                <div class="result-card" style="text-align: center;">
                    <p style="color: #00d4ff; font-size: 1.1rem; font-weight: bold; margin: 0;">
                        ⚡ Processing Time: {inference_time:.3f}s
                    </p>
                </div>
                """, unsafe_allow_html=True)

                progress_bar.empty()
                status_text.empty()
                
                # Enhanced webcam features
                if image_source in ["Use Webcam", "Real-time Webcam"]:
                    st.markdown("#### 🎥 Enhanced Webcam Features:")
                    
                    webcam_col1, webcam_col2, webcam_col3 = st.columns(3)
                    
                    with webcam_col1:
                        if st.button("📸 Capture New", key="take_another"):
                            st.rerun()
                    
                    with webcam_col2:
                        if st.button("💾 Export Results", key="download_results"):
                            if 'results' in st.session_state:
                                import json
                                from datetime import datetime
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                results_json = json.dumps({
                                    "timestamp": timestamp,
                                    "mode": app_mode,
                                    "source": "webcam",
                                    "results": st.session_state.results
                                }, indent=2)
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=results_json,
                                    file_name=f"aida_webcam_{timestamp}.json",
                                    mime="application/json"
                                )
                    
                    with webcam_col3:
                        if st.button("🔄 Live Mode", key="continuous_mode"):
                            st.balloons()
                            st.success("Live mode activated! Refresh to capture continuously.")
                    
                    # Additional webcam controls
                    st.markdown("#### 🎛️ Advanced Controls:")
                    
                    control_col1, control_col2 = st.columns(2)
                    
                    with control_col1:
                        if st.button("🎨 Apply Filters", key="apply_filters"):
                            if apply_webcam_filters and image:
                                filtered_image = apply_webcam_filters(image, "auto_adjust")
                                st.image(filtered_image, caption="Filtered Image", width=200)
                    
                    with control_col2:
                        if st.button("📈 Quality Report", key="quality_report"):
                            if analyze_webcam_image_quality and image:
                                quality = analyze_webcam_image_quality(image)
                                st.json(quality)

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during analysis: {str(e)}")
                
                # Enhanced webcam error recovery
                if image_source in ["Use Webcam", "Real-time Webcam"]:
                    st.markdown("#### 🔧 Webcam Troubleshooting:")
                    
                    error_col1, error_col2 = st.columns(2)
                    
                    with error_col1:
                        st.info("💡 Camera Tips:")
                        st.text("• Check browser permissions")
                        st.text("• Ensure good lighting")
                        st.text("• Hold camera steady")
                    
                    with error_col2:
                        if st.button("🔄 Reset Camera", key="reset_webcam"):
                            st.rerun()
                        
                        if st.button("📹 Test Camera", key="test_camera"):
                            st.success("Camera test initiated! Check if camera light turns on.")

# --- Text Analysis ---
elif text_input and text_input.strip():
    st.markdown(f"<h3 style='font-size: 2.5rem; color: #00d4ff; text-align: center; margin-bottom: 30px; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);'>📝 Text Analysis</h3>", unsafe_allow_html=True)
    
    with st.spinner("Analyzing text for AI generation..."):
        try:
            text_result = detect_ai_text(text_input)
            
            if "error" not in text_result:
                result_color = "#ff4444" if "AI-Generated" in text_result["result"] else "#44ff44"
                st.markdown(f"""
                <div class="result-card">
                    <h4 style="color: {result_color}; font-size: 1.5rem; margin: 0;">
                        📝 {text_result["result"]}
                    </h4>
                    <p style="color: #ffffff; font-size: 1.2rem; margin: 10px 0;">
                        Confidence: {text_result["confidence"]:.1%}
                    </p>
                    <p style="color: #00d4ff; font-size: 1rem;">
                        AI Probability: {text_result["ai_probability"]:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### 📊 Text Analysis Details:")
                details = text_result["details"]
                
                for metric, score in details.items():
                    metric_name = metric.replace("_", " ").title()
                    color = "#ff6b6b" if score > 0.6 else "#ffd93d" if score > 0.3 else "#6bcf7f"
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <p style="color: #00d4ff; font-size: 1.1rem; margin: 0;">
                            {metric_name}
                        </p>
                        <p style="color: {color}; font-size: 1rem; margin: 5px 0 0 0;">
                            Score: {score:.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"Analysis failed: {text_result['error']}")
        except Exception as e:
            st.error(f"Error during text analysis: {str(e)}")

# --- Video Analysis ---
elif video_file is not None:
    if app_mode == "AI Video Detection":
        st.markdown(f"<h3 style='font-size: 2.5rem; color: #00d4ff; text-align: center; margin-bottom: 30px; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);'>🎥 AI Video Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.video(video_file)
        
        with col2:
            with st.spinner("Analyzing video for AI generation..."):
                try:
                    video_result = detect_ai_video(video_file)
                    
                    if "error" not in video_result:
                        result_color = "#ff4444" if "AI-Generated" in video_result["result"] else "#44ff44"
                        st.markdown(f"""
                        <div class="result-card">
                            <h4 style="color: {result_color}; font-size: 1.5rem; margin: 0;">
                                🎥 {video_result["result"]}
                            </h4>
                            <p style="color: #ffffff; font-size: 1.2rem; margin: 10px 0;">
                                Confidence: {video_result["confidence"]:.1%}
                            </p>
                            <p style="color: #00d4ff; font-size: 1rem;">
                                AI Probability: {video_result["ai_probability"]:.1%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📊 Video Analysis Details:")
                        details = video_result["details"]
                        
                        for metric, score in details.items():
                            if metric != "detected_objects":
                                metric_name = metric.replace("_", " ").title()
                                color = "#ff6b6b" if score > 0.6 else "#ffd93d" if score > 0.3 else "#6bcf7f"
                                
                                st.markdown(f"""
                                <div class="result-card">
                                    <p style="color: #00d4ff; font-size: 1.1rem; margin: 0;">
                                        {metric_name}
                                    </p>
                                    <p style="color: {color}; font-size: 1rem; margin: 5px 0 0 0;">
                                        Score: {score:.2f}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show detected objects if available
                        if "detected_objects" in details:
                            objects = details["detected_objects"]
                            st.markdown("#### 🎯 Detected Objects:")
                            st.markdown(f"""
                            <div class="result-card">
                                <p style="color: #00d4ff; font-size: 1.1rem; margin: 0;">
                                    Total Objects: {objects.get('total_objects', 0)}
                                </p>
                                <p style="color: #ffffff; font-size: 1rem; margin: 5px 0 0 0;">
                                    Unique Types: {objects.get('unique_objects', 0)}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if objects.get('object_counts'):
                                for obj, count in list(objects['object_counts'].items())[:5]:
                                    st.markdown(f"""
                                    <div class="result-card">
                                        <p style="color: #00d4ff; font-size: 1rem; margin: 0;">
                                            {obj.title()}: {count}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.error(f"Analysis failed: {video_result['error']}")
                except Exception as e:
                    st.error(f"Error during video analysis: {str(e)}")
    
    elif app_mode == "Video Object Detection":
        st.markdown(f"<h3 style='font-size: 2.5rem; color: #00d4ff; text-align: center; margin-bottom: 30px; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);'>🎯 Video Object Detection</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.video(video_file)
        
        with col2:
            with st.spinner("Detecting objects in video..."):
                try:
                    from models import detect_objects_in_video_file
                    objects_result = detect_objects_in_video_file(video_file)
                    
                    if "error" not in objects_result:
                        st.markdown(f"""
                        <div class="result-card">
                            <h4 style="color: #00d4ff; font-size: 1.5rem; margin: 0;">
                                🎯 Objects Detected
                            </h4>
                            <p style="color: #ffffff; font-size: 1.2rem; margin: 10px 0;">
                                Total Objects: {objects_result['total_objects']}
                            </p>
                            <p style="color: #00d4ff; font-size: 1rem;">
                                Unique Types: {objects_result['unique_objects']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if objects_result.get('object_counts'):
                            st.markdown("#### 📊 Object Counts:")
                            for obj, count in objects_result['object_counts'].items():
                                confidence_color = "#44ff44" if count > 5 else "#ffd93d" if count > 2 else "#ff6b6b"
                                st.markdown(f"""
                                <div class="result-card">
                                    <p style="color: #00d4ff; font-size: 1.1rem; margin: 0;">
                                        {obj.title()}
                                    </p>
                                    <p style="color: {confidence_color}; font-size: 1rem; margin: 5px 0 0 0;">
                                        Count: {count}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if objects_result.get('detections'):
                            st.markdown("#### 🔍 Detection Details:")
                            for i, detection in enumerate(objects_result['detections'][:10]):
                                st.markdown(f"""
                                <div class="result-card">
                                    <p style="color: #00d4ff; font-size: 1rem; margin: 0;">
                                        Frame {detection['frame']}: {detection['label'].title()}
                                    </p>
                                    <p style="color: #ffffff; font-size: 0.9rem; margin: 5px 0 0 0;">
                                        Confidence: {detection['confidence']:.1%}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        if objects_result.get('video_info'):
                            info = objects_result['video_info']
                            st.markdown(f"""
                            <div class="result-card">
                                <p style="color: #00d4ff; font-size: 1rem; margin: 0;">
                                    📹 Video Info
                                </p>
                                <p style="color: #ffffff; font-size: 0.9rem; margin: 5px 0 0 0;">
                                    Total Frames: {info['total_frames']}<br>
                                    Analyzed: {info['frames_analyzed']}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"Object detection failed: {objects_result['error']}")
                except Exception as e:
                    st.error(f"Error during object detection: {str(e)}")

else:
    if app_mode in ["Image Classification", "Object Detection", "AI Image Detection"]:
        st.info("Please upload an image or select the webcam option to proceed.")
    elif app_mode == "AI Text Detection":
        st.info("Please enter text in the text area above to analyze.")
    elif app_mode in ["AI Video Detection", "Video Object Detection"]:
        st.info("Please upload a video file to analyze.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 15px; margin-top: 20px; position: relative;">
        <p style="color: #00d4ff; font-size: 0.9rem; font-weight: bold; font-family: Orbitron;">
            🎯 AIDA - AI Detection & Analysis Suite | 🚀 Developed By Muhammad Abdullah
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)