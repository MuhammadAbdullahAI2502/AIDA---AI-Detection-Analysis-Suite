import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import resnet18, ResNet18_Weights
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageStat
import cv2
import re
from collections import Counter
import tempfile
import os
from scipy import ndimage
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import warnings
warnings.filterwarnings('ignore')

# Import advanced detection utilities and configuration
try:
    from advanced_detection import enhanced_ai_detection_pipeline
except ImportError:
    enhanced_ai_detection_pipeline = None

try:
    from detection_config import get_detection_result, get_config, ENHANCED_DETECTION_WEIGHTS
except ImportError:
    get_detection_result = None
    get_config = None
    ENHANCED_DETECTION_WEIGHTS = {'base_model_weight': 0.7, 'enhanced_model_weight': 0.3}

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

@st.cache_resource
def get_classification_model():
    """Loads and caches the ResNet18 model."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    # Enable GPU if available for better performance
    if torch.cuda.is_available():
        model.to('cuda')
    return model

@st.cache_resource
def get_detection_model():
    """Loads and caches the Faster R-CNN model."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    # Enable GPU if available for better performance
    if torch.cuda.is_available():
        model.to('cuda')
    return model

@st.cache_resource
def get_classes():
    """Loads and caches the ImageNet class names."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    classes = [line.strip() for line in response.text.split("\n")]
    return classes

@st.cache_resource
def get_coco_names():
    """Loads and caches the COCO class names."""
    return [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

@st.cache_resource
def get_ai_detection_model():
    """Simple AI-generated content detector using image analysis."""
    # Using ResNet18 for feature extraction
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    return model

def detect_ai_generated_content(image):
    """Advanced AI-generated image detection using multiple sophisticated techniques."""
    try:
        img_array = np.array(image)
        
        # Advanced detection methods
        spectral_score = analyze_spectral_features(img_array)
        texture_score = analyze_texture_patterns(img_array)
        edge_score = analyze_edge_artifacts(img_array)
        color_score = analyze_color_distribution(img_array)
        frequency_score = analyze_frequency_domain(img_array)
        metadata_score = analyze_image_metadata(image)
        
        # Enhanced detection with advanced pipeline
        enhanced_result = None
        if enhanced_ai_detection_pipeline:
            try:
                enhanced_result = enhanced_ai_detection_pipeline(img_array, 'image')
            except:
                pass
        
        # Weighted combination with refined weights
        base_probability = (
            spectral_score * 0.25 + 
            texture_score * 0.20 + 
            edge_score * 0.20 + 
            color_score * 0.15 + 
            frequency_score * 0.15 + 
            metadata_score * 0.05
        )
        
        # Integrate enhanced detection if available
        if enhanced_result and 'enhanced_ai_probability' in enhanced_result:
            ai_probability = (base_probability * 0.7 + enhanced_result['enhanced_ai_probability'] * 0.3)
        else:
            ai_probability = base_probability
        
        # Use configuration-based thresholds if available
        if get_detection_result:
            result, confidence = get_detection_result(ai_probability, 'image')
        else:
            # Fallback to enhanced threshold logic
            if ai_probability > 0.75:
                result = "AI-Generated (High Confidence)"
                confidence = min(0.95, ai_probability + 0.1)
            elif ai_probability > 0.55:
                result = "Likely AI-Generated"
                confidence = ai_probability
            elif ai_probability > 0.35:
                result = "Possibly AI-Generated"
                confidence = ai_probability * 0.8
            else:
                result = "Likely Real"
                confidence = 1 - ai_probability
            
        return {
            "result": result,
            "confidence": confidence,
            "ai_probability": ai_probability,
            "details": {
                "spectral_analysis": spectral_score,
                "texture_patterns": texture_score,
                "edge_artifacts": edge_score,
                "color_distribution": color_score,
                "frequency_domain": frequency_score,
                "metadata_analysis": metadata_score
            }
        }
    except Exception as e:
        return {"result": "Error", "confidence": 0, "error": str(e)}

def analyze_spectral_features(img_array):
    """Analyze spectral characteristics unique to AI-generated images."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # FFT analysis
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Analyze high-frequency components
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Create masks for different frequency regions
    y, x = np.ogrid[:h, :w]
    mask_low = (x - center_w)**2 + (y - center_h)**2 <= (min(h, w) * 0.1)**2
    mask_high = (x - center_w)**2 + (y - center_h)**2 >= (min(h, w) * 0.3)**2
    
    low_freq_energy = np.mean(magnitude_spectrum[mask_low])
    high_freq_energy = np.mean(magnitude_spectrum[mask_high])
    
    # AI images typically have unusual frequency distributions
    freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
    
    if freq_ratio < 0.15:
        return 0.85  # Very low high-freq content, likely AI
    elif freq_ratio < 0.25:
        return 0.65
    elif freq_ratio > 0.6:
        return 0.2   # Natural frequency distribution
    else:
        return 0.4

def analyze_texture_patterns(img_array):
    """Advanced texture analysis using Local Binary Patterns and Gabor filters."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Local Binary Pattern analysis
    def local_binary_pattern(image, radius=3, n_points=24):
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                binary_string = ''
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                        binary_string += '1' if image[x, y] >= center else '0'
                lbp[i, j] = int(binary_string, 2) if binary_string else 0
        return lbp
    
    lbp = local_binary_pattern(gray)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-8)
    
    # Calculate texture uniformity (AI images often have more uniform textures)
    texture_entropy = entropy(lbp_hist + 1e-8)
    
    # Gabor filter responses
    gabor_responses = []
    for theta in [0, 45, 90, 135]:
        kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        gabor_responses.append(np.var(filtered))
    
    gabor_variance = np.var(gabor_responses)
    
    # AI images typically have lower texture entropy and gabor variance
    if texture_entropy < 4.5 and gabor_variance < 1000:
        return 0.8
    elif texture_entropy < 5.5 or gabor_variance < 2000:
        return 0.6
    else:
        return 0.25

def analyze_edge_artifacts(img_array):
    """Detect edge artifacts common in AI-generated images."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale edge detection
    edges_canny = cv2.Canny(gray, 50, 150)
    edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    
    # Analyze edge continuity and sharpness
    edge_density = np.sum(edges_canny > 0) / edges_canny.size
    edge_sharpness = np.std(edges_sobel)
    
    # Check for unnatural edge patterns
    # AI images often have either too sharp or too smooth edges
    if edge_sharpness > 80 or edge_sharpness < 15:
        return 0.75
    elif edge_density < 0.05 or edge_density > 0.3:
        return 0.65
    else:
        return 0.3

def analyze_color_distribution(img_array):
    """Analyze color distribution patterns typical of AI generation."""
    # Convert to different color spaces
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Analyze color histogram entropy
    color_entropies = []
    for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-8)
        color_entropies.append(entropy(hist + 1e-8))
    
    avg_color_entropy = np.mean(color_entropies)
    
    # Analyze saturation distribution
    sat_hist, _ = np.histogram(hsv[:, :, 1], bins=256, range=(0, 256))
    sat_hist = sat_hist.astype(float)
    sat_hist /= (sat_hist.sum() + 1e-8)
    sat_entropy = entropy(sat_hist + 1e-8)
    
    # AI images often have unnatural color distributions
    if avg_color_entropy < 6.5 or sat_entropy < 5.0:
        return 0.7
    elif avg_color_entropy < 7.0 or sat_entropy < 6.0:
        return 0.5
    else:
        return 0.25

def analyze_frequency_domain(img_array):
    """Advanced frequency domain analysis for AI detection."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # DCT analysis
    dct = cv2.dct(np.float32(gray))
    
    # Analyze DCT coefficient distribution
    dct_flat = dct.flatten()
    dct_std = np.std(dct_flat)
    dct_mean = np.mean(np.abs(dct_flat))
    
    # Wavelet analysis using simple approximation
    def simple_wavelet_transform(image):
        # Simple Haar-like wavelet transform
        h, w = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        
        # Horizontal differences
        result[:, :-1] = image[:, 1:] - image[:, :-1]
        
        return result
    
    wavelet = simple_wavelet_transform(gray.astype(np.float32))
    wavelet_energy = np.sum(wavelet**2)
    
    # AI images often have specific frequency characteristics
    if dct_std < 50 or wavelet_energy < 1e6:
        return 0.75
    elif dct_std < 100 or wavelet_energy < 5e6:
        return 0.5
    else:
        return 0.2

def analyze_image_metadata(image):
    """Analyze image metadata for AI generation indicators."""
    try:
        # Check image properties
        width, height = image.size
        
        # AI generators often produce images with specific dimensions
        common_ai_sizes = [(512, 512), (1024, 1024), (768, 768), (256, 256), (1024, 768), (768, 1024)]
        
        if (width, height) in common_ai_sizes or (height, width) in common_ai_sizes:
            return 0.6
        
        # Check if dimensions are multiples of common AI block sizes
        if width % 64 == 0 and height % 64 == 0:
            return 0.4
        
        return 0.1
    except:
        return 0.1

def detect_ai_text(text):
    """Advanced AI text detection using multiple linguistic analysis techniques."""
    try:
        if len(text.strip()) < 50:
            return {"result": "Text too short for analysis", "confidence": 0, "ai_probability": 0.5}
        
        # Advanced analysis methods
        perplexity_score = analyze_text_perplexity(text)
        semantic_score = analyze_semantic_coherence(text)
        syntactic_score = analyze_syntactic_patterns(text)
        stylistic_score = analyze_stylistic_features(text)
        ngram_score = analyze_ngram_patterns(text)
        pos_score = analyze_pos_patterns(text)
        
        # Enhanced text detection
        enhanced_result = None
        if enhanced_ai_detection_pipeline:
            try:
                enhanced_result = enhanced_ai_detection_pipeline(text, 'text')
            except:
                pass
        
        # Adjusted weights to reduce false positives
        base_probability = (
            perplexity_score * 0.15 +
            semantic_score * 0.15 +
            syntactic_score * 0.25 +
            stylistic_score * 0.30 +
            ngram_score * 0.10 +
            pos_score * 0.05
        )
        
        # Integrate enhanced detection if available
        if enhanced_result and 'enhanced_ai_probability' in enhanced_result:
            ai_probability = (base_probability * 0.7 + enhanced_result['enhanced_ai_probability'] * 0.3)
        else:
            ai_probability = base_probability
        
        # Apply human text calibration
        try:
            from text_calibration import calibrate_text_detection
            ai_probability = calibrate_text_detection(text, ai_probability)
        except:
            pass
        
        # Use configuration-based thresholds if available
        if get_detection_result:
            result, confidence = get_detection_result(ai_probability, 'text')
            # Adjust result text for text content
            if "Likely Real" in result:
                result = result.replace("Likely Real", "Likely Human-Written")
        else:
            # Fallback logic
            if ai_probability > 0.75:
                result = "AI-Generated (High Confidence)"
                confidence = min(0.95, ai_probability + 0.1)
            elif ai_probability > 0.55:
                result = "Likely AI-Generated"
                confidence = ai_probability
            elif ai_probability > 0.35:
                result = "Possibly AI-Generated"
                confidence = ai_probability * 0.8
            else:
                result = "Likely Human-Written"
                confidence = 1 - ai_probability
            
        return {
            "result": result,
            "confidence": confidence,
            "ai_probability": ai_probability,
            "details": {
                "perplexity_analysis": perplexity_score,
                "semantic_coherence": semantic_score,
                "syntactic_patterns": syntactic_score,
                "stylistic_features": stylistic_score,
                "ngram_patterns": ngram_score,
                "pos_patterns": pos_score
            }
        }
    except Exception as e:
        return {"result": "Error", "confidence": 0, "error": str(e)}

def analyze_text_perplexity(text):
    """Analyze text perplexity patterns typical of AI generation."""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.5
        
        # Calculate sentence length variance (AI tends to be more uniform)
        lengths = [len(word_tokenize(s)) for s in sentences]
        if len(lengths) < 2:
            return 0.5
            
        length_cv = np.std(lengths) / (np.mean(lengths) + 1e-8)  # Coefficient of variation
        
        # AI text typically has lower coefficient of variation
        if length_cv < 0.2:
            return 0.7
        elif length_cv < 0.4:
            return 0.4
        else:
            return 0.1
    except:
        return 0.5

def analyze_semantic_coherence(text):
    """Analyze semantic coherence using TF-IDF similarity."""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 0.5
        
        # Create TF-IDF vectors for sentences
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
                similarities.append(sim)
            
            if not similarities:
                return 0.5
                
            avg_similarity = np.mean(similarities)
            
            # AI text often has unnaturally high semantic coherence
            if avg_similarity > 0.8:
                return 0.7
            elif avg_similarity > 0.6:
                return 0.5
            elif avg_similarity < 0.15:
                return 0.6  # Too low coherence also suspicious
            else:
                return 0.2
        except:
            return 0.5
    except:
        return 0.5

def analyze_syntactic_patterns(text):
    """Analyze syntactic patterns and complexity."""
    try:
        sentences = sent_tokenize(text)
        
        # Analyze sentence complexity
        complexities = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            if len(words) > 0:
                # Simple complexity measure: ratio of function words to content words
                stop_words = set(stopwords.words('english'))
                function_words = sum(1 for word in words if word.lower() in stop_words)
                complexity = function_words / len(words)
                complexities.append(complexity)
        
        if not complexities:
            return 0.5
            
        avg_complexity = np.mean(complexities)
        complexity_variance = np.var(complexities)
        
        # AI text often has more uniform syntactic complexity
        if complexity_variance < 0.005 and 0.35 < avg_complexity < 0.55:
            return 0.6
        elif complexity_variance < 0.015:
            return 0.4
        else:
            return 0.15
    except:
        return 0.5

def analyze_stylistic_features(text):
    """Analyze stylistic features typical of AI writing."""
    try:
        # Check for AI-typical phrases and patterns
        ai_indicators = [
            r'\bit\'s important to note\b', r'\bit\'s worth noting\b', r'\bkeep in mind\b',
            r'\bin conclusion\b', r'\bto summarize\b', r'\boverall\b', r'\bin summary\b',
            r'\bfurthermore\b', r'\bmoreover\b', r'\badditionally\b', r'\bhowever\b',
            r'\bnevertheless\b', r'\bon the other hand\b'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for pattern in ai_indicators if re.search(pattern, text_lower))
        
        # Check for repetitive transition words
        transition_density = indicator_count / max(1, len(sent_tokenize(text)))
        
        # Check for overly formal language patterns
        formal_patterns = [r'\butilize\b', r'\bfacilitate\b', r'\bdemonstrate\b', r'\bimplement\b']
        formal_count = sum(1 for pattern in formal_patterns if re.search(pattern, text_lower))
        
        if transition_density > 0.4 or formal_count > 4:
            return 0.8
        elif transition_density > 0.25 or formal_count > 2:
            return 0.5
        else:
            return 0.1
    except:
        return 0.5

def analyze_ngram_patterns(text):
    """Analyze n-gram patterns for AI detection."""
    try:
        words = word_tokenize(text.lower())
        if len(words) < 10:
            return 0.5
        
        # Create bigrams and trigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        
        # Calculate repetition rates
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        bigram_repetition = sum(1 for count in bigram_counts.values() if count > 1) / len(bigram_counts)
        trigram_repetition = sum(1 for count in trigram_counts.values() if count > 1) / len(trigram_counts) if trigram_counts else 0
        
        # AI text often has higher n-gram repetition
        if bigram_repetition > 0.4 or trigram_repetition > 0.25:
            return 0.7
        elif bigram_repetition > 0.25 or trigram_repetition > 0.15:
            return 0.4
        else:
            return 0.1
    except:
        return 0.5

def analyze_pos_patterns(text):
    """Analyze part-of-speech patterns."""
    try:
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        # Extract POS tag sequence
        pos_sequence = [tag for word, tag in pos_tags]
        
        # Calculate POS diversity
        pos_counts = Counter(pos_sequence)
        pos_diversity = len(pos_counts) / len(pos_sequence) if pos_sequence else 0
        
        # AI text often has less diverse POS patterns
        if pos_diversity < 0.3:
            return 0.7
        elif pos_diversity < 0.4:
            return 0.5
        else:
            return 0.2
    except:
        return 0.5

def detect_objects_in_video_file(video_file):
    """Standalone video object detection function."""
    try:
        # Check file size
        video_file.seek(0, 2)
        file_size = video_file.tell()
        video_file.seek(0)
        
        if file_size > 200 * 1024 * 1024:
            return {"error": f"File size {file_size/(1024*1024):.1f}MB exceeds 200MB limit"}
        
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            os.unlink(temp_path)
            return {"error": "Could not open video"}
        
        # Enhanced frame extraction for better accuracy
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Smart frame sampling - extract frames from different time intervals
        if total_frames > 50:
            # For longer videos, sample strategically
            frame_indices = [
                int(total_frames * 0.1),   # 10% into video
                int(total_frames * 0.25),  # 25% into video
                int(total_frames * 0.4),   # 40% into video
                int(total_frames * 0.6),   # 60% into video
                int(total_frames * 0.75),  # 75% into video
                int(total_frames * 0.9),   # 90% into video
                total_frames - 1           # Last frame
            ]
        else:
            # For shorter videos, sample evenly
            frame_interval = max(1, total_frames // 7)
            frame_indices = list(range(0, total_frames, frame_interval))[:7]
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Higher resolution for better detection
            frame_resized = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_LANCZOS4)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if len(frames) >= 7:
                break
        
        cap.release()
        os.unlink(temp_path)
        
        if not frames:
            return {"error": "No frames extracted"}
        
        # Detect objects
        result = detect_objects_in_video(frames)
        result["video_info"] = {
            "total_frames": total_frames,
            "frames_analyzed": len(frames)
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

def detect_ai_video(video_file):
    """Advanced AI-generated video detection with temporal analysis."""
    try:
        # Check file size (limit to 200MB for Streamlit)
        video_file.seek(0, 2)
        file_size = video_file.tell()
        video_file.seek(0)
        
        if file_size > 200 * 1024 * 1024:
            return {"result": "Error", "confidence": 0, "error": f"File size {file_size/(1024*1024):.1f}MB exceeds 200MB limit. Use Flask API for larger files."}
        
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            os.unlink(temp_path)
            return {"result": "Error", "confidence": 0, "error": "Could not open video"}
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Analyze more frames for better accuracy (up to 10 frames)
        max_frames = min(10, total_frames)
        frame_interval = max(1, total_frames // max_frames)
        
        frames = []
        frame_scores = []
        
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame for processing
            frame_resized = cv2.resize(frame, (256, 256))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Analyze individual frame
            pil_frame = Image.fromarray(frame_rgb)
            frame_result = detect_ai_generated_content(pil_frame)
            if "error" not in frame_result:
                frame_scores.append(frame_result["ai_probability"])
            
            if len(frames) >= max_frames:
                break
        
        cap.release()
        os.unlink(temp_path)
        
        if not frame_scores or len(frames) < 2:
            return {"result": "Error", "confidence": 0, "error": "Insufficient frames for analysis"}
        
        # Advanced temporal analysis
        temporal_score = analyze_temporal_consistency(frames)
        motion_score = analyze_motion_patterns(frames)
        compression_score = analyze_video_compression_artifacts(frames)
        
        # Object detection in video frames
        detected_objects = detect_objects_in_video(frames)
        
        # Calculate individual metrics
        avg_frame_score = np.mean(frame_scores)
        frame_consistency = 1 - np.std(frame_scores) if len(frame_scores) > 1 else 0.5
        
        # Enhanced video detection
        enhanced_result = None
        if enhanced_ai_detection_pipeline:
            try:
                enhanced_result = enhanced_ai_detection_pipeline(frames, 'video')
            except:
                pass
        
        # Weighted combination of all factors
        base_prob = (
            avg_frame_score * 0.4 +
            temporal_score * 0.25 +
            motion_score * 0.2 +
            compression_score * 0.15
        )
        
        # Integrate enhanced detection if available
        if enhanced_result and 'enhanced_ai_probability' in enhanced_result:
            final_prob = (base_prob * 0.7 + enhanced_result['enhanced_ai_probability'] * 0.3)
        else:
            final_prob = base_prob
        
        # Use configuration-based thresholds if available
        if get_detection_result:
            result, confidence = get_detection_result(final_prob, 'video')
        else:
            # Fallback to enhanced result determination
            if final_prob > 0.75:
                result = "AI-Generated (High Confidence)"
                confidence = min(0.95, final_prob + 0.1)
            elif final_prob > 0.55:
                result = "Likely AI-Generated"
                confidence = final_prob
            elif final_prob > 0.35:
                result = "Possibly AI-Generated"
                confidence = final_prob * 0.8
            else:
                result = "Likely Real"
                confidence = 1 - final_prob
        
        return {
            "result": result,
            "confidence": confidence,
            "ai_probability": final_prob,
            "details": {
                "frames_analyzed": len(frame_scores),
                "avg_frame_score": avg_frame_score,
                "temporal_consistency": temporal_score,
                "motion_analysis": motion_score,
                "compression_artifacts": compression_score,
                "frame_consistency": frame_consistency,
                "detected_objects": detected_objects
            }
        }
        
    except Exception as e:
        return {"result": "Error", "confidence": 0, "error": str(e)}

def analyze_temporal_consistency(frames):
    """Analyze temporal consistency between consecutive frames."""
    if len(frames) < 2:
        return 0.5
    
    try:
        consistencies = []
        
        for i in range(len(frames) - 1):
            frame1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            # Calculate structural similarity
            diff = cv2.absdiff(frame1, frame2)
            consistency = 1 - (np.mean(diff) / 255.0)
            consistencies.append(consistency)
        
        avg_consistency = np.mean(consistencies)
        
        # AI videos often have unnaturally high temporal consistency
        if avg_consistency > 0.95:
            return 0.85  # Too consistent, likely AI
        elif avg_consistency > 0.85:
            return 0.65
        elif avg_consistency < 0.3:
            return 0.7   # Too inconsistent, also suspicious
        else:
            return 0.25  # Natural variation
            
    except Exception:
        return 0.5

def analyze_motion_patterns(frames):
    """Analyze motion patterns for AI-generated characteristics."""
    if len(frames) < 3:
        return 0.5
    
    try:
        motion_vectors = []
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, 
                np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                motion_magnitude = np.linalg.norm(flow[0][0])
                motion_vectors.append(motion_magnitude)
        
        if not motion_vectors:
            return 0.5
        
        motion_variance = np.var(motion_vectors)
        avg_motion = np.mean(motion_vectors)
        
        # AI videos often have unnatural motion patterns
        if motion_variance < 1.0 and avg_motion < 5.0:
            return 0.8  # Too uniform motion
        elif motion_variance < 5.0:
            return 0.6
        else:
            return 0.3
            
    except Exception:
        return 0.5

def analyze_video_compression_artifacts(frames):
    """Analyze compression artifacts specific to AI-generated videos."""
    try:
        artifact_scores = []
        
        for frame in frames[:5]:  # Analyze first 5 frames
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect blocking artifacts
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            artifact_score = np.var(laplacian)
            artifact_scores.append(artifact_score)
        
        if not artifact_scores:
            return 0.5
        
        avg_artifacts = np.mean(artifact_scores)
        
        # AI videos often have specific compression characteristics
        if avg_artifacts < 100:
            return 0.75  # Low artifacts, possibly AI
        elif avg_artifacts < 500:
            return 0.5
        else:
            return 0.25
            
    except Exception:
        return 0.5

def detect_objects_in_video(frames):
    """Enhanced video object detection with improved accuracy."""
    try:
        model = get_detection_model()
        coco_names = get_coco_names()
        
        all_detections = []
        frame_objects_list = []
        confidence_scores = []
        
        # Analyze more frames for better accuracy (up to 7 frames)
        max_frames = min(7, len(frames))
        
        for i in range(max_frames):
            frame = frames[i]
            # Convert frame to PIL Image with better resolution
            pil_frame = Image.fromarray(frame)
            
            # Resize for better detection (larger size = better accuracy)
            if pil_frame.size[0] < 640 or pil_frame.size[1] < 480:
                pil_frame = pil_frame.resize((640, 480), Image.Resampling.LANCZOS)
            
            # Preprocess for detection
            from utils import preprocess_image
            img_tensor = preprocess_image(pil_frame, detection=True)
            
            # Run detection with model
            with torch.no_grad():
                predictions = model(img_tensor)
            
            frame_objects = []
            for j in range(len(predictions[0]["boxes"])):
                confidence = predictions[0]["scores"][j].item()
                # Lower confidence threshold for better recall
                if confidence > 0.3:
                    box = predictions[0]["boxes"][j].detach().cpu().numpy().tolist()
                    label_index = predictions[0]["labels"][j].item()
                    label = coco_names[label_index]
                    
                    # Skip background class
                    if label != '__background__' and label != 'N/A':
                        detection = {
                            "label": label,
                            "confidence": confidence,
                            "box": box,
                            "frame": i,
                            "area": (box[2] - box[0]) * (box[3] - box[1])
                        }
                        frame_objects.append(detection)
                        confidence_scores.append(confidence)
            
            frame_objects_list.append(frame_objects)
            all_detections.extend(frame_objects)
        
        # Enhanced object counting with confidence weighting
        weighted_counts = Counter()
        high_conf_counts = Counter()
        
        for detection in all_detections:
            label = detection['label']
            conf = detection['confidence']
            
            # Weight by confidence and area (larger objects more reliable)
            weight = conf * min(1.0, detection['area'] / 10000)
            weighted_counts[label] += weight
            
            # High confidence detections (>0.7)
            if conf > 0.7:
                high_conf_counts[label] += 1
        
        # Temporal consistency check
        consistent_objects = analyze_temporal_consistency_objects(frame_objects_list)
        
        # Filter and rank objects by reliability
        reliable_objects = {}
        for label in weighted_counts:
            total_weight = weighted_counts[label]
            high_conf = high_conf_counts.get(label, 0)
            consistency = consistent_objects.get(label, 0)
            
            # Reliability score combining confidence, consistency, and frequency
            reliability = (total_weight * 0.4 + high_conf * 0.3 + consistency * 0.3)
            
            if reliability > 0.5:  # Only include reliable detections
                reliable_objects[label] = int(round(total_weight))
        
        # Enhanced summary with accuracy metrics
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        summary = {
            "total_objects": len([d for d in all_detections if d['confidence'] > 0.5]),
            "unique_objects": len(reliable_objects),
            "object_counts": dict(sorted(reliable_objects.items(), key=lambda x: x[1], reverse=True)[:10]),
            "detections": sorted([d for d in all_detections if d['confidence'] > 0.5], 
                               key=lambda x: x['confidence'], reverse=True)[:15],
            "accuracy_metrics": {
                "avg_confidence": avg_confidence,
                "frames_analyzed": max_frames,
                "high_confidence_detections": len([d for d in all_detections if d['confidence'] > 0.7]),
                "temporal_consistency": len(consistent_objects)
            }
        }
        
        return summary
        
    except Exception as e:
        return {"error": str(e), "total_objects": 0}

def analyze_temporal_consistency_objects(frame_objects_list):
    """Analyze temporal consistency of detected objects across frames."""
    try:
        object_appearances = Counter()
        
        for frame_objects in frame_objects_list:
            frame_labels = set()
            for obj in frame_objects:
                if obj['confidence'] > 0.5:
                    frame_labels.add(obj['label'])
            
            for label in frame_labels:
                object_appearances[label] += 1
        
        # Objects appearing in multiple frames are more reliable
        total_frames = len(frame_objects_list)
        consistent_objects = {}
        
        for label, appearances in object_appearances.items():
            consistency_ratio = appearances / total_frames
            if consistency_ratio > 0.3:  # Appears in at least 30% of frames
                consistent_objects[label] = appearances
        
        return consistent_objects
        
    except Exception:
        return {}
