"""
AI Detection Configuration
Fine-tuned parameters for optimal detection accuracy
"""

# Image Detection Thresholds
IMAGE_DETECTION_CONFIG = {
    'spectral_analysis': {
        'low_freq_threshold': 0.15,
        'high_freq_threshold': 0.25,
        'natural_freq_threshold': 0.6
    },
    'texture_analysis': {
        'entropy_threshold_low': 4.5,
        'entropy_threshold_high': 5.5,
        'gabor_variance_low': 1000,
        'gabor_variance_high': 2000
    },
    'edge_detection': {
        'sharpness_low': 15,
        'sharpness_high': 80,
        'density_low': 0.05,
        'density_high': 0.3
    },
    'color_analysis': {
        'entropy_threshold': 6.5,
        'saturation_threshold': 5.0
    },
    'frequency_domain': {
        'dct_std_low': 50,
        'dct_std_high': 100,
        'wavelet_energy_low': 1e6,
        'wavelet_energy_high': 5e6
    }
}

# Text Detection Thresholds - Adjusted for better accuracy
TEXT_DETECTION_CONFIG = {
    'perplexity': {
        'cv_low': 0.2,
        'cv_medium': 0.4
    },
    'semantic_coherence': {
        'similarity_high': 0.8,
        'similarity_medium': 0.6,
        'similarity_low': 0.15
    },
    'syntactic_patterns': {
        'complexity_variance_low': 0.005,
        'complexity_variance_medium': 0.015,
        'complexity_range_low': 0.35,
        'complexity_range_high': 0.55
    },
    'stylistic_features': {
        'transition_density_high': 0.4,
        'transition_density_medium': 0.25,
        'formal_count_high': 4,
        'formal_count_medium': 2
    },
    'ngram_patterns': {
        'bigram_repetition_high': 0.4,
        'bigram_repetition_medium': 0.25,
        'trigram_repetition_high': 0.25,
        'trigram_repetition_medium': 0.15
    }
}

# Video Detection Thresholds
VIDEO_DETECTION_CONFIG = {
    'temporal_consistency': {
        'consistency_very_high': 0.95,
        'consistency_high': 0.85,
        'consistency_low': 0.3
    },
    'motion_analysis': {
        'variance_low': 1.0,
        'motion_low': 5.0,
        'variance_medium': 5.0
    },
    'compression_artifacts': {
        'artifact_low': 100,
        'artifact_medium': 500
    }
}

# Overall Detection Thresholds - More conservative for text
DETECTION_THRESHOLDS = {
    'high_confidence': 0.80,
    'medium_confidence': 0.65,
    'low_confidence': 0.45
}

# Confidence Adjustments
CONFIDENCE_ADJUSTMENTS = {
    'high_confidence_boost': 0.1,
    'medium_confidence_factor': 0.8,
    'max_confidence': 0.95
}

# Model Weights for Ensemble - Adjusted for text accuracy
MODEL_WEIGHTS = {
    'image': {
        'spectral': 0.25,
        'texture': 0.20,
        'edge': 0.20,
        'color': 0.15,
        'frequency': 0.15,
        'metadata': 0.05
    },
    'text': {
        'perplexity': 0.15,
        'semantic': 0.15,
        'syntactic': 0.25,
        'stylistic': 0.30,
        'ngram': 0.10,
        'pos': 0.05
    },
    'video': {
        'frame_analysis': 0.4,
        'temporal': 0.25,
        'motion': 0.2,
        'compression': 0.15
    }
}

# Enhanced Detection Integration
ENHANCED_DETECTION_WEIGHTS = {
    'base_model_weight': 0.7,
    'enhanced_model_weight': 0.3
}

# Performance Optimization
PERFORMANCE_CONFIG = {
    'max_video_frames': 10,
    'max_video_size_mb': 200,
    'image_resize_target': (256, 256),
    'text_min_length': 50
}

def get_detection_result(ai_probability, content_type='general'):
    """
    Get standardized detection result based on probability and thresholds
    
    Args:
        ai_probability (float): AI probability score (0-1)
        content_type (str): Type of content being analyzed
    
    Returns:
        tuple: (result_text, confidence_score)
    """
    
    thresholds = DETECTION_THRESHOLDS
    adjustments = CONFIDENCE_ADJUSTMENTS
    
    if ai_probability > thresholds['high_confidence']:
        result = f"AI-Generated (High Confidence)"
        confidence = min(adjustments['max_confidence'], 
                        ai_probability + adjustments['high_confidence_boost'])
    elif ai_probability > thresholds['medium_confidence']:
        result = "Likely AI-Generated"
        confidence = ai_probability
    elif ai_probability > thresholds['low_confidence']:
        result = "Possibly AI-Generated"
        confidence = ai_probability * adjustments['medium_confidence_factor']
    else:
        result = "Likely Real"
        confidence = 1 - ai_probability
    
    return result, confidence

def get_model_weights(content_type):
    """Get model weights for specific content type"""
    return MODEL_WEIGHTS.get(content_type, MODEL_WEIGHTS['image'])

def get_config(content_type, analysis_type=None):
    """Get configuration for specific analysis"""
    configs = {
        'image': IMAGE_DETECTION_CONFIG,
        'text': TEXT_DETECTION_CONFIG,
        'video': VIDEO_DETECTION_CONFIG
    }
    
    config = configs.get(content_type, IMAGE_DETECTION_CONFIG)
    
    if analysis_type and analysis_type in config:
        return config[analysis_type]
    
    return config