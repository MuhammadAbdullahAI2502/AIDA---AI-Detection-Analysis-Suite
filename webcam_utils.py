"""
Enhanced Webcam Utilities
Advanced webcam processing and controls
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance

def enhance_webcam_image(image):
    """Enhanced image processing for webcam captures."""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Auto-adjust brightness and contrast
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced_rgb)
    except:
        return image

def apply_webcam_filters(image, filter_type="enhance"):
    """Apply various filters to webcam images."""
    try:
        if filter_type == "enhance":
            # Brightness and contrast enhancement
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
        elif filter_type == "denoise":
            # Noise reduction
            img_array = np.array(image)
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            image = Image.fromarray(denoised)
            
        elif filter_type == "auto_adjust":
            # Auto color adjustment
            img_array = np.array(image)
            
            # Auto white balance
            result = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
            
            image = Image.fromarray(result.astype(np.uint8))
        
        return image
    except:
        return image

def get_webcam_quality_settings(quality_level):
    """Get webcam quality settings based on user selection."""
    settings = {
        "High (640x480)": {"width": 640, "height": 480, "quality": 95},
        "Medium (320x240)": {"width": 320, "height": 240, "quality": 85},
        "Low (160x120)": {"width": 160, "height": 120, "quality": 75}
    }
    
    return settings.get(quality_level, settings["High (640x480)"])

def analyze_webcam_image_quality(image):
    """Analyze webcam image quality and provide recommendations."""
    try:
        img_array = np.array(image)
        
        # Calculate image quality metrics
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = np.std(gray)
        
        # Quality assessment
        quality_score = 0
        recommendations = []
        
        if sharpness > 100:
            quality_score += 25
        else:
            recommendations.append("📷 Try holding camera steady for sharper images")
        
        if 80 <= brightness <= 180:
            quality_score += 25
        else:
            if brightness < 80:
                recommendations.append("💡 Increase lighting for better visibility")
            else:
                recommendations.append("🌞 Reduce lighting to avoid overexposure")
        
        if contrast > 30:
            quality_score += 25
        else:
            recommendations.append("🎨 Improve contrast by adjusting lighting")
        
        # Overall assessment
        if quality_score >= 75:
            quality_status = "Excellent"
        elif quality_score >= 50:
            quality_status = "Good"
        elif quality_score >= 25:
            quality_status = "Fair"
        else:
            quality_status = "Poor"
        
        return {
            "quality_score": quality_score,
            "quality_status": quality_status,
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "recommendations": recommendations
        }
    except:
        return {
            "quality_score": 50,
            "quality_status": "Unknown",
            "recommendations": ["Unable to analyze image quality"]
        }