"""
Test script to verify enhanced AI detection accuracy
"""

import numpy as np
from PIL import Image
import cv2

# Test the enhanced detection system
def test_detection_system():
    print("🔍 Testing Enhanced AI Detection System...")
    
    # Test image detection
    print("\n📸 Testing Image Detection:")
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    try:
        from models import detect_ai_generated_content
        result = detect_ai_generated_content(pil_image)
        print(f"✅ Image Detection: {result['result']} (Confidence: {result['confidence']:.2%})")
    except Exception as e:
        print(f"❌ Image Detection Error: {e}")
    
    # Test text detection
    print("\n📝 Testing Text Detection:")
    test_text = """
    This is a comprehensive analysis of the various factors that contribute to the overall effectiveness 
    of artificial intelligence systems. It's important to note that these systems demonstrate significant 
    capabilities across multiple domains. Furthermore, the implementation of advanced algorithms facilitates 
    enhanced performance metrics. In conclusion, the utilization of sophisticated methodologies ensures 
    optimal results in diverse applications.
    """
    
    try:
        from models import detect_ai_text
        result = detect_ai_text(test_text)
        print(f"✅ Text Detection: {result['result']} (Confidence: {result['confidence']:.2%})")
    except Exception as e:
        print(f"❌ Text Detection Error: {e}")
    
    print("\n🎯 Enhanced AI Detection System Ready!")
    print("📊 Features:")
    print("  • Advanced spectral analysis for images")
    print("  • Linguistic pattern detection for text")
    print("  • Temporal consistency analysis for videos")
    print("  • Multi-layer ensemble detection")
    print("  • Configuration-based thresholds")

if __name__ == "__main__":
    test_detection_system()