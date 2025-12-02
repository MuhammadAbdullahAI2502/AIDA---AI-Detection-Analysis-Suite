"""
Human vs AI Text Detection Calibration
Improved thresholds to reduce false positives
"""

def calibrate_text_detection(text, base_score):
    """
    Calibrate text detection to reduce false positives for human-written content
    """
    
    # Human writing indicators
    human_indicators = [
        # Personal expressions
        r'\bi\s+think\b', r'\bi\s+feel\b', r'\bin\s+my\s+opinion\b', r'\bi\s+believe\b',
        # Casual language
        r'\byeah\b', r'\bokay\b', r'\bwell\b', r'\bactually\b', r'\bbasically\b',
        # Contractions
        r"don't\b", r"can't\b", r"won't\b", r"i'm\b", r"it's\b", r"that's\b",
        # Informal transitions
        r'\banyway\b', r'\bso\b', r'\bbut\b', r'\balso\b'
    ]
    
    # Count human indicators
    import re
    text_lower = text.lower()
    human_count = sum(1 for pattern in human_indicators if re.search(pattern, text_lower))
    
    # Calculate adjustment factor
    text_length = len(text.split())
    human_density = human_count / max(1, text_length / 10)  # Per 10 words
    
    # Reduce AI probability if human indicators are present
    if human_density > 0.3:
        adjustment = -0.3  # Strong human indicators
    elif human_density > 0.15:
        adjustment = -0.2  # Moderate human indicators
    elif human_density > 0.05:
        adjustment = -0.1  # Some human indicators
    else:
        adjustment = 0
    
    # Apply adjustment
    calibrated_score = max(0.0, min(1.0, base_score + adjustment))
    
    return calibrated_score