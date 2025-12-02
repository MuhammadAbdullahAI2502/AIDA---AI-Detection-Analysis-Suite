"""
Advanced AI Detection Utilities
Enhanced algorithms for accurate detection of AI-generated content
"""

import numpy as np
import cv2
from PIL import Image
import re
from collections import Counter
from scipy.fft import fft2, fftshift
from scipy.stats import entropy, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

class AdvancedImageDetector:
    """Advanced image analysis for AI detection"""
    
    @staticmethod
    def detect_deepfake_artifacts(image_array):
        """Detect specific deepfake artifacts in images"""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            
            eye_score = 0.3  # Default score
            if len(eyes) >= 2:
                # Analyze eye symmetry and consistency
                eye_areas = [w * h for (x, y, w, h) in eyes]
                if len(eye_areas) >= 2:
                    area_ratio = min(eye_areas) / max(eye_areas)
                    if area_ratio < 0.7:  # Asymmetric eyes
                        eye_score = 0.8
                    elif area_ratio < 0.85:
                        eye_score = 0.6
            
            return eye_score
        except:
            return 0.3
    
    @staticmethod
    def analyze_pixel_interpolation(image_array):
        """Detect pixel interpolation patterns common in AI generation"""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate second-order derivatives
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Analyze interpolation artifacts
            interpolation_variance = np.var(laplacian)
            
            # AI images often have specific interpolation patterns
            if interpolation_variance < 50:
                return 0.85
            elif interpolation_variance < 150:
                return 0.6
            else:
                return 0.25
        except:
            return 0.5
    
    @staticmethod
    def detect_gan_fingerprints(image_array):
        """Detect GAN-specific fingerprints in images"""
        try:
            # Convert to frequency domain
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            f_transform = fft2(gray)
            f_shift = fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Analyze specific frequency patterns typical of GANs
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Check for periodic patterns (GAN artifacts)
            # Sample radial frequencies
            radial_profile = []
            for r in range(1, min(center_h, center_w) // 4):
                mask = np.zeros((h, w))
                y, x = np.ogrid[:h, :w]
                mask_condition = ((x - center_w)**2 + (y - center_h)**2 >= r**2) & \
                                ((x - center_w)**2 + (y - center_h)**2 < (r+1)**2)
                mask[mask_condition] = 1
                radial_profile.append(np.mean(magnitude_spectrum[mask == 1]))
            
            if len(radial_profile) > 5:
                # Check for unnatural periodicity
                profile_fft = np.abs(fft2(radial_profile))
                peak_ratio = np.max(profile_fft[1:]) / (np.mean(profile_fft[1:]) + 1e-8)
                
                if peak_ratio > 3.0:
                    return 0.8  # Strong GAN signature
                elif peak_ratio > 2.0:
                    return 0.6
                else:
                    return 0.3
            
            return 0.4
        except:
            return 0.4

class AdvancedTextDetector:
    """Advanced text analysis for AI detection"""
    
    @staticmethod
    def analyze_linguistic_complexity(text):
        """Analyze linguistic complexity patterns"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 0.5
            
            # Calculate readability metrics
            total_words = 0
            total_syllables = 0
            complex_words = 0
            
            for sentence in sentences:
                words = re.findall(r'\b\w+\b', sentence.lower())
                total_words += len(words)
                
                for word in words:
                    syllables = AdvancedTextDetector.count_syllables(word)
                    total_syllables += syllables
                    if syllables >= 3:
                        complex_words += 1
            
            if total_words == 0:
                return 0.5
            
            # Flesch Reading Ease approximation
            avg_sentence_length = total_words / len(sentences)
            avg_syllables_per_word = total_syllables / total_words
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # AI text often has specific readability patterns
            if 60 <= flesch_score <= 80:  # Optimal AI range
                return 0.7
            elif 50 <= flesch_score <= 90:
                return 0.5
            else:
                return 0.3
                
        except:
            return 0.5
    
    @staticmethod
    def count_syllables(word):
        """Simple syllable counting"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    @staticmethod
    def detect_ai_writing_patterns(text):
        """Detect specific AI writing patterns"""
        try:
            # Common AI phrases and structures
            ai_patterns = [
                r'\b(it\'s important to|it\'s worth|keep in mind|bear in mind)\b',
                r'\b(furthermore|moreover|additionally|however|nevertheless)\b',
                r'\b(in conclusion|to summarize|in summary|overall)\b',
                r'\b(comprehensive|extensive|significant|substantial)\b',
                r'\b(various|numerous|multiple|several)\b'
            ]
            
            text_lower = text.lower()
            pattern_matches = 0
            
            for pattern in ai_patterns:
                matches = len(re.findall(pattern, text_lower))
                pattern_matches += matches
            
            # Calculate pattern density
            word_count = len(re.findall(r'\b\w+\b', text))
            if word_count == 0:
                return 0.5
            
            pattern_density = pattern_matches / word_count
            
            if pattern_density > 0.05:
                return 0.85
            elif pattern_density > 0.02:
                return 0.65
            else:
                return 0.25
                
        except:
            return 0.5
    
    @staticmethod
    def analyze_sentence_flow(text):
        """Analyze sentence flow and transitions"""
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 3:
                return 0.5
            
            # Analyze sentence beginnings
            beginnings = []
            for sentence in sentences:
                words = sentence.split()
                if words:
                    beginnings.append(words[0].lower())
            
            # Check for repetitive sentence starters
            beginning_counts = Counter(beginnings)
            max_repetition = max(beginning_counts.values()) if beginning_counts else 1
            repetition_ratio = max_repetition / len(beginnings)
            
            # AI often has repetitive sentence structures
            if repetition_ratio > 0.4:
                return 0.8
            elif repetition_ratio > 0.25:
                return 0.6
            else:
                return 0.3
                
        except:
            return 0.5

class VideoAnalyzer:
    """Advanced video analysis utilities"""
    
    @staticmethod
    def detect_frame_interpolation(frames):
        """Detect artificial frame interpolation"""
        try:
            if len(frames) < 3:
                return 0.5
            
            interpolation_scores = []
            
            for i in range(1, len(frames) - 1):
                prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                next_frame = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
                
                # Calculate expected interpolated frame
                expected = (prev_frame.astype(np.float32) + next_frame.astype(np.float32)) / 2
                
                # Compare with actual frame
                diff = np.abs(curr_frame.astype(np.float32) - expected)
                similarity = 1 - (np.mean(diff) / 255.0)
                
                interpolation_scores.append(similarity)
            
            avg_similarity = np.mean(interpolation_scores)
            
            # High similarity suggests artificial interpolation
            if avg_similarity > 0.9:
                return 0.85
            elif avg_similarity > 0.8:
                return 0.65
            else:
                return 0.3
                
        except:
            return 0.5
    
    @staticmethod
    def analyze_compression_consistency(frames):
        """Analyze compression consistency across frames"""
        try:
            compression_metrics = []
            
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Calculate compression-related metrics
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                compression_metrics.append(laplacian_var)
            
            if len(compression_metrics) < 2:
                return 0.5
            
            # AI videos often have very consistent compression
            compression_cv = np.std(compression_metrics) / (np.mean(compression_metrics) + 1e-8)
            
            if compression_cv < 0.1:
                return 0.8  # Too consistent
            elif compression_cv < 0.2:
                return 0.6
            else:
                return 0.3
                
        except:
            return 0.5

def enhanced_ai_detection_pipeline(content, content_type):
    """
    Enhanced AI detection pipeline combining multiple advanced techniques
    
    Args:
        content: Image array, text string, or video frames
        content_type: 'image', 'text', or 'video'
    
    Returns:
        Enhanced detection result with detailed analysis
    """
    
    if content_type == 'image':
        detector = AdvancedImageDetector()
        
        # Run multiple detection algorithms
        deepfake_score = detector.detect_deepfake_artifacts(content)
        interpolation_score = detector.analyze_pixel_interpolation(content)
        gan_score = detector.detect_gan_fingerprints(content)
        
        # Combine scores
        combined_score = (deepfake_score * 0.4 + interpolation_score * 0.3 + gan_score * 0.3)
        
        return {
            'enhanced_ai_probability': combined_score,
            'deepfake_artifacts': deepfake_score,
            'pixel_interpolation': interpolation_score,
            'gan_fingerprints': gan_score
        }
    
    elif content_type == 'text':
        detector = AdvancedTextDetector()
        
        complexity_score = detector.analyze_linguistic_complexity(content)
        pattern_score = detector.detect_ai_writing_patterns(content)
        flow_score = detector.analyze_sentence_flow(content)
        
        combined_score = (complexity_score * 0.4 + pattern_score * 0.35 + flow_score * 0.25)
        
        return {
            'enhanced_ai_probability': combined_score,
            'linguistic_complexity': complexity_score,
            'ai_patterns': pattern_score,
            'sentence_flow': flow_score
        }
    
    elif content_type == 'video':
        analyzer = VideoAnalyzer()
        
        interpolation_score = analyzer.detect_frame_interpolation(content)
        compression_score = analyzer.analyze_compression_consistency(content)
        
        combined_score = (interpolation_score * 0.6 + compression_score * 0.4)
        
        return {
            'enhanced_ai_probability': combined_score,
            'frame_interpolation': interpolation_score,
            'compression_consistency': compression_score
        }
    
    else:
        return {'enhanced_ai_probability': 0.5, 'error': 'Unknown content type'}