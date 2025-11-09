import cv2
import numpy as np
from scipy import stats

from libs_week3.preprocessing import ImagePreprocessStep

def detect_noise(image, noise_threshold=0.045):
    
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 0]  # L channel only
    else:
        gray = image.copy()
        
    gray_norm = gray.astype(np.float32) / 255.0
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_var = laplacian.var()
    
    blurred = cv2.GaussianBlur(gray_norm, (5,5), 1.0)
    
    noise = gray_norm - blurred
    
    noise_flat = noise.flatten()
    kurtosis_val = stats.kurtosis(noise_flat, fisher=True)
    
    noise_std = np.std(noise_flat)
    
    # Calculate signal-to-noise ratio for better detection
    signal_std = np.std(gray_norm)
    snr = signal_std / noise_std if noise_std > 0 else float('inf')
    
    hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    
    gray_float = gray.astype(np.float32)
    median = cv2.medianBlur(gray_float, 3)
    diff = np.abs(gray_float - median)
    std_local = np.std(gray_float)

    # Pixels much higher/lower than local median
    salt_mask = diff > (3 * std_local)
    impulse_ratio = np.sum(salt_mask) / gray.size

    # Detect salt vs pepper
    pepper_mask = (gray_float < median - 3 * std_local)
    salt_mask = (gray_float > median + 3 * std_local)
    salt_ratio = np.sum(salt_mask) / gray.size
    pepper_ratio = np.sum(pepper_mask) / gray.size

    has_salt = salt_ratio > 0.005
    has_pepper = pepper_ratio > 0.005
    has_salt_pepper = (salt_ratio + pepper_ratio) > 0.02
    
    # print(f"Has noise: {has_salt_pepper}, Impulse ratio: {impulse_ratio:.4f}")
  
    # More conservative noise detection
    has_noise = noise_std >= noise_threshold and snr < 15
    
    noise_type = ""
    confidence = 0.0
    
    # print(f"Kurtosis: {kurtosis_val:.2f}, Impulse ratio: {impulse_ratio:.4f}, SNR: {snr:.2f}")
    if not has_noise:
        noise_type = "none"
        confidence = 1.0
        noise_level = "none"
        
    elif impulse_ratio != 0.0:
        if ((kurtosis_val > 5.0 or (has_salt and has_pepper))) and snr < 4.0:
            noise_type = "salt_and_pepper"
            confidence = min(1.0, (kurtosis_val / 10.0) if kurtosis_val > 0 else 0.5)
        elif -0.5 <= kurtosis_val < 5.0:
            noise_type = "gaussian"
            confidence = 1.0 - abs(kurtosis_val) / 3.0
        # elif kurtosis_val < -0.5:
        #     noise_type = "uniform"
        #     confidence = min(1.0, abs(kurtosis_val) / 2.0)
        # else:
        #     noise_type = "mixed"
        #     confidence = 0.5
        
    # Adjusted thresholds for noise levels
    if noise_std < 0.025:
        noise_level = "very_low"
    elif noise_std < 0.06:
        noise_level = "low"
    elif noise_std < 0.12:
        noise_level = "medium"
    elif noise_std < 0.25:
        noise_level = "high"
    else:
        noise_level = "very_high"
        
    return {
        'noise_type': noise_type,
        'kurtosis': kurtosis_val,
        'noise_std': noise_std,
        'noise_level': noise_level,
        'laplacian_variance': noise_var,
        'has_salt': has_salt,
        'has_pepper': has_pepper,
        'confidence': confidence,
        'snr': snr
    }


class DenoiseWithNonLocalMeans(ImagePreprocessStep):
    def __init__(self, h=10, template_window_size=7, search_window_size=21):
        super().__init__()
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        detected_noise = detect_noise(image)['noise_type']
        if detected_noise != 'salt_and_pepper':
            return image, mask

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Apply NLM to L channel
        lab[:, :, 0] = cv2.fastNlMeansDenoising(
            lab[:, :, 0], 
            None, 
            h=self.h, 
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size
        )
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image, mask

    def to_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'h': self.h,
            'template_window_size': self.template_window_size,
            'search_window_size': self.search_window_size,
        }


class DenoiseWithMedianFilter(ImagePreprocessStep):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kernel_size = self.kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Denoise only the L channel
        lab[:, :, 0] = cv2.medianBlur(lab[:, :, 0], kernel_size)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image, mask

    def to_dict(self) -> dict:
        return {
            'class': self.__class__.__name__,
            'kernel_size': self.kernel_size,
        }

