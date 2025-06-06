#!/usr/bin/env python3
# utils/helpers.py
import os
import re
import time
import random
import logging
import hashlib
import requests
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from bs4 import BeautifulSoup
import speech_recognition as sr
from utils.logger import setup_logging

# Load environment variables
load_dotenv()
logger = setup_logging("output/logs/passripperhydra.log")

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 5
PROXY_TIMEOUT = 30
CAPTCHA_TIMEOUT = 60
IMAGE_PROCESSING_TIMEOUT = 10
MODEL_INPUT_SIZE = (64, 128)  # height, width
CHAR_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CAPTCHA_LENGTH = 6  # Typical captcha length
MAX_IMAGE_SIZE_MB = 5

# Configure requests session with retry logic
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[408, 429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

class ModelLoader:
    """Singleton class to manage ML model loading"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Lazy load models when first needed"""
        self.captcha_model = None
        try:
            self.captcha_model = self._load_captcha_model()
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
    
    def _load_captcha_model(self) -> tf.keras.Model:
        """Load and cache the captcha model"""
        model_path = os.getenv("CAPTCHA_MODEL_PATH", "models/captcha_cnn.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading captcha model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model

class SecurityUtils:
    """Security-related helper functions"""
    
    @staticmethod
    def sanitize_input(input_str: str, input_type: str = "generic") -> str:
        """
        Sanitize input based on type with strict validation
        Args:
            input_str: Input string to sanitize
            input_type: Type of input (url, path, generic)
        Returns:
            Sanitized string
        Raises:
            ValueError: If input is invalid
        """
        if not input_str or not isinstance(input_str, str):
            return ""
        
        # Basic sanitization
        sanitized = re.sub(r'[;&|<>\"\']', '', input_str).strip()
        
        # Type-specific validation
        if input_type == "url":
            if not SecurityUtils._is_valid_url(sanitized):
                raise ValueError(f"Invalid URL: {sanitized}")
        elif input_type == "path":
            if not SecurityUtils._is_safe_path(sanitized):
                raise ValueError(f"Invalid path: {sanitized}")
        
        return sanitized
    
    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Validate URL format and safety"""
        url_pattern = re.compile(
            r'^(https?://)?'  # http:// or https://
            r'(([A-Z0-9-]+\.)+[A-Z]{2,63})'  # domain
            r'(:\d+)?'  # port
            r'(/.*)?$',  # path
            re.IGNORECASE
        )
        return bool(url_pattern.match(url))
    
    @staticmethod
    def _is_safe_path(path: str) -> bool:
        """Validate path safety"""
        if not os.path.abspath(path).startswith(os.getcwd()):
            return False
        return True

class CaptchaSolver:
    """Handles captcha solving with multiple fallback strategies"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.captcha_apis = {
            "2captcha": os.getenv("2CAPTCHA_API_KEY"),
            "anti-captcha": os.getenv("ANTI_CAPTCHA_API_KEY")
        }
    
    def solve(self, target_url: str, attempt: int = 1) -> Optional[str]:
        """
        Solve captcha with multiple fallback methods
        Args:
            target_url: URL where captcha appears
            attempt: Current attempt number
        Returns:
            Solved captcha text or None
        """
        strategies = [
            self._solve_with_service,
            self._solve_with_ml,
            self._solve_with_audio
        ]
        
        for strategy in strategies:
            try:
                solution = strategy(target_url, attempt)
                if solution:
                    return solution
            except Exception as e:
                logger.warning(f"Captcha solving attempt failed: {str(e)}")
                continue
        
        return None
    
    def _solve_with_service(self, target_url: str, attempt: int) -> Optional[str]:
        """Solve using paid captcha solving service"""
        for service, api_key in self.captcha_apis.items():
            if not api_key:
                continue
                
            try:
                if service == "2captcha":
                    return self._solve_2captcha(target_url, api_key)
                elif service == "anti-captcha":
                    return self._solve_anticaptcha(target_url, api_key)
            except Exception as e:
                logger.warning(f"{service} failed: {str(e)}")
        
        return None
    
    def _solve_2captcha(self, target_url: str, api_key: str) -> Optional[str]:
        """Solve using 2Captcha API"""
        params = {
            "key": api_key,
            "method": "userrecaptcha",
            "googlekey": self._extract_captcha_key(target_url),
            "pageurl": target_url,
            "json": 1
        }
        
        response = session.post(
            "https://2captcha.com/in.php",
            data=params,
            timeout=CAPTCHA_TIMEOUT
        ).json()
        
        if response.get("status") != 1:
            raise ValueError(f"2Captcha error: {response.get('request', 'Unknown error')}")
        
        captcha_id = response["request"]
        start_time = time.time()
        
        while time.time() - start_time < CAPTCHA_TIMEOUT:
            time.sleep(RETRY_DELAY)
            result = session.get(
                f"https://2captcha.com/res.php?key={api_key}&action=get&id={captcha_id}&json=1",
                timeout=CAPTCHA_TIMEOUT
            ).json()
            
            if result.get("status") == 1:
                return result["request"]
            elif result.get("request") != "CAPCHA_NOT_READY":
                raise ValueError(f"2Captcha error: {result.get('request', 'Unknown error')}")
        
        raise TimeoutError("2Captcha solving timed out")
    
    def _solve_anticaptcha(self, target_url: str, api_key: str) -> Optional[str]:
        """Solve using Anti-Captcha API"""
        payload = {
            "clientKey": api_key,
            "task": {
                "type": "NoCaptchaTaskProxyless",
                "websiteURL": target_url,
                "websiteKey": self._extract_captcha_key(target_url)
            }
        }
        
        response = session.post(
            "https://api.anti-captcha.com/createTask",
            json=payload,
            timeout=CAPTCHA_TIMEOUT
        ).json()
        
        if response.get("errorId", 1) != 0:
            raise ValueError(f"Anti-Captcha error: {response.get('errorDescription', 'Unknown error')}")
        
        task_id = response["taskId"]
        start_time = time.time()
        
        while time.time() - start_time < CAPTCHA_TIMEOUT:
            time.sleep(RETRY_DELAY)
            result = session.post(
                "https://api.anti-captcha.com/getTaskResult",
                json={"clientKey": api_key, "taskId": task_id},
                timeout=CAPTCHA_TIMEOUT
            ).json()
            
            if result.get("status") == "ready":
                return result["solution"]["gRecaptchaResponse"]
            elif result.get("errorId", 0) != 0:
                raise ValueError(f"Anti-Captcha error: {result.get('errorDescription', 'Unknown error')}")
        
        raise TimeoutError("Anti-Captcha solving timed out")
    
    def _solve_with_ml(self, target_url: str, attempt: int) -> Optional[str]:
        """Solve using ML model"""
        if not self.model_loader.captcha_model:
            raise RuntimeError("No ML model available")
        
        try:
            image_url = self._get_captcha_image_url(target_url)
            image = self._download_and_preprocess_image(image_url)
            image = image.reshape(1, *MODEL_INPUT_SIZE, 1)  # Add batch and channel dimensions
            prediction = self.model_loader.captcha_model.predict(image)
            solution = self._decode_prediction(prediction)
            return solution
        except Exception as e:
            logger.warning(f"ML solving failed: {str(e)}")
            return None
    
    def _solve_with_audio(self, target_url: str, attempt: int) -> Optional[str]:
        """Solve audio captcha fallback using speech recognition"""
        try:
            # Extract audio captcha URL (simplified for demonstration)
            audio_url = f"{target_url}/audio-captcha.mp3"  # Placeholder
            response = session.get(audio_url, timeout=IMAGE_PROCESSING_TIMEOUT)
            response.raise_for_status()
            
            # Save audio to temporary file
            audio_file = "temp_audio_captcha.mp3"
            with open(audio_file, "wb") as f:
                f.write(response.content)
            
            # Convert audio to text using speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                solution = recognizer.recognize_google(audio).lower()
            
            logger.info(f"Audio captcha solved: {solution}")
            return solution
        except Exception as e:
            logger.warning(f"Audio captcha solving failed: {str(e)}")
            return None
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)
    
    def _extract_captcha_key(self, target_url: str) -> str:
        """Extract captcha site key from page"""
        try:
            response = session.get(target_url, timeout=CAPTCHA_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for reCAPTCHA site key
            captcha_div = soup.find('div', class_='g-recaptcha')
            if captcha_div and 'data-sitekey' in captcha_div.attrs:
                return captcha_div['data-sitekey']
            
            # Look for hCaptcha site key
            hcaptcha_div = soup.find('div', class_='h-captcha')
            if hcaptcha_div and 'data-sitekey' in hcaptcha_div.attrs:
                return hcaptcha_div['data-sitekey']
            
            logger.warning("Could not extract captcha site key.")
            return "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-"  # Fallback
        except Exception as e:
            logger.error(f"Failed to extract captcha key: {str(e)}")
            return "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-"  # Fallback
    
    def _get_captcha_image_url(self, target_url: str) -> str:
        """Get captcha image URL from target page"""
        try:
            response = session.get(target_url, timeout=CAPTCHA_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for common captcha image tags
            captcha_img = soup.find('img', id=lambda x: x and 'captcha' in x.lower())
            if not captcha_img:
                captcha_img = soup.find('img', class_=lambda x: x and 'captcha' in x.lower())
            if not captcha_img:
                captcha_img = soup.find('img', src=lambda x: x and 'captcha' in x.lower())
            
            if captcha_img and 'src' in captcha_img.attrs:
                image_url = captcha_img['src']
                # Resolve relative URLs
                if not image_url.startswith('http'):
                    base_url = "/".join(target_url.split("/")[:3])
                    image_url = base_url + ("" if image_url.startswith("/") else "/") + image_url
                return image_url
            
            raise ValueError("Could not find captcha image URL.")
        except Exception as e:
            logger.error(f"Failed to extract captcha image URL: {str(e)}")
            raise
    
    @staticmethod
    def _download_and_preprocess_image(url: str) -> np.ndarray:
        """Download and preprocess captcha image"""
        response = session.get(url, timeout=IMAGE_PROCESSING_TIMEOUT)
        response.raise_for_status()
        
        if int(response.headers.get('content-length', 0)) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise ValueError("Image too large")
        
        img = Image.open(BytesIO(response.content)).convert('L')
        img = img.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
        return np.array(img) / 255.0
    
    @staticmethod
    def _decode_prediction(prediction: np.ndarray) -> str:
        """Convert model output to captcha text"""
        prediction = prediction.reshape(CAPTCHA_LENGTH, len(CHAR_SET))
        return "".join([CHAR_SET[i] for i in np.argmax(prediction, axis=-1)])

class ProxyManager:
    """Handles proxy rotation and management"""
    
    def __init__(self):
        self.proxy_sources = {
            "brightdata": os.getenv("BRIGHTDATA_API_KEY"),
            "proxyrack": os.getenv("PROXYRACK_API_KEY")
        }
        self.current_proxy = None
        self.last_rotation = 0
        self.rotation_interval = 300  # Rotate every 5 minutes
    
    @lru_cache(maxsize=32)
    def get_proxy(self) -> Optional[str]:
        """
        Get a working proxy with rotation
        Returns:
            Proxy URL or None if no proxies available
        """
        if self.current_proxy and time.time() - self.last_rotation < self.rotation_interval:
            return self.current_proxy
        
        proxies = self._get_proxy_list()
        if not proxies:
            return None
        
        with ThreadPoolExecutor(max_workers=min(4, len(proxies))) as executor:
            working_proxies = list(filter(None, executor.map(self._test_proxy, proxies)))
        
        if working_proxies:
            self.current_proxy = random.choice(working_proxies)
            self.last_rotation = time.time()
            return self.current_proxy
        
        return None
    
    def _get_proxy_list(self) -> List[str]:
        """Get proxies from all available sources"""
        proxies = []
        
        # Paid services
        if self.proxy_sources["brightdata"]:
            proxies.extend(self._get_brightdata_proxies())
        if self.proxy_sources["proxyrack"]:
            proxies.extend(self._get_proxyrack_proxies())
        
        # Free fallback
        if not proxies:
            proxies.extend([
                "http://123.45.67.89:8080",
                "http://98.76.54.32:3128"
            ])
        
        return list(set(proxies))  # Remove duplicates
    
    def _get_brightdata_proxies(self) -> List[str]:
        """Get proxies from BrightData"""
        try:
            response = session.get(
                "https://api.brightdata.com/proxy/list",
                headers={"Authorization": f"Bearer {self.proxy_sources['brightdata']}"},
                timeout=PROXY_TIMEOUT
            )
            return [f"http://{p['ip']}:{p['port']}" for p in response.json().get("proxies", [])]
        except Exception as e:
            logger.warning(f"BrightData proxy fetch failed: {str(e)}")
            return []
    
    def _get_proxyrack_proxies(self) -> List[str]:
        """Get proxies from ProxyRack"""
        try:
            response = session.get(
                "https://api.proxyrack.com/v1/proxies",
                headers={"Authorization": f"Bearer {self.proxy_sources['proxyrack']}"},
                timeout=PROXY_TIMEOUT
            )
            return [f"http://{p['host']}:{p['port']}" for p in response.json().get("data", [])]
        except Exception as e:
            logger.warning(f"ProxyRack proxy fetch failed: {str(e)}")
            return []
    
    @staticmethod
    def _test_proxy(proxy: str) -> Optional[str]:
        """Test if a proxy is working"""
        try:
            test_url = "http://httpbin.org/ip"
            response = session.get(
                test_url,
                proxies={"http": proxy, "https": proxy},
                timeout=PROXY_TIMEOUT
            )
            if response.status_code == 200:
                return proxy
        except Exception:
            pass
        return None

class ResourceLoader:
    """Handles loading of external resources"""
    
    @staticmethod
    def load_logo(logo_path: str) -> str:
        """
        Load logo with validation
        Args:
            logo_path: Path to logo file
        Returns:
            Path to valid logo or placeholder
        """
        try:
            if os.path.exists(logo_path):
                with Image.open(logo_path) as img:
                    img.verify()  # Verify it's a valid image
                return logo_path
        except Exception as e:
            logger.warning(f"Invalid logo file: {str(e)}")
        
        return "https://via.placeholder.com/150?text=PassRipperHydra"
    
    @staticmethod
    def load_wordlist(path: str) -> List[str]:
        """
        Load wordlist with validation
        Args:
            path: Path to wordlist file
        Returns:
            List of words
        Raises:
            ValueError: If wordlist is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Wordlist not found at {path}")
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Failed to load wordlist: {str(e)}")

# Initialize global helpers
security = SecurityUtils()
captcha_solver = CaptchaSolver()
proxy_manager = ProxyManager()
resource_loader = ResourceLoader()

# Public interface functions
def solve_captcha(target_url: str, attempt: int = 1) -> Optional[str]:
    """Public interface for captcha solving"""
    return captcha_solver.solve(target_url, attempt)

def sanitize_input(input_str: str, input_type: str = "generic") -> str:
    """Public interface for input sanitization"""
    return security.sanitize_input(input_str, input_type)

def load_logo(logo_path: str) -> str:
    """Public interface for logo loading"""
    return resource_loader.load_logo(logo_path)

def rotate_proxy() -> Optional[str]:
    """Public interface for proxy rotation"""
    return proxy_manager.get_proxy()

def get_random_user_agent() -> str:
    """Get a random user agent"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    ]
    return random.choice(user_agents)

if __name__ == "__main__":
    # Test the helpers
    print("Testing helpers...")
    print("Sanitized input:", sanitize_input("test<input>"))
    print("Random UA:", get_random_user_agent())
    print("Current proxy:", rotate_proxy())