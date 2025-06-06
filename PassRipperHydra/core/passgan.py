# core/passgan.py
import tensorflow as tf
import numpy as np
import random
import logging
import os
import sys
import re
import psutil
from utils.logger import setup_logging
from utils.helpers import sanitize_input

# Adjust path to your cloned PassGAN repo
sys.path.append("path/to/PassGAN")
from sample import sample_from_model

logger = setup_logging("output/logs/passripperhydra.log")

# Default character vocabulary (matches PassGAN's training setup)
DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
CHAR2IDX = {c: i for i, c in enumerate(DEFAULT_CHARSET)}
IDX2CHAR = {i: c for i, c in enumerate(DEFAULT_CHARSET)}
MAX_LEN = 16
MIN_LEN = 4

class PassGANModel:
    def __init__(self, weights_path, checkpoint_path="fine_tuned/checkpoints/checkpoint_200000.ckpt", charset=None, min_len=4, max_len=16):
        """
        Initialize the PassGAN model.
        Args:
            weights_path (str): Path to save/load weights.
            checkpoint_path (str): Path to pretrained/fine-tuned checkpoint.
            charset (str, optional): Custom character set for generation.
            min_len (int): Minimum password length.
            max_len (int): Maximum password length.
        """
        self.weights_path = weights_path
        self.checkpoint_path = checkpoint_path
        self.input_dir = os.path.dirname(checkpoint_path)
        self.charset = charset if charset else DEFAULT_CHARSET
        self.char2idx = {c: i for i, c in enumerate(self.charset)}
        self.idx2char = {i: c for i, c in enumerate(self.charset)}
        self.min_len = max(min_len, MIN_LEN)
        self.max_len = min(max_len, MAX_LEN)
        self.seq_length = 10  # Pretrained model default
        self.batch_size = self._dynamic_batch_size()
        self.sess = None
        self.load_model()

    def _dynamic_batch_size(self):
        """
        Dynamically adjust batch size based on available memory.
        Returns:
            int: Batch size.
        """
        try:
            available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB
            if available_memory > 8000:
                return 2048
            elif available_memory > 4000:
                return 1024
            else:
                return 512
        except Exception as e:
            logger.warning(f"Could not determine available memory: {str(e)}. Using default batch size.")
            return 512

    def load_model(self):
        """
        Initialize TensorFlow session and load the pretrained/fine-tuned model.
        """
        try:
            logger.info(f"Loading PassGAN model from {self.checkpoint_path}...")
            # Configure GPU memory growth to prevent OOM errors
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # Verify checkpoint exists
            if not os.path.exists(self.checkpoint_path + ".meta"):
                raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load PassGAN model: {str(e)}")
            if self.sess:
                self.sess.close()
            raise

    def _filter_password(self, password):
        """
        Filter passwords based on complexity and constraints.
        Args:
            password (str): Generated password.
        Returns:
            bool: True if password meets criteria, False otherwise.
        """
        if not (self.min_len <= len(password) <= self.max_len):
            return False
        # Ensure password has at least one digit and one special character
        has_digit = any(c in string.digits for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        return has_digit and has_special

    def generate(self, metadata, num_passwords, min_complexity=True):
        """
        Generate passwords using the pretrained PassGAN model.
        Args:
            metadata (list): User-provided metadata (e.g., usernames, keywords).
            num_passwords (int): Number of passwords to generate.
            min_complexity (bool): Enforce minimum complexity (digit + special char).
        Returns:
            list: Generated passwords.
        """
        try:
            logger.info(f"Generating {num_passwords} passwords with PassGAN...")
            passwords = set()
            
            # Sanitize metadata
            metadata = [sanitize_input(m) for m in metadata if isinstance(m, str) and m]

            # Adjust sequence length if max_len > pretrained seq_length
            gen_seq_length = min(self.max_len, self.seq_length)

            # Generate passwords in batches
            while len(passwords) < num_passwords:
                try:
                    gen_passwords = sample_from_model(
                        sess=self.sess,
                        input_dir=self.input_dir,
                        checkpoint=self.checkpoint_path,
                        batch_size=self.batch_size,
                        num_samples=self.batch_size,
                        seq_length=gen_seq_length
                    )
                    
                    # Convert indices to characters
                    for gen_pwd in gen_passwords:
                        pwd = "".join(self.idx2char.get(idx, '') for idx in gen_pwd if idx in self.idx2char)
                        pwd = pwd.strip(self.charset[0])  # Remove padding
                        if min_complexity and not self._filter_password(pwd):
                            continue
                        if pwd and pwd not in passwords:
                            passwords.add(pwd)
                            if len(passwords) >= num_passwords:
                                break
                except Exception as e:
                    logger.warning(f"Batch generation failed: {str(e)}. Retrying...")
                    time.sleep(1)  # Brief pause to avoid hammering

            # Enhance with metadata
            if metadata:
                enhanced_passwords = set(passwords)
                for pwd in list(passwords):
                    for meta in metadata:
                        # Basic combinations
                        enhanced_passwords.add(f"{pwd}{meta}")
                        enhanced_passwords.add(f"{meta}{pwd}")
                        # Add year-based variations
                        year = random.randint(1990, 2025)
                        enhanced_passwords.add(f"{meta}{year}")
                        enhanced_passwords.add(f"{pwd}{year}")
                        # Add leetspeak variations
                        meta_leet = meta.lower().replace("a", "@").replace("e", "3").replace("i", "1")
                        enhanced_passwords.add(f"{pwd}{meta_leet}")
                passwords = enhanced_passwords

            # Final filtering
            passwords = [pwd for pwd in passwords if self.min_len <= len(pwd) <= self.max_len][:num_passwords]
            logger.info(f"Generated {len(passwords)} unique passwords.")
            return passwords
        except Exception as e:
            logger.error(f"PassGAN generation failed: {str(e)}")
            raise
        finally:
            if self.sess:
                self.sess.close()