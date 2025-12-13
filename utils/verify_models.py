"""
Verify trained models are saved & can be loaded properly
"""

import json

import joblib
import tensorflow as tf

TAG = "final_v1"

# Load VAE encoder
encoder = tf.keras.models.load_model(
    f"/datax/scratch/zachy/models/aetherscan/vae_encoder_{TAG}.keras"
)
print(f"Encoder loaded: {encoder.summary()}")

# Load VAE decoder
decoder = tf.keras.models.load_model(
    f"/datax/scratch/zachy/models/aetherscan/vae_decoder_{TAG}.keras"
)
print(f"Decoder loaded: {decoder.summary()}")

# Load Random Forest
rf = joblib.load(f"/datax/scratch/zachy/models/aetherscan/random_forest_{TAG}.joblib")
print(f"Random Forest loaded: {rf.n_estimators} trees")

# Load config
with open(f"/datax/scratch/zachy/models/aetherscan/config_{TAG}.json") as f:
    config = json.load(f)
print(f"Config loaded: {config.keys()}")
