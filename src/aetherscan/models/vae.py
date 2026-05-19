# TODO: refactor to expose public APIs for creating & destroying BetaVAE instances
# TODO: replace hard-coded values with config values
"""
Beta-VAE model implementation for Aetherscan Pipeline
Uses custom clustering loss components to implicitly differentiate SETI signals from RFI in the
model's learned latent space
"""

from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant, GlorotNormal, HeNormal, Zeros
from tensorflow.keras.regularizers import l1, l2

from aetherscan.config import get_config

logger = logging.getLogger(__name__)


# Use keras.utils.* rather than keras.saving.* — the latter is the canonical
# Keras 3 path, but `from tensorflow import keras` in TF 2.17 + NGC 25.02
# resolves to the tf-keras compat shim (`keras._tf_keras.keras`), which
# doesn't re-export the `saving` submodule. keras.utils.register_keras_serializable
# is the back-compat alias that exists in both tf.keras lineages and standalone
# Keras 3 — pick the path that works everywhere.
@keras.utils.register_keras_serializable(package="aetherscan")
class Sampling(layers.Layer):
    """
    Sampling layer for Beta-VAE using reparameterization trick

    Since sampling is a non-differentiable operation (can't backprop through random sampling)
    But we need to sample from the Beta-VAE's learned distribution to produce the latent vector (z)
    We isolate the randomness (epsilon) to be independent of the learned params (z_mean, z_log_var)
    Such that gradients can flow through without issue
    """

    def call(self, inputs):
        # Get the learned mean & log-varience of the latent distribution
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        # Sample random noise from a standard normal N(0, 1) with same shape as z_mean.
        # tf.random.normal is the stable canonical API; tf.keras.backend.random_normal was
        # removed/deprecated in Keras 3 (shipped in TF 2.16+) and had inconsistent graph/seed semantics.
        epsilon = tf.random.normal(shape=(batch, dim))

        # Compute latent vector using reparameterization
        # Equivalent to sampling from N(z_mean, exp(z_log_var))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        return z


class BetaVAE(keras.Model):
    """
    Beta-VAE model with custom loss functions for SETI
    """

    def __init__(self, encoder, decoder, alpha=10.0, beta=1.5, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Hyperparameters
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs, training=None):
        """
        Forward pass through the Beta-VAE
        """
        batch_size = tf.shape(inputs)[0]

        # Reshape inputs for encoder
        encoder_input = tf.reshape(inputs, (batch_size * 6, 16, 512, 1))

        # Encode: observations -> latents
        z_mean, z_log_var, z = self.encoder(encoder_input, training=training)

        # Decode: latents -> observations
        reconstruction = self.decoder(z, training=training)

        # Reshape outputs back to cadence format
        reconstruction = tf.reshape(reconstruction, (batch_size, 6, 16, 512))

        return reconstruction, z_mean, z_log_var, z

    @tf.function
    def loss_same(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Distance between ON-ON or OFF-OFF (to be minimized)
        """
        return tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=1))

    @tf.function
    def loss_diff(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """
        Distance between ON-OFF (to be maximized)
        """
        return tf.reduce_mean(1.0 / (tf.reduce_sum(tf.square(a - b), axis=1) + 1e-8))

    @tf.function
    def compute_clustering_loss_true(self, true_data: tf.Tensor) -> tf.Tensor:
        """
        Clustering loss for true signals
        """
        batch_size = tf.shape(true_data)[0]

        # Process all observations at once for efficiency
        all_obs = tf.reshape(true_data, (batch_size * 6, 16, 512, 1))
        _, _, all_latents = self.encoder(all_obs, training=True)

        # Reshape back to (batch, 6, latent_dim)
        latent_dim = tf.shape(all_latents)[1]
        latents_reshaped = tf.reshape(all_latents, (batch_size, 6, latent_dim))

        # Extract ON and OFF observations
        a1 = latents_reshaped[:, 0, :]  # ON
        b = latents_reshaped[:, 1, :]  # OFF
        a2 = latents_reshaped[:, 2, :]  # ON
        c = latents_reshaped[:, 3, :]  # OFF
        a3 = latents_reshaped[:, 4, :]  # ON
        d = latents_reshaped[:, 5, :]  # OFF

        # Difference terms (ON-OFF should be maximized, so use loss_diff)
        difference = 0.0
        difference += self.loss_diff(a1, b)
        difference += self.loss_diff(a1, c)
        difference += self.loss_diff(a1, d)
        difference += self.loss_diff(a2, b)
        difference += self.loss_diff(a2, c)
        difference += self.loss_diff(a2, d)
        difference += self.loss_diff(a3, b)
        difference += self.loss_diff(a3, c)
        difference += self.loss_diff(a3, d)

        # Same terms (ON-ON and OFF-OFF should be minimized, so use loss_same)
        same = 0.0
        same += self.loss_same(a1, a2)
        same += self.loss_same(a1, a3)
        same += self.loss_same(a2, a1)
        same += self.loss_same(a2, a3)
        same += self.loss_same(a3, a1)
        same += self.loss_same(a3, a2)
        same += self.loss_same(b, c)
        same += self.loss_same(b, d)
        same += self.loss_same(c, b)
        same += self.loss_same(c, d)
        same += self.loss_same(d, b)
        same += self.loss_same(d, c)

        similarity = same + difference
        return similarity

    @tf.function
    def compute_clustering_loss_false(self, false_data: tf.Tensor) -> tf.Tensor:
        """
        Clustering loss for false signals
        """
        batch_size = tf.shape(false_data)[0]

        # Process all observations at once for efficiency
        all_obs = tf.reshape(false_data, (batch_size * 6, 16, 512, 1))
        _, _, all_latents = self.encoder(all_obs, training=True)

        # Reshape back to (batch, 6, latent_dim)
        latent_dim = tf.shape(all_latents)[1]
        latents_reshaped = tf.reshape(all_latents, (batch_size, 6, latent_dim))

        # Extract OFF observations
        a1 = latents_reshaped[:, 0, :]  # OFF
        b = latents_reshaped[:, 1, :]  # OFF
        a2 = latents_reshaped[:, 2, :]  # OFF
        c = latents_reshaped[:, 3, :]  # OFF
        a3 = latents_reshaped[:, 4, :]  # OFF
        d = latents_reshaped[:, 5, :]  # OFF

        # For RFI/false signals, all observations should look similar
        # So we minimize distances between all pairs
        difference = 0.0
        difference += self.loss_same(a1, b)
        difference += self.loss_same(a1, c)
        difference += self.loss_same(a1, d)
        difference += self.loss_same(a2, b)
        difference += self.loss_same(a2, c)
        difference += self.loss_same(a2, d)
        difference += self.loss_same(a3, b)
        difference += self.loss_same(a3, c)
        difference += self.loss_same(a3, d)

        same = 0.0
        same += self.loss_same(a1, a2)
        same += self.loss_same(a1, a3)
        same += self.loss_same(a2, a1)
        same += self.loss_same(a2, a3)
        same += self.loss_same(a3, a1)
        same += self.loss_same(a3, a2)
        same += self.loss_same(b, c)
        same += self.loss_same(b, d)
        same += self.loss_same(c, b)
        same += self.loss_same(c, d)
        same += self.loss_same(d, b)
        same += self.loss_same(d, c)

        similarity = same + difference
        return similarity

    @tf.function
    def compute_total_loss(self, main_data, true_data, false_data, target_data, training=True):
        """
        Perform forward pass and compute losses
        """
        # Perform forward pass through Beta-VAE
        reconstruction, z_mean, z_log_var, z = self.call(main_data, training=training)

        # Ensure reconstruction shape matches target for loss computation
        reconstruction = tf.reshape(reconstruction, tf.shape(target_data))

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(
                    target_data,
                    reconstruction,
                    from_logits=False,  # Use from_logits=False for stability since decoder's final activation is sigmoid (reconstruction is bounded [0,1])
                ),
                axis=(1, 2),
            )
        )

        # Compute KL loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        # Compute clustering losses
        false_loss = self.compute_clustering_loss_false(false_data)
        true_loss = self.compute_clustering_loss_true(true_data)

        # Compute total loss
        total_loss = (
            reconstruction_loss + self.beta * kl_loss + self.alpha * (true_loss + false_loss)
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "true_loss": true_loss,
            "false_loss": false_loss,
        }


def build_encoder(
    latent_dim: int = 8, dense_size: int = 512, kernel_size: tuple[int, int] = (3, 3)
) -> keras.Model:
    """Build encoder network for Beta-VAE.

    The encoder compresses input spectrograms into a lower-dimensional latent space.
    It uses a series of convolutional layers followed by dense layers to learn
    the parameters (mean & log-variance) of the Gaussian posterior over the latent space.

    Architecture Overview:
    ---------------------
    Input: (16, 512, 1) - spectrogram with 16 time bins, 512 frequency bins, 1 polarization channel

    Convolutional Layers (9 total):
        - 4 downsampling layers (stride=2) reduce spatial dims: 16→8→4→2→1, 512→256→128→64→32
        - 5 feature extraction layers (stride=1) maintain spatial dims
        - Filter progression: 1→16→16→32→32→32→64→64→128→256

    Dense Layers:
        - Flatten: (1, 32, 256) → (8192,)  [1*32*256 = 8192]
        - Dense: (8192,) → (dense_size,)   [default 512]
        - z_mean: (dense_size,) → (latent_dim,)  [default 8]
        - z_log_var: (dense_size,) → (latent_dim,)  [default 8]

    Output: z_mean, z_log_var, z (sampled latent vector)

    Layer Details:
    -------------
    | Enc Layer | Filters | Stride | Input Shape      | Output Shape     |
    |-----------|---------|--------|------------------|------------------|
    | 1         | 16      | 2      | (16, 512, 1)     | (8, 256, 16)     |
    | 2         | 16      | 1      | (8, 256, 16)     | (8, 256, 16)     |
    | 3         | 32      | 2      | (8, 256, 16)     | (4, 128, 32)     |
    | 4         | 32      | 1      | (4, 128, 32)     | (4, 128, 32)     |
    | 5         | 32      | 1      | (4, 128, 32)     | (4, 128, 32)     |
    | 6         | 64      | 2      | (4, 128, 32)     | (2, 64, 64)      |
    | 7         | 64      | 1      | (2, 64, 64)      | (2, 64, 64)      |
    | 8         | 128     | 1      | (2, 64, 64)      | (2, 64, 128)     |
    | 9         | 256     | 2      | (2, 64, 128)     | (1, 32, 256)     |

    Regularization (all layers):
        - kernel_initializer: HeNormal (conv/dense) or GlorotNormal (latent)
        - bias_initializer: Zeros (except z_log_var uses -3.0 for tighter initial posterior)
        - activity_regularizer: L1(0.001) - encourages sparse activations
        - kernel_regularizer: L2(0.01) - prevents large weights
        - bias_regularizer: L2(0.01) - prevents large biases

    Args:
        latent_dim: Dimensionality of the latent space (default: 8)
        dense_size: Size of the dense layer before latent projection (default: 512)
        kernel_size: Convolutional kernel size (default: (3, 3))

    Returns:
        keras.Model: Encoder model with outputs [z_mean, z_log_var, z]

    Note:
        The decoder (build_decoder) must be an exact mirror of this architecture
        for proper VAE symmetry. See build_decoder docstring for details.
    """
    # Input shape: (batch, 16, 512, 1) - "grayscale" spectrogram
    encoder_inputs = keras.Input(shape=(16, 512, 1), name="encoder_input")

    # Convolutional layers
    # Layer 1: (16, 512, 1) → (8, 256, 16) - stride 2 downsamples spatial dims by 2x
    x = layers.Conv2D(
        16,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(encoder_inputs)

    # Layer 2: (8, 256, 16) → (8, 256, 16) - stride 1 maintains spatial dims
    x = layers.Conv2D(
        16,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 3: (8, 256, 16) → (4, 128, 32) - stride 2 downsamples
    x = layers.Conv2D(
        32,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 4: (4, 128, 32) → (4, 128, 32) - stride 1 maintains
    x = layers.Conv2D(
        32,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 5: (4, 128, 32) → (4, 128, 32) - stride 1 maintains
    x = layers.Conv2D(
        32,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 6: (4, 128, 32) → (2, 64, 64) - stride 2 downsamples
    x = layers.Conv2D(
        64,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 7: (2, 64, 64) → (2, 64, 64) - stride 1 maintains
    x = layers.Conv2D(
        64,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 8: (2, 64, 64) → (2, 64, 128) - stride 1, increase filters
    x = layers.Conv2D(
        128,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 9: (2, 64, 128) → (1, 32, 256) - stride 2 final downsample
    x = layers.Conv2D(
        256,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Dense layers
    # Flatten: (1, 32, 256) → (8192,)
    # The 32 comes from: 512 / 2^4 = 32 (4 stride-2 layers)
    x = layers.Flatten()(x)

    # Dense: (8192,) → (dense_size,) - compress to intermediate representation
    x = layers.Dense(
        dense_size,
        activation="relu",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Latent space
    # z_mean: (dense_size,) → (latent_dim,) - mean of latent distribution
    z_mean = layers.Dense(
        latent_dim,
        name="z_mean",
        kernel_initializer=GlorotNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # z_log_var: (dense_size,) → (latent_dim,) - log-variance of latent distribution
    z_log_var = layers.Dense(
        latent_dim,
        name="z_log_var",
        kernel_initializer=GlorotNormal(),
        bias_initializer=Constant(
            -3.0  # Negative bias initialization tightens initial posterior around prior
        ),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Sampling: sample z from N(z_mean, exp(z_log_var)) using reparameterization trick
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder


def build_decoder(
    latent_dim: int = 8, dense_size: int = 512, kernel_size: tuple[int, int] = (3, 3)
) -> keras.Model:
    """Build decoder network for Beta-VAE - exact mirror of encoder.

    The decoder reconstructs spectrograms from latent vectors. It is architecturally
    symmetric to the encoder: each encoder layer has a corresponding decoder layer
    that reverses the transformation.

    Architecture Overview:
    ---------------------
    Input: (latent_dim,) - sampled latent vector z (default: 8)

    Dense Layers (mirrors encoder's dense → flatten path in reverse):
        - Dense: (latent_dim,) → (dense_size,)   [mirrors z_mean/z_log_var]
        - Dense: (dense_size,) → (8192,)         [mirrors Dense before Flatten]
        - Reshape: (8192,) → (1, 32, 256)        [mirrors Flatten]

    Conv Transpose Layers (9 total, mirrors encoder's 9 conv layers in reverse):
        - 4 upsampling layers (stride=2) expand spatial dims: 1→2→4→8→16, 32→64→128→256→512
        - 5 feature transformation layers (stride=1) maintain spatial dims
        - Filter progression: 256→128→64→64→32→32→32→16→16→1

    Output: (16, 512, 1) - reconstructed spectrogram (matches encoder input)

    Symmetry Principle:
    ------------------
    For each encoder layer i, decoder layer (9-i+1) reverses it:
        - Decoder layer outputs the same number of filters as encoder layer's INPUT channels
        - Decoder layer uses the same stride as encoder layer (stride-2 upsamples instead of downsamples)
        - Shapes match: decoder layer output shape == encoder layer input shape

    Layer Details:
    -------------
    | Dec Layer | Filters | Stride | Input Shape      | Output Shape     | Mirrors Enc |
    |-----------|---------|--------|------------------|------------------|-------------|
    | 1         | 128     | 2      | (1, 32, 256)     | (2, 64, 128)     | Enc 9       |
    | 2         | 64      | 1      | (2, 64, 128)     | (2, 64, 64)      | Enc 8       |
    | 3         | 64      | 1      | (2, 64, 64)      | (2, 64, 64)      | Enc 7       |
    | 4         | 32      | 2      | (2, 64, 64)      | (4, 128, 32)     | Enc 6       |
    | 5         | 32      | 1      | (4, 128, 32)     | (4, 128, 32)     | Enc 5       |
    | 6         | 32      | 1      | (4, 128, 32)     | (4, 128, 32)     | Enc 4       |
    | 7         | 16      | 2      | (4, 128, 32)     | (8, 256, 16)     | Enc 3       |
    | 8         | 16      | 1      | (8, 256, 16)     | (8, 256, 16)     | Enc 2       |
    | 9         | 1       | 2      | (8, 256, 16)     | (16, 512, 1)     | Enc 1       |

    Regularization (all layers including output):
        - kernel_initializer: HeNormal (hidden layers) or GlorotNormal (output layer)
        - bias_initializer: Zeros
        - activity_regularizer: L1(0.001) - encourages sparse activations
        - kernel_regularizer: L2(0.01) - prevents large weights
        - bias_regularizer: L2(0.01) - prevents large biases

    Output Layer Notes:
        - Uses sigmoid activation (not relu) to bound output to [0, 1] for BCE loss
        - Uses GlorotNormal initialization (matches encoder's latent layer style)
        - Includes full regularization for symmetry with encoder's first conv layer

    Args:
        latent_dim: Dimensionality of the latent space (default: 8)
        dense_size: Size of the dense layer after latent input (default: 512)
        kernel_size: Convolutional kernel size (default: (3, 3))

    Returns:
        keras.Model: Decoder model that outputs reconstructed spectrograms

    Note:
        This architecture is the exact mirror of build_encoder. Any changes to the
        encoder's layer structure must be reflected here to maintain symmetry.
    """
    # Input shape: (batch, latent_dim) - sampled latent vector z
    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")

    # Dense layers: these mirror the encoder's latent → dense → flatten path in reverse
    # Dense: (latent_dim,) → (dense_size,) - mirrors encoder's z_mean/z_log_var layers
    x = layers.Dense(
        dense_size,
        activation="relu",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(latent_inputs)

    # Dense: (dense_size,) → (8192,) - mirrors encoder's Dense(dense_size) layer
    # The 8192 = 1 * 32 * 256 comes from encoder's final conv output shape
    # Where 32 = 512 / 2^4 (4 stride-2 layers reduce width from 512 to 32)
    x = layers.Dense(
        1 * 32 * 256,
        activation="relu",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Reshape: (8192,) → (1, 32, 256) - mirrors encoder's Flatten layer
    x = layers.Reshape((1, 32, 256))(x)

    # Convolutional transpose layers
    # These mirror the encoder's conv layers in reverse order
    # Each layer outputs filters matching the INPUT channels of the corresponding encoder layer
    # Layer 1: (1, 32, 256) → (2, 64, 128) - mirrors encoder layer 9 (256 filters, s=2)
    # Output 128 filters to match encoder layer 9's input channels
    x = layers.Conv2DTranspose(
        128,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 2: (2, 64, 128) → (2, 64, 64) - mirrors encoder layer 8 (128 filters, s=1)
    # Output 64 filters to match encoder layer 8's input channels
    x = layers.Conv2DTranspose(
        64,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 3: (2, 64, 64) → (2, 64, 64) - mirrors encoder layer 7 (64 filters, s=1)
    # Output 64 filters to match encoder layer 7's input channels
    x = layers.Conv2DTranspose(
        64,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 4: (2, 64, 64) → (4, 128, 32) - mirrors encoder layer 6 (64 filters, s=2)
    # Output 32 filters to match encoder layer 6's input channels
    x = layers.Conv2DTranspose(
        32,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 5: (4, 128, 32) → (4, 128, 32) - mirrors encoder layer 5 (32 filters, s=1)
    # Output 32 filters to match encoder layer 5's input channels
    x = layers.Conv2DTranspose(
        32,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 6: (4, 128, 32) → (4, 128, 32) - mirrors encoder layer 4 (32 filters, s=1)
    # Output 32 filters to match encoder layer 4's input channels
    x = layers.Conv2DTranspose(
        32,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 7: (4, 128, 32) → (8, 256, 16) - mirrors encoder layer 3 (32 filters, s=2)
    # Output 16 filters to match encoder layer 3's input channels
    x = layers.Conv2DTranspose(
        16,
        kernel_size,
        activation="relu",
        strides=2,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Layer 8: (8, 256, 16) → (8, 256, 16) - mirrors encoder layer 2 (16 filters, s=1)
    # Output 16 filters to match encoder layer 2's input channels
    x = layers.Conv2DTranspose(
        16,
        kernel_size,
        activation="relu",
        strides=1,
        padding="same",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Output layer
    # Layer 9: (8, 256, 16) → (16, 512, 1) - mirrors encoder layer 1 (16 filters, s=2)
    # Output 1 filter to match encoder input channels ("grayscale" spectrogram)
    # Uses sigmoid activation to bound output to [0, 1] for binary cross-entropy loss
    # Uses GlorotNormal (like encoder's latent layers) as this is the "boundary" layer
    # Includes full regularization for symmetry with encoder's first conv layer
    decoder_outputs = layers.Conv2DTranspose(
        1,
        kernel_size,
        activation="sigmoid",
        strides=2,
        padding="same",
        kernel_initializer=GlorotNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


def create_beta_vae_model():
    """Create and compile Beta-VAE model"""

    logger.info("Creating Beta-VAE model...")

    config = get_config()
    if config is None:
        raise ValueError("get_config() returned None")

    encoder = build_encoder(
        latent_dim=config.beta_vae.latent_dim,
        dense_size=config.beta_vae.dense_layer_size,
        kernel_size=config.beta_vae.kernel_size,
    )

    decoder = build_decoder(
        latent_dim=config.beta_vae.latent_dim,
        dense_size=config.beta_vae.dense_layer_size,
        kernel_size=config.beta_vae.kernel_size,
    )

    beta_vae = BetaVAE(
        encoder,
        decoder,
        alpha=config.beta_vae.alpha,
        beta=config.beta_vae.beta,
    )

    beta_vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.training.base_learning_rate)
    )

    logger.info(
        f"Created Beta-VAE model: latent_dim={config.beta_vae.latent_dim}, "
        f"beta={config.beta_vae.beta}, alpha={config.beta_vae.alpha}"
    )

    encoder.summary(print_fn=logger.info)
    decoder.summary(print_fn=logger.info)

    return beta_vae
