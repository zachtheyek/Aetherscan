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

from config import get_config

logger = logging.getLogger(__name__)


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

        # Sample random noise from a standard normal N(0, 1) with same shape as z_mean
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

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
    """Build encoder network"""

    encoder_inputs = keras.Input(shape=(16, 512, 1), name="encoder_input")

    # Convolutional layers with regularization
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

    # Flatten and dense layers
    x = layers.Flatten()(x)

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
    z_mean = layers.Dense(
        latent_dim,
        name="z_mean",
        kernel_initializer=GlorotNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    z_log_var = layers.Dense(
        latent_dim,
        name="z_log_var",
        kernel_initializer=GlorotNormal(),
        bias_initializer=Constant(
            -3.0
        ),  # Use negative bias initialization to tighten initial posterior
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    # Sampling
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder


def build_decoder(
    latent_dim: int = 8, dense_size: int = 512, kernel_size: tuple[int, int] = (3, 3)
) -> keras.Model:
    """Build decoder network - exact mirror of encoder"""

    latent_inputs = keras.Input(shape=(latent_dim,), name="decoder_input")

    # Dense layers with regularization
    x = layers.Dense(
        dense_size,
        activation="relu",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(latent_inputs)

    # Reshape to start transposed convolutions
    x = layers.Dense(
        1 * 32 * 256,
        activation="relu",
        kernel_initializer=HeNormal(),
        bias_initializer=Zeros(),
        activity_regularizer=l1(0.001),
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01),
    )(x)

    x = layers.Reshape((1, 32, 256))(x)

    # Transposed convolutions (exact reverse of encoder)
    x = layers.Conv2DTranspose(
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

    x = layers.Conv2DTranspose(
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

    x = layers.Conv2DTranspose(
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

    # Output layer with sigmoid activation
    decoder_outputs = layers.Conv2DTranspose(
        1,
        kernel_size,
        activation="sigmoid",
        padding="same",
        kernel_initializer=GlorotNormal(),
        bias_initializer=Zeros(),
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


def create_beta_vae_model():
    """Create and compile Beta-VAE model with author's exact settings"""

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
