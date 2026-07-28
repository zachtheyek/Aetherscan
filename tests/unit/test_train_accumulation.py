"""Unit tests for the graph-side accumulated train step (#276 follow-up): the traced
K-micro-batch optimizer step must apply the same update as the reference
accumulate-in-Python implementation it replaced (per-replica mean losses summed over K,
averaged, clipped by global L2 norm at 1.0, applied once), guard NaN/Inf gradients without
touching the weights, and average the returned losses over the K micro-batches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from aetherscan.train import _GRADIENT_CLIP_NORM, TrainingPipeline

pytestmark = pytest.mark.slow  # builds TF graphs on CPU

_LATENT_DIM = 3


class _StubModel(tf.Module):
    """Minimal stand-in for the beta-VAE: a two-weight linear model whose loss dict has the
    same keys compute_total_loss returns. Gradients depend on the batch contents, so
    accumulation across distinct micro-batches is actually exercised."""

    def __init__(self, learning_rate=0.25):
        super().__init__()
        self.w = tf.Variable([1.0, -2.0], dtype=tf.float32, name="w")
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def compute_total_loss(self, main_data, true_data, false_data, y, training=False):
        del y, training
        signal = (
            tf.reduce_mean(main_data) + 2.0 * tf.reduce_mean(true_data) - tf.reduce_mean(false_data)
        )
        total = tf.reduce_sum(self.w * tf.stack([signal, tf.square(signal)]))
        return {
            "total_loss": total,
            "reconstruction_loss": 0.5 * total,
            "kl_loss": 0.25 * total,
            "true_loss": 0.15 * total,
            "false_loss": 0.1 * total,
            "reg_loss": 0.05 * total,
            "kl_per_dim": tf.fill([_LATENT_DIM], 0.01 * total),
        }


def _make_pipeline(model):
    """TrainingPipeline.__new__ + just the attributes the accumulation surface touches
    (the same stub pattern test_inference.py uses for InferencePipeline)."""
    tp = TrainingPipeline.__new__(TrainingPipeline)
    tp.strategy = tf.distribute.get_strategy()
    tp.vae = model
    tp.config = SimpleNamespace(beta_vae=SimpleNamespace(latent_dim=_LATENT_DIM))
    tp._grad_accumulators = None
    tp._train_loss_accumulators = None
    tp._val_loss_accumulator = None
    tp._unconnected_grad_indices = set()
    tp._accumulated_train_step_fns = {}
    tp._val_loop_fns = {}
    return tp


def _micro_batches(k, seed=0, poison_step=None, scale=1.0):
    """K distinct ((main, true, false), y) micro-batches; poison_step injects an inf;
    scale magnifies the values (and therefore the gradient norm past the clip threshold)."""
    rng = np.random.default_rng(seed)
    batches = []
    for i in range(k):
        arrays = [(scale * rng.normal(size=(4, 2, 3, 4))).astype(np.float32) for _ in range(3)]
        if i == poison_step:
            arrays[0][0, 0, 0, 0] = np.inf
        batches.append(((arrays[0], arrays[1], arrays[2]), arrays[0]))
    return batches


def _dataset_from(batches):
    def gen():
        yield from batches

    spec = tf.TensorSpec(shape=(4, 2, 3, 4), dtype=tf.float32)
    return tf.data.Dataset.from_generator(gen, output_signature=((spec, spec, spec), spec)).repeat()


def _reference_step(model, batches):
    """The replaced implementation, eagerly: per-micro-batch gradients summed, averaged over
    K, clipped by global norm, applied once. Returns (losses dict, global_norm)."""
    accumulated = None
    loss_sums = None
    for (main, true, false), y in batches:
        with tf.GradientTape() as tape:
            losses = model.compute_total_loss(main, true, false, y, training=True)
        grads = tape.gradient(losses["total_loss"], model.trainable_variables)
        accumulated = (
            grads
            if accumulated is None
            else [a + g for a, g in zip(accumulated, grads, strict=False)]
        )
        if loss_sums is None:
            loss_sums = {k: tf.identity(v) for k, v in losses.items()}
        else:
            loss_sums = {k: loss_sums[k] + v for k, v in losses.items()}
    averaged = [g / len(batches) for g in accumulated]
    clipped, global_norm = tf.clip_by_global_norm(averaged, _GRADIENT_CLIP_NORM)
    model.optimizer.apply_gradients(zip(clipped, model.trainable_variables, strict=False))
    return {k: v / len(batches) for k, v in loss_sums.items()}, global_norm


class TestAccumulatedTrainStep:
    # scale=25 pushes the averaged gradient's global norm past _GRADIENT_CLIP_NORM so the
    # clip's rescale branch is actually exercised, not just traced (at scale=1 the norms
    # for these seeds sit below 1.0 and clipping is a mathematical no-op)
    @pytest.mark.parametrize(("k", "scale"), [(1, 1.0), (3, 1.0), (3, 25.0)])
    def test_matches_reference_accumulation(self, k, scale):
        batches = _micro_batches(k, seed=7, scale=scale)
        graph_model = _StubModel()
        eager_model = _StubModel()
        eager_model.w.assign(graph_model.w)

        tp = _make_pipeline(graph_model)
        tp._ensure_accumulation_state()
        step_fn = tp._get_accumulated_train_step(k)
        iterator = iter(_dataset_from(batches))
        losses, global_norm, applied = step_fn(iterator)

        ref_losses, ref_norm = _reference_step(eager_model, batches)

        assert bool(applied.numpy())
        np.testing.assert_allclose(graph_model.w.numpy(), eager_model.w.numpy(), rtol=1e-5)
        np.testing.assert_allclose(global_norm.numpy(), ref_norm.numpy(), rtol=1e-5)
        for key, ref in ref_losses.items():
            name = "total" if key == "total_loss" else key.removesuffix("_loss")
            np.testing.assert_allclose(losses[name].numpy(), ref.numpy(), rtol=1e-5, err_msg=key)

    def test_second_step_reaccumulates_from_zero(self):
        """The accumulators must reset between optimizer steps: two identical steps from the
        same weights-with-reset must equal one step's delta applied twice, not a growing sum."""
        batches = _micro_batches(2, seed=3)
        model = _StubModel()
        tp = _make_pipeline(model)
        tp._ensure_accumulation_state()
        step_fn = tp._get_accumulated_train_step(2)

        _, norm_first, _ = step_fn(iter(_dataset_from(batches)))
        w_after_first = model.w.numpy().copy()
        _, norm_second, _ = step_fn(iter(_dataset_from(batches)))

        # Same batches, but weights moved => a different (finite) norm; a leaking
        # accumulator would instead double the gradient magnitude on step two
        assert np.isfinite(norm_second.numpy())
        ref = _StubModel()
        ref.w.assign(w_after_first)
        _, ref_norm = _reference_step(ref, batches)
        np.testing.assert_allclose(norm_second.numpy(), ref_norm.numpy(), rtol=1e-5)

    def test_nan_guard_skips_apply_and_reports(self):
        batches = _micro_batches(2, seed=5, poison_step=1)
        model = _StubModel()
        w_before = model.w.numpy().copy()
        tp = _make_pipeline(model)
        tp._ensure_accumulation_state()
        step_fn = tp._get_accumulated_train_step(2)

        _, _, applied = step_fn(iter(_dataset_from(batches)))

        assert not bool(applied.numpy())
        np.testing.assert_array_equal(model.w.numpy(), w_before)

    def test_val_loop_averages_losses(self):
        batches = _micro_batches(3, seed=9)
        model = _StubModel()
        tp = _make_pipeline(model)
        tp._ensure_accumulation_state()
        val_loop = tp._get_val_loop(3)
        totals = val_loop(iter(_dataset_from(batches))).numpy()

        expected = np.zeros(5)
        for (main, true, false), y in batches:
            losses = model.compute_total_loss(main, true, false, y, training=False)
            expected += np.array(
                [
                    losses["total_loss"].numpy(),
                    losses["reconstruction_loss"].numpy(),
                    losses["kl_loss"].numpy(),
                    losses["true_loss"].numpy(),
                    losses["false_loss"].numpy(),
                ]
            )
        np.testing.assert_allclose(totals, expected / 3, rtol=1e-5)
