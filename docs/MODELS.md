# Models

This document specifies the two models: the **Beta-VAE** feature extractor
([`src/aetherscan/models/vae.py`](../src/aetherscan/models/vae.py)) — architecture, the
composite loss including the clustering terms, and the design rationale — and the
**Random Forest** cadence classifier
([`src/aetherscan/models/random_forest.py`](../src/aetherscan/models/random_forest.py)) —
feature layout and threshold/confidence semantics — plus the meaning of the persisted
evaluation/SHAP artifacts. How the models are *trained* is
[`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)'s territory; how they are *served* is
[`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md)'s.

## Design overview

The division of labor:

- The **Beta-VAE encoder** turns each observation spectrogram `(16, 512, 1)` into an
  8-dimensional latent describing its morphology. It never sees cadence structure directly —
  it processes observations independently — but the *clustering loss* (below) shapes the
  latent space so that ON/OFF structure becomes linearly visible in it.
- The **Random Forest** sees a whole cadence as the concatenation of its six latents
  (48 features) and decides true (ETI-like: present in ONs, absent in OFFs) vs false
  (noise/RFI). Trees over a 48-dim space are cheap, interpretable (SHAP), and robust at the
  extreme class-imbalance operating point inference runs at.

Hyperparameters live on `BetaVAEConfig` / `RandomForestConfig`
([`config.py`](../src/aetherscan/config.py)): `latent_dim` 8, `dense_layer_size` 512,
`kernel_size` (3, 3), `beta` 1.5, `alpha` 10.0; RF: 1000 trees, bootstrap, `max_features`
"sqrt", seeded (11).

## Beta-VAE

### Encoder (`build_encoder`)

Input `(16, 512, 1)` — one observation, "grayscale". Nine Conv2D layers (four stride-2
downsampling, five stride-1 feature layers), then dense compression to the latent posterior:

| Enc layer | Filters | Stride | Input shape | Output shape |
| --- | --- | --- | --- | --- |
| 1 | 16 | 2 | (16, 512, 1) | (8, 256, 16) |
| 2 | 16 | 1 | (8, 256, 16) | (8, 256, 16) |
| 3 | 32 | 2 | (8, 256, 16) | (4, 128, 32) |
| 4 | 32 | 1 | (4, 128, 32) | (4, 128, 32) |
| 5 | 32 | 1 | (4, 128, 32) | (4, 128, 32) |
| 6 | 64 | 2 | (4, 128, 32) | (2, 64, 64) |
| 7 | 64 | 1 | (2, 64, 64) | (2, 64, 64) |
| 8 | 128 | 1 | (2, 64, 64) | (2, 64, 128) |
| 9 | 256 | 2 | (2, 64, 128) | (1, 32, 256) |

Then `Flatten` → `(8192,)` → `Dense(dense_layer_size=512)` → two parallel heads
`z_mean` and `z_log_var`, each `Dense(latent_dim=8)`, and a `Sampling` layer producing `z`.
Every layer carries the same regularization stack: HeNormal init (GlorotNormal on the latent
heads), L1(0.001) activity regularization (sparse activations), L2(0.01) kernel + bias
regularization. `z_log_var`'s bias initializes at **−3.0**, tightening the initial posterior
around the prior so early training isn't swamped by sampling noise.

### Sampling layer (reparameterization)

`Sampling` implements the reparameterization trick: sampling is non-differentiable, so the
randomness is isolated in `ε ~ N(0, I)` and the sample expressed as a deterministic function
of the learned parameters:

```
z = z_mean + exp(0.5 · z_log_var) · ε
```

Gradients flow through `z_mean`/`z_log_var` while `ε` carries no parameters. The layer is
registered with `keras.utils.register_keras_serializable(package="aetherscan")`, so saved
`.keras` encoders load anywhere without `custom_objects` (this is what makes cross-machine
checkpoint interop in [`GPU_RUNTIME_GUIDE.md`](GPU_RUNTIME_GUIDE.md) work).

### Decoder (`build_decoder`)

An exact mirror: `Dense(512)` → `Dense(8192)` → `Reshape(1, 32, 256)` → nine
`Conv2DTranspose` layers reversing the encoder table row-by-row (decoder layer *i* outputs
the *input* channel count of encoder layer *10−i*, with the same stride), ending in a
1-filter, stride-2 layer with **sigmoid** activation — the output is bounded to [0, 1] to
match the log-normalized input and license the BCE reconstruction loss. The symmetry is a
maintained invariant: any layer-structure change on one side must be mirrored on the other
(both docstrings say so, and the unit tests check output shapes).

The decoder is not needed at inference time; it ships as `vae_decoder_{tag}.keras` because
the latent-traversal diagnostic decodes perturbed latents
([`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)).

### Composite loss (`BetaVAE.compute_total_loss`)

Each training batch is a triplet `(main, true, false)` of cadence batches
(shape `(B, 6, 16, 512)` each; see [`PREPROCESSING.md`](PREPROCESSING.md) for how the three
arrays are generated):

```
total = reconstruction(main) + β · KL(main) + α · (L_true(true) + L_false(false))
```

- **Reconstruction**: `main` is reshaped to `(B·6, 16, 512, 1)`, encoded, decoded, and scored
  with binary cross-entropy summed over the spectrogram and averaged over the batch
  (`from_logits=False`; the decoder output is already sigmoid-bounded).
- **KL divergence** against the standard-normal prior, closed form per dimension, summed over
  the latent and averaged over the batch:
  `KL = −½ · Σ_d (1 + log σ_d² − μ_d² − σ_d²)`.
  `β = 1.5` is the Beta-VAE knob: β > 1 buys a more disentangled, better-regularized latent
  space at some reconstruction cost — which is exactly the trade we want, since the latents
  (not the reconstructions) are the product.
- **Clustering losses** — the SETI-specific part. Two primitives over per-observation latents
  (the `z` samples of a cadence's six observations, encoded with `training=True`):

  ```
  loss_same(a, b) = mean ‖a − b‖²          # minimized → pulls a and b together
  loss_diff(a, b) = mean 1 / (‖a − b‖² + 1e-8)   # minimized → pushes a and b apart
  ```

  With a cadence's latents labeled `a1, b, a2, c, a3, d` (ON, OFF, ON, OFF, ON, OFF):

  - **`L_true`** (`compute_clustering_loss_true`), applied to true-class cadences: the sum of
    `loss_same` over all ordered ON/ON pairs (a1a2, a1a3, a2a1, a2a3, a3a1, a3a2) and all
    ordered OFF/OFF pairs (bc, bd, cb, cd, db, dc), **plus** `loss_diff` over all nine ON×OFF
    cross pairs. Minimizing it forces ON latents into one cluster, OFF latents into another,
    and the two clusters apart — the latent-space encoding of "appears when pointed at the
    target, disappears when pointed away".
  - **`L_false`** (`compute_clustering_loss_false`), applied to false-class cadences: the same
    pair enumeration but with `loss_same` **everywhere** (the nine cross pairs included) — in
    a noise/RFI cadence, all six observations are morphologically the same thing, so all six
    latents should collapse to a single cluster.

  `α = 10.0` weights the pair; the true/false batches feed only these heads (their gradients
  don't flow through the reconstruction term). This is what makes the RF's job possible with
  1000 shallow trees: the class geometry is pre-built into the representation.

Loss components are logged per epoch to `training_stats` and plotted separately
(`beta_vae_loss_curves_{tag}.png`), so a failing term is directly attributable.

## Random Forest

### Feature layout

`prepare_latent_features()` reshapes `(n_cadences · 6, latent_dim)` per-observation latents
into `(n_cadences, 6 · latent_dim)` rows by concatenating each cadence's six latents
**in observation order** (row-major `.ravel()`):

```
feature index f  =  obs_idx · latent_dim + dim     # obs 0..5 (ABACAD), dim 0..7
[ obs0_z0 .. obs0_z7 | obs1_z0 .. obs1_z7 | ... | obs5_z0 .. obs5_z7 ]
   ON            OFF             ON            OFF            ON           OFF
```

The caller must keep rows `i·6 .. i·6+5` grouped as cadence *i* — both training
(`shuffle=False` datasets pin encode order to the index arrays) and inference (snippet-major
`.npy` order preserved through padding/truncation) uphold this contract. Features are indexed
`obs{i}_z{d}` in the SHAP plots; even-numbered observations are ON-source.

Note the two models consume **different latent tensors**: the RF trains and infers on the
sampled `z`, while the latent-space visualizations use the deterministic `z_mean` — one is
the representation under the model's own noise, the other the noise-free embedding.

### Classifier and thresholds

`RandomForestClassifier(n_estimators=1000, bootstrap=True, max_features="sqrt",
random_state=11, n_jobs=-1)`, fit on binary labels (`true_*` subtypes = 1). Prediction
surfaces:

- `predict_proba` → `(n, 2)` columns `[P(false), P(true)]`. With 1000 bootstrap trees the
  probability resolution is fine enough for a 0.99 operating point to be meaningful.
- `predict(threshold=0.5)` / `predict_verbose(threshold=0.5)` — generic helpers; the
  **pipeline never uses 0.5**. Inference thresholds at
  `config.inference.classification_threshold` (default **0.99**): a snippet is a candidate iff
  `P(true) > 0.99`. The RF diagnostics evaluate at this same deployment threshold so the
  confusion matrix and error markers reflect production behavior, not sklearn's argmax.
- **Confidence semantics**: the `confidence` stored in `inference_results` is the probability
  of the **predicted** class — `P(true)` for candidates, `P(false)` for rejections — so a
  confident rejection also scores ~1.0. The full `P(true)` vector per cadence is summarized
  (quantiles, above-threshold counts) into the `inference_cadences` manifest
  ([`DATABASE.md`](DATABASE.md)); don't confuse the two columns.

Why 0.99: candidates are forwarded to human review, real positives are expected to be
vanishingly rare, and the synthetic training distribution is easier than the real sky — the
threshold buys precision and the confidence-distribution / calibration plots
([`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md), [`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md))
tell you what it costs in recall.

## Persisted model artifacts

| File | Producer | Contents / purpose |
| --- | --- | --- |
| `vae_encoder_{tag}.keras` | `save_models` | The inference feature extractor (outputs `z_mean, z_log_var, z`). |
| `vae_decoder_{tag}.keras` | `save_models` | Mirror decoder; needed only for latent traversal / reconstruction diagnostics. |
| `random_forest_{tag}.joblib` | `train_random_forest` / `save_models` | The fitted sklearn classifier (joblib of the bare `RandomForestClassifier`). |
| `rf_eval_artifacts_{tag}.joblib` | `train_random_forest` | Dict of train/val features, binary + subtype labels, val probabilities and threshold-consistent predictions, the threshold and SNR range used — the single source every RF plot consumes (and what lets a resumed run skip RF retraining). |
| `rf_shap_values_{tag}.joblib` | `_compute_or_load_shap_values` | Cached SHAP outputs: positive-class summary values (+ the val row indices they correspond to), pairwise interaction values, and a log-loss decomposition (`model_output="log_loss"`), normalized across shap versions by `_select_positive_class_shap`. Computing these is minutes of work; every SHAP figure reuses the cache. |
| `umap_{obs,cadence}_nn{n}_md{m}_{tag}.joblib` | `plot_latent_space_gif` | Fitted UMAP reducers per (n_neighbors, min_dist): obs-level (8-dim inputs) and cadence-level (48-dim). Reused by the RF decision-boundary plot and by inference's latent-projection figure — which is why deleting them breaks those two figures but nothing else. |

Interpretation guidance for the SHAP/diagnostic figures lives with the plot catalog in
[`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md). The short version of what the SHAP artifacts
*mean*: summary values attribute each cadence's `P(true)` to individual `obs{i}_z{d}`
features (sign = direction, magnitude = influence); interaction values split attributions
into per-feature main effects (diagonal) and pure pairwise effects (off-diagonal — the
ON×OFF blocks are where cadence logic shows); the log-loss decomposition re-attributes *loss*
instead of probability, separating features the model uses well (loss-decreasing) from ones
it uses badly (loss-increasing).
