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
  (48 base features, optionally extended with encoder-uncertainty features — the
  [latent-representation variants](#latent-representation-variants-282) below) and decides
  true (ETI-like: present in ONs, absent in OFFs) vs false (noise/RFI). Trees over a
  ~48-dim space are cheap, interpretable (SHAP), and robust at the extreme class-imbalance
  operating point inference runs at.

Hyperparameters live on `BetaVAEConfig` / `RandomForestConfig`
([`config.py`](../src/aetherscan/config.py)): `latent_dim` 8, `dense_layer_size` 512,
`kernel_size` (3, 3), `beta` 1.5, `alpha` 10.0; RF: 1000 trees, bootstrap, `max_features`
"sqrt", `random_state` derived from the pipeline root seed (`Config.resolved_rf_seed()`;
see [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md#reproducibility)).

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
Every layer uses the same initialization: HeNormal (GlorotNormal on the latent heads) with
zero biases. `z_log_var`'s bias initializes at **−3.0**, tightening the initial posterior
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

- **Regularization** *(evaluated and removed, #293)*: the conv/dense layers historically
  declared L1/L2 penalties (`activity_regularizer`, `kernel_regularizer`, `bias_regularizer`)
  that the custom training loop never added to the objective — dead code since inception, so
  every model this pipeline trained was effectively unregularized. A calibration sweep (#293)
  activated them across a range of penalty strengths (the added penalty spanning ~0.3–1.6% of
  the objective) and found no benefit at this architecture and these data scales:
  recall@0.01FPR, validation AUC, and active-latent-dim count were all statistically
  indistinguishable from the unregularized model. The declarations were removed.
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

### Latent-representation variants (#282)

The encoder emits **three** tensors per observation — `z_mean`, `z_log_var`, and the sampled
`z = z_mean + exp(z_log_var/2)·ε`. `z` is a *lossy, stochastic* collapse of the two posterior
parameters into one number per dimension: training on it gives the forest a dataset-level
noise regularization (each cadence lands somewhere in its posterior cloud — sampling as
augmentation) but discards the per-sample uncertainty, while training on `z_mean` (+
`z_log_var`-derived features) is noise-free and lets the forest reason *conditionally* on
each cadence's own uncertainty. The two carry complementary information — which is why the
choice is made empirically: `rf_train` sweeps the full catalogue in
[`latent_variants.py`](../src/aetherscan/latent_variants.py) on one shared dataset/split and
records the winner in the saved config (`rf.latent_variant`), which inference reads to
rebuild features identically (see [`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md) for the
selection protocol). Feature layout is always `[lead block | extras]`, obs-major within
blocks; feature counts below assume the defaults (6 obs × 8 dims = 48).

| Variant (`VARIANT_ORDER`, simple → complex) | Lead block | Extras | F |
| --- | --- | --- | --- |
| `z_mean` | per-obs posterior means | — | 48 |
| `z` | one stochastic sample (the legacy baseline) | — | 48 |
| `z_aug` | trained on `z_mean` + `rf.z_aug_draws` sampled draws as extra *rows*; evaluated on plain `z_mean` | — | 48 |
| `z_mean_total_kl` | `z_mean` | total KL per cadence | 49 |
| `z_mean_obs_logvar` | `z_mean` | per-observation mean `log_var` | 54 |
| `z_mean_dim_logvar` | `z_mean` | per-dim mean `log_var` over obs | 56 |
| `z_mean_logvar_active` | `z_mean` | `log_var` restricted to ACTIVE dims (Burda et al. active units, `rf.active_units_threshold`; recorded as `rf.active_dims`) | 48–96 |
| `z_mean_logvar` | `z_mean` | full per-dimension `log_var` | 96 |

The ordering is a tie-break contract: selection walks it simple → complex, so a variant only
loses to a *more* complex one that beats it beyond bootstrap noise. Variant names are
persisted in `config_{tag}.json` and artifact filenames — treat them as stable. The
latent-space *visualizations* always use the deterministic `z_mean` regardless of variant.

### Classifier and thresholds

`RandomForestClassifier(n_estimators=1000, bootstrap=True, max_features="sqrt",
random_state=<derived from the root seed>, n_jobs=-1)`, fit on binary labels (`true_*`
subtypes = 1). `n_jobs` is a predict-time execution knob: `RandomForestModel.load()`
re-pins it from the **runtime** config (#301) — the pickled artifact otherwise carried the
training host's value wholesale. Prediction surfaces:

- `predict_proba` → `(n, 2)` columns `[P(false), P(true)]`. With 1000 bootstrap trees the
  probability resolution is fine enough for a 0.99 operating point to be meaningful.
- `predict(threshold=0.5)` / `predict_verbose(threshold=0.5)` — generic helpers; the
  **pipeline never uses 0.5**. Inference thresholds at
  `config.inference.classification_threshold` (default **0.99**), applied since #282 to the
  final score of the two-pass cascade (the seeded MC mean for snippets that survive the
  permissive `screening_threshold` pass — see
  [`INFERENCE_PIPELINE.md`](INFERENCE_PIPELINE.md)). The RF diagnostics evaluate at this same
  deployment threshold so the confusion matrix and error markers reflect production behavior,
  not sklearn's argmax.
- **Confidence semantics**: the `confidence` stored in `inference_results` is the probability
  of the **predicted** class — `P(true)` for candidates, `1 − P(true)` for rejections — so a
  confident rejection also scores ~1.0. The full `P(true)` vector per cadence is summarized
  (quantiles, above-threshold counts) into the `inference_cadences` manifest
  ([`DATABASE.md`](DATABASE.md)); don't confuse the two columns.

### Probability calibration

A forest's probabilities are rank-informative but not necessarily *calibrated* — "P = 0.99"
need not mean 99 % empirical frequency, which matters when the operating point lives in the
top percentile. Training therefore measures the winner's ECE on a held-out calibration split
and, only when it exceeds `rf.max_ece`, fits a calibrator (isotonic on large splits, else
sigmoid/Platt) that is **kept only if** it improves ECE without worsening Brier on a further
held-out test split ([`TRAINING_PIPELINE.md`](TRAINING_PIPELINE.md)). Calibration is
**monotonic**: rank metrics (AUC, recall@FPR) are unchanged — only probability *values*
move. A kept calibrator (`rf_calibrator_{tag}.joblib`, recorded as
`rf.calibration_active`/`rf.calibration_method`) is applied identically to every probability
at inference — pass 1, the MC draws, and the reference cloud — because an unapplied
calibrator would be a silent train/serve mismatch (inference hard-errors if the artifact is
missing).

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
| `random_forest_{tag}.joblib` | `train_random_forest` / `save_models` | The fitted sklearn classifier (joblib of the bare `RandomForestClassifier`) — the *winning* variant of the #282 sweep, under the canonical name. |
| `random_forest_{tag}_{variant}.joblib` (×8) | `train_random_forest` | Every variant's fitted forest from the sweep, kept for post-hoc comparison; only the winner is deployed. |
| `rf_calibrator_{tag}.joblib` | `train_random_forest` | The kept probability calibrator (`{method, model}` dict); exists only when calibration was fit *and* survived the test-split gate. Required by inference when `rf.calibration_active`. |
| `rf_eval_artifacts_{tag}.joblib` | `train_random_forest` | Dict of train/val features (winning variant), binary + subtype labels, raw + deployment-scored val probabilities and threshold-consistent predictions, the threshold and SNR range used, plus the sweep record (winner, active dims, per-variant metrics, calibration outcome, val partition) — the single source every RF plot consumes (and what lets a resumed run skip RF retraining and restore the sweep outcome). |
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
