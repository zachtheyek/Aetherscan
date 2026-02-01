# Aetherscan: Scaling Technosignature Detection with Deep Learning

## Technical Presentation Outline

**Duration**: 45-60 minutes
**Slides**: 35-40
**Audience**: Researchers, engineers, and technical stakeholders

---

## Section 1: Introduction (5 slides)

### Slide 1: Title
- **Title**: Aetherscan: Scaling Technosignature Detection with Deep Learning
- **Subtitle**: Breakthrough Listen's First Production-Grade ML Pipeline for SETI @ Scale
- **Authors**: Zach Yek, Peter Ma, Matt Lebofsky, Steve Croft
- **Affiliation**: Breakthrough Listen, UC Berkeley SETI Research Center

### Slide 2: The Challenge
- Breakthrough Listen data scale: Petabytes of radio telescope data
- Traditional manual review bottleneck: ~1000 candidates per hour per human reviewer
- Goal: Process entire BL archive with ML, reducing human review to high-confidence candidates
- Visual: Data volume growth chart, human bandwidth bottleneck illustration

### Slide 3: What is a Technosignature?
- Definition: Observable signature of technology (narrowband drifting signals)
- Properties of interest:
  - Narrowband (< 10 Hz bandwidth)
  - Doppler drift (signal frequency changes due to relative motion)
  - ON/OFF cadence pattern (signal follows target, not local interference)
- Visual: Waterfall plot showing narrowband drifting signal vs RFI

### Slide 4: Prior Work - Ma et al. 2023
- Nature Astronomy publication: "A deep-learning search for technosignatures from 820 unique stars"
- Key results:
  - Analyzed 480 hours of GBT observations
  - Found 8 promising candidates (all later attributed to RFI)
  - Demonstrated DL viability for SETI
- Limitation: Research prototype, not production-ready
- Visual: Paper figure showing candidate signals

### Slide 5: Aetherscan's Mission
- Goal: Extend Ma et al. approach to production scale
- Target: Process entire BL archive (millions of observations)
- Key requirements:
  - Multi-GPU training for large models
  - Fault tolerance for long-running jobs
  - Real-time monitoring and alerts
  - Reproducible, configurable pipeline
- Visual: Scale comparison (820 stars → millions)

---

## Section 2: Technical Architecture (8 slides)

### Slide 6: System Architecture Overview
- Two-stage pipeline: Beta-VAE + Random Forest
- Training path: Background data → Signal injection → Curriculum learning → Model training
- Inference path: New observations → Preprocessing → Encoding → Classification → Candidates
- Visual: Full architecture diagram (from README)

### Slide 7: Why Two Stages?
- **Stage 1 - Beta-VAE**: Unsupervised feature extraction
  - Learns compressed representation of spectrograms
  - Custom clustering loss exploits ON/OFF pattern
  - Output: 8-dimensional latent vector per observation
- **Stage 2 - Random Forest**: Supervised classification
  - Trained on latent features (48 features per cadence)
  - Binary classification: technosignature vs non-technosignature
  - Interpretable feature importances

### Slide 8: Beta-VAE Architecture
- Encoder: 9 Conv2D layers → Dense → 8-dim latent (z_mean, z_log_var)
- Decoder: Mirror of encoder with Conv2DTranspose
- Reparameterization trick for differentiable sampling
- Input shape: (16, 512, 1) per observation
- Visual: Network diagram with layer shapes

### Slide 9: Custom Clustering Loss
- Key insight: ON/OFF pattern provides supervision signal
- **L_same**: Minimize distance between same-class observations
  - ON₁ ↔ ON₂ ↔ ON₃ (should cluster together)
  - OFF₁ ↔ OFF₂ ↔ OFF₃ (should cluster together)
- **L_diff**: Maximize distance between different-class observations
  - ON ↔ OFF (should be separated)
- Visual: Latent space visualization showing clusters

### Slide 10: Total Loss Function
```
L_total = L_reconstruction + β·L_KL + α·(L_true + L_false)
```
- **L_reconstruction**: Binary cross-entropy for reconstruction quality
- **L_KL**: KL divergence for latent regularization (β = 1.5)
- **L_true**: Clustering loss for true signals (ON ≠ OFF)
- **L_false**: Clustering loss for false signals (all similar)
- α = 10.0 balances clustering vs reconstruction
- Visual: Loss component breakdown

### Slide 11: Random Forest Classifier
- 1000 decision trees with bootstrap sampling
- Input: 48 features (8 latent dims × 6 observations)
- Feature selection: sqrt(48) ≈ 7 features per split
- Parallel training: -1 jobs (all CPU cores)
- Output: Probability of technosignature class
- Visual: Feature importance plot

### Slide 12: Data Generation with setigen
- Synthetic signal injection using setigen library
- Signal parameters:
  - Drift rate: Uniform distribution
  - SNR: Controlled by curriculum learning
  - Start frequency: Random within band
- ON observations: Signal injected
- OFF observations: Background only
- Visual: Example injected signals at various SNR

### Slide 13: Comparison Table - Ma et al. vs Aetherscan

| Aspect | Ma et al. 2023 | Aetherscan |
|--------|----------------|------------|
| **Purpose** | Research | Production |
| **VAE Type** | Beta-VAE | Beta-VAE |
| **Latent Dim** | 8 | 8 |
| **β (KL weight)** | 1.5 | 1.5 |
| **α (clustering)** | 10.0 | 10.0 |
| **RF Estimators** | 1000 | 1000 |
| **Multi-GPU** | No | MirroredStrategy |
| **Gradient Accumulation** | No | Yes (batch=3072) |
| **Curriculum Learning** | Fixed SNR | 3 schedules |
| **Checkpointing** | Manual | Per-round auto |
| **Monitoring** | Manual | Real-time + Slack |
| **Configuration** | Scripts | Dataclass + CLI |

---

## Section 3: Engineering Innovations (8 slides)

### Slide 14: Curriculum Learning
- Problem: Low-SNR signals are hard to detect initially
- Solution: Start with high-SNR (easy), progressively decrease to low-SNR (hard)
- Benefits:
  - Faster initial convergence
  - Better final performance
  - More stable training
- Visual: SNR range progression over training rounds

### Slide 15: Curriculum Schedule Visualization
- Three schedule types:
  - **Linear**: Constant difficulty increase
  - **Exponential**: Fast initial progress, slow refinement
  - **Step**: Discrete difficulty levels
- Parameters: snr_base, initial_snr_range, final_snr_range
- Visual: Plot of SNR range vs training round for each schedule

### Slide 16: Multi-GPU Training
- TensorFlow MirroredStrategy for data parallelism
- NCCL AllReduce for efficient gradient synchronization
- Key considerations:
  - Models created within strategy.scope()
  - Datasets distributed with experimental_distribute_dataset()
  - Batch sizes scaled by num_replicas
- Visual: GPU communication diagram

### Slide 17: Gradient Accumulation
- Problem: Limited GPU memory constrains batch size
- Solution: Accumulate gradients over multiple mini-batches
- Formula: effective_batch = per_replica_batch × num_replicas × accum_steps
- Default: 128 × 4 × 6 = 3072 effective batch size
- Benefits: Large batch training on limited hardware
- Visual: Accumulation timeline diagram

### Slide 18: Memory Optimization
- **Shared Memory**: Inter-process data sharing without copying
  - Background data loaded once, shared across workers
  - Explicit cleanup required (unlink on creator process)
- **DataHolder Pattern**: Thread-safe data management with generators
  - Lock protects against premature clearing
  - Local reference caching for generator lifetime
- **Generator-based Datasets**: Data stays on CPU, transferred to GPU on-demand
- Visual: Memory flow diagram

### Slide 19: Fault Tolerance
- **Per-round Checkpointing**: Encoder, decoder, RF saved after each round
- **Automatic Retry**: Configurable max_retries with retry_delay
- **Graceful Degradation**: Training continues from last checkpoint on failure
- **Signal Handling**: SIGTERM/SIGINT trigger cleanup before exit
- Visual: Checkpoint recovery flowchart

### Slide 20: Real-time Monitoring
- **Resource Tracking**: CPU, RAM, GPU utilization per second
- **Database Logging**: Async SQLite writes for all metrics
- **Slack Integration**:
  - Error alerts (configurable level)
  - Training milestones
  - Completion notifications
- Visual: Sample Slack alert and resource utilization plot

### Slide 21: Thread-Safe Architecture
- **Singleton Pattern**: Config, Database, ResourceManager
  - Double-checked locking for thread safety
  - Single instance across all threads
- **Async Database**: Queue-based writes, single writer thread
  - Eliminates SQLITE_BUSY errors
  - Buffered writes for performance
- **ResourceManager**: Centralized cleanup for pools, shared memory, threads
- Visual: Threading architecture diagram

---

## Section 4: Performance & Results (6 slides)

### Slide 22: Training Performance
- Metrics:
  - Time per training round
  - GPU utilization during training
  - Memory usage patterns
- Example configuration:
  - 20 rounds × 100 epochs
  - ~500K samples per round
  - 4× A100 GPUs
- Visual: Training time breakdown by phase

### Slide 23: Model Convergence
- Loss curves:
  - Total loss vs epoch
  - Component losses (reconstruction, KL, clustering)
- Effect of curriculum learning:
  - Comparison with fixed-SNR training
  - Convergence speed improvement
- Visual: Multi-panel loss curve plots

### Slide 24: Classification Performance
- Metrics:
  - ROC curve and AUC
  - Precision-Recall curve
  - Confusion matrix at various thresholds
- Performance at 0.9 threshold:
  - Precision: X%
  - Recall: Y%
  - F1: Z%
- Visual: ROC curve, confusion matrix

### Slide 25: Scalability Analysis
- Multi-GPU scaling efficiency:
  - 1 GPU baseline
  - 2, 4, 8 GPU speedup
  - Communication overhead analysis
- Memory scaling:
  - Per-GPU memory usage
  - Effective batch size impact
- Visual: Scaling efficiency plot

### Slide 26: Inference Throughput
- Cadences per second at various batch sizes
- End-to-end latency:
  - Preprocessing time
  - Encoding time
  - Classification time
  - Database write time
- Visual: Throughput vs batch size plot

### Slide 27: Resource Utilization
- Sample monitoring output:
  - CPU utilization over training run
  - RAM usage patterns
  - GPU memory and utilization
- Bottleneck identification:
  - Data generation (CPU-bound)
  - Training (GPU-bound)
- Visual: Resource utilization timeline

---

## Section 5: Deployment & Operations (5 slides)

### Slide 28: Installation & Setup
```bash
# Clone and create environment
git clone https://github.com/zachtheyek/Aetherscan.git
conda env create -f environment.yml
pip install -e .

# Configure paths
export AETHERSCAN_DATA_PATH="/path/to/data"
export AETHERSCAN_MODEL_PATH="/path/to/models"
export AETHERSCAN_OUTPUT_PATH="/path/to/outputs"
```
- Visual: Setup workflow diagram

### Slide 29: Training Workflow
- Command examples:
  ```bash
  # Fresh training run
  aetherscan train --curriculum-schedule exponential --save-tag v1

  # Resume from checkpoint
  aetherscan train --load-tag round_10 --save-tag v1_resumed
  ```
- Output artifacts:
  - Checkpoints: `vae_encoder_round_XX.keras`, `vae_decoder_round_XX.keras`
  - RF model: `rf_model.joblib`
  - Metrics: `aetherscan.db`
  - Plots: `resource_utilization.png`

### Slide 30: Inference Pipeline
- Command example:
  ```bash
  aetherscan inference --test-files observations.npy --threshold 0.9
  ```
- Pipeline steps:
  1. Load preprocessed cadences
  2. Distribute across GPUs
  3. Generate latent representations
  4. Run RF classification
  5. Write candidates to database
- Visual: Inference pipeline diagram

### Slide 31: Candidate Review Process
- Candidates stored in SQLite database
- Query interface for filtering:
  - By confidence score
  - By source file
  - By timestamp
- Export for manual review
- Future: Web interface for candidate visualization
- Visual: Mock candidate review interface

### Slide 32: Operational Monitoring
- Slack alerts for:
  - Training errors
  - Round completion
  - Candidate detection
- Log files:
  - `aetherscan.log` (file)
  - Console output (real-time)
- Database queries for historical metrics
- Visual: Monitoring dashboard concept

---

## Section 6: Future Work & Conclusions (4 slides)

### Slide 33: Roadmap
- **Near-term**:
  - Complete evaluation pipeline
  - Inference pipeline for real observations
  - Web-based candidate review interface
- **Medium-term**:
  - HuggingFace model hosting
  - Multi-node distributed training
  - Integration with BL data archive
- **Long-term**:
  - Real-time streaming inference
  - Active learning for candidate labeling
  - Multi-telescope support
- Visual: Roadmap timeline

### Slide 34: Known Limitations
- Current limitations:
  - Single-node training only (multi-GPU, not multi-node)
  - Fixed observation pattern (3 ON + 3 OFF)
  - No online learning capability
- Technical debt:
  - Some hard-coded dimensions in VAE
  - Inference pipeline not fully tested
  - Limited automated testing coverage
- Visual: Limitation matrix with priority

### Slide 35: Conclusions
- Key contributions:
  - Production-ready implementation of Ma et al. approach
  - Curriculum learning for improved training
  - Multi-GPU distributed training
  - Comprehensive monitoring and fault tolerance
- Impact:
  - Enables processing of BL archive at scale
  - Reduces human review burden
  - Provides framework for future SETI ML work
- Visual: Key metrics summary

### Slide 36: Questions & Discussion
- Contact information
- GitHub repository: https://github.com/zachtheyek/Aetherscan
- Breakthrough Listen: https://breakthroughlisten.org/
- Ma et al. 2023: https://arxiv.org/abs/2301.12670
- Visual: QR code to repository

---

## Appendix Slides (Optional)

### A1: Configuration Reference
- Full table of configuration options
- Environment variables
- CLI argument mapping

### A2: Database Schema
- Tables: training_metrics, inference_results, system_resources
- Key fields and indexes

### A3: Latent Space Visualizations
- t-SNE/UMAP of latent representations
- Class separation visualization

### A4: Hardware Recommendations
- Training: 4× A100 (80GB), 64GB RAM, 500GB SSD
- Inference: 1× GPU (16GB), 32GB RAM
- Cost estimates for cloud deployment

---

## Speaker Notes Summary

### Key Messages
1. Aetherscan brings research-grade SETI ML to production scale
2. Two-stage approach (Beta-VAE + RF) is proven and interpretable
3. Engineering innovations enable reliable long-running training
4. Open source and designed for community contribution

### Anticipated Questions
- "How does it compare to other anomaly detection approaches?"
  - Custom clustering loss specifically designed for ON/OFF pattern
  - RF provides interpretability vs black-box classifiers

- "What about false positives from complex RFI?"
  - Curriculum learning helps model distinguish subtle patterns
  - High confidence threshold (0.9) prioritizes precision

- "Can this be applied to other telescope data?"
  - Architecture is general; main work is adapting preprocessing
  - ON/OFF pattern is common in SETI observations

### Demo Options
- Live training run (short, 1-2 rounds)
- Inference on pre-loaded data
- Resource monitoring visualization
- Candidate database query
