# Aetherscan runtime image (OCI) — the registry-published twin of aetherscan.def.
# (No `# syntax=` frontend directive on purpose: nothing here uses BuildKit-specific syntax, so
# we avoid an unpinned, rate-limited Docker Hub pull of the frontend image on the CI runner.)
#
# CI (.github/workflows/publish-image.yml) builds this and pushes it to GHCR so clusters can
# `pull` a prebuilt image instead of each building a .sif locally. Publishing OCI (not a .sif)
# is deliberate: both Apptainer and SingularityCE consume it via
#     apptainer|singularity pull aetherscan-ngc25.02.sif docker://ghcr.io/zachtheyek/aetherscan:vX.Y.Z
# and each converts the runtime-neutral OCI layers into its OWN native .sif — so we never ship
# a fork-specific artifact, sidestepping any Apptainer-vs-SingularityCE SIF-compat question.
#
# The Aetherscan Python code is NOT baked in — utils/run_container.sh bind-mounts the repo at
# runtime. So this image = NGC base + the pinned pip extras only, and it changes ONLY when this
# Dockerfile (its base digest, the LABELs below, any layer) or requirements-container.txt changes —
# never on a code-only release. Editing a LABEL below IS a Dockerfile change: it rebuilds. Keep it in
# lockstep with aetherscan.def / requirements-container.txt / environment.yml / pyproject.toml
# (SECURITY.md Version Selection Policy). The base MUST match aetherscan.def's `From:` digest.
#
# The NGC TensorFlow 25.02 base ships TF 2.17 / Python 3.12 / CUDA 12.8 / cuDNN 9.7.1 /
# NCCL 2.25.1 with Blackwell-ready (sm_120) and Ampere (sm_86) kernels.
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3@sha256:c83b37d26f19ab00d8a13cf974edd079c3d099918ec3110c304a989d6e2f75d5

# Licensing (see docs/RELEASE.md "Container image licensing"): this is a "Compatible derived
# CONTAINER" under the NVIDIA Deep Learning Container License §1(c). The image as a whole is
# governed by that license (NOT the repo's BSD-3-Clause, which covers only the bind-mounted
# Aetherscan source, absent here). com.nvidia.notice carries the §2(b)-required notice verbatim.
LABEL org.opencontainers.image.title="Aetherscan runtime" \
      org.opencontainers.image.source="https://github.com/zachtheyek/Aetherscan" \
      org.opencontainers.image.description="Aetherscan runtime: NGC TF 25.02 base + pinned pip extras. Aetherscan code (BSD-3-Clause) is bind-mounted at runtime, not included here. The image is governed by the NVIDIA Deep Learning Container License." \
      org.opencontainers.image.licenses="LicenseRef-NVIDIA-Deep-Learning-Container-License" \
      com.nvidia.notice="This software contains source code provided by NVIDIA Corporation."

# pip extras layered onto the NGC base (identical to aetherscan.def's %post).
COPY requirements-container.txt /opt/aetherscan-requirements.txt
RUN pip install --no-cache-dir -r /opt/aetherscan-requirements.txt \
    && rm -f /opt/aetherscan-requirements.txt

# Mirrors aetherscan.def's %environment: don't pick up host-side Python packages that would
# shadow the pinned image versions; quieten TF's C++ INFO flood (INFO+ still prints).
ENV PYTHONNOUSERSITE=1 \
    TF_CPP_MIN_LOG_LEVEL=1
