#!/usr/bin/env python3
"""
Bless trained weights as a release: create the HF release tag (vX.Y.Z) pointing at a
training upload's commit (docs/RELEASE.md step 5).

    python utils/hf_tag_release.py --save-tag train_20260101_120000 --release v1.0.0

This is the human "these weights are the release" decision — the release CD workflow
(.github/workflows/release.yml) verifies the tag exists but deliberately cannot create it.
A thin wrapper over HfApi.create_tag: the new release tag points at the same commit as the
training run's save-tag (created by `train --hf-upload`).

Auth: needs a write-scoped HF_TOKEN in the environment; a gitignored .env in the working
directory is loaded when python-dotenv is available. Standalone on purpose — no PYTHONPATH
or aetherscan install required.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional here — HF_TOKEN may already be exported
    load_dotenv = None

# Must match config.hf.repo_id's default (src/aetherscan/config.py).
DEFAULT_REPO_ID = "zachtheyek/aetherscan"

_RELEASE_TAG_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bless trained weights as a release: create the HF release tag (vX.Y.Z) "
        "pointing at a training upload's commit (docs/RELEASE.md step 5)."
    )
    parser.add_argument(
        "--save-tag",
        required=True,
        help="Training run tag whose uploaded weights are being blessed (e.g. train_20260101_120000); "
        "must already exist on the HF repo (created by `train --hf-upload`)",
    )
    parser.add_argument(
        "--release",
        required=True,
        help="Release tag to create, vX.Y.Z (must match the version the release PR ships)",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF model repo id (default: {DEFAULT_REPO_ID})",
    )
    args = parser.parse_args()

    if not _RELEASE_TAG_PATTERN.match(args.release):
        print(f"ERROR: --release must look like vX.Y.Z (e.g. v1.0.0), got {args.release!r}")
        return 1
    # Fail a mangled save-tag here with a clear message rather than as a Hub 404 later.
    if not args.save_tag or any(c.isspace() for c in args.save_tag):
        print(
            f"ERROR: --save-tag must be a non-empty tag without whitespace, got {args.save_tag!r}"
        )
        return 1

    if load_dotenv is not None:
        load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: no HF_TOKEN in the environment. Creating tags needs a write-scoped "
            "token — put it in the gitignored .env (never commit it)."
        )
        return 1

    api = HfApi()
    try:
        api.create_tag(args.repo_id, tag=args.release, revision=args.save_tag)
    except RevisionNotFoundError:
        print(
            f"ERROR: save-tag {args.save_tag!r} not found on {args.repo_id}. Was the run "
            f"trained with --hf-upload (and did the hf_upload stage succeed)?"
        )
        return 1
    except HfHubHTTPError as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 409:
            print(
                f"ERROR: release tag {args.release!r} already exists on {args.repo_id}. "
                f"Released weights are immutable by convention — bless a new version "
                f"instead, or delete the tag on the Hub first if it was created in error."
            )
            return 1
        raise
    print(
        f"Blessed: {args.release} -> weights of training run {args.save_tag!r} "
        f"(https://huggingface.co/{args.repo_id}/tree/{args.release})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
