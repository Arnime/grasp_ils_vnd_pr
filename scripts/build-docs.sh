#!/usr/bin/env bash
# Build script for Cloudflare Pages (and local docs builds).
# Cloudflare build command: bash scripts/build-docs.sh
set -euo pipefail

pip install -r python/requirements/docs.txt
pip install -e .
mkdocs build --strict
