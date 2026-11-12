#!/usr/bin/env bash
# macOS: torch bundles its own libomp, xgboost uses Homebrew's; two OpenMP runtimes in one process segfault/hang.
# Point torch's copy at Homebrew's so both share one. Rerun after reinstalling torch.
set -euo pipefail
cd "$(dirname "$0")/.."
lib=$(ls -d .venv/lib/python*/site-packages/torch/lib)
brew=$(brew --prefix libomp)/lib/libomp.dylib
[[ -f "$brew" ]] || { echo "brew install libomp first"; exit 1; }
ln -sf "$brew" "$lib/libomp.dylib" && echo "torch -> $brew"
