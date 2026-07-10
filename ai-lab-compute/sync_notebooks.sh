#!/usr/bin/env bash
#
# Sync jupytext-paired notebooks in this folder.
#
# Each paired .py carries a jupytext header; this finds those files and runs
# `jupytext --sync` on each, propagating edits between the .py and its .ipynb
# ("newer file wins"). Editing the .py and running `python3.11 the_script.py`
# does NOT update the .ipynb — that only happens here.
#
# Usage:
#   ./sync_notebooks.sh              # sync only (fast; does not refresh outputs)
#   ./sync_notebooks.sh --execute    # sync and re-run, so outputs/charts land in the .ipynb
#   ./sync_notebooks.sh --execute deepmind_compute_model.py   # limit to specific files
#
set -euo pipefail

cd "$(dirname "$0")"

PY=python3.11
EXECUTE=""
FILES=()

for arg in "$@"; do
  if [[ "$arg" == "--execute" ]]; then
    EXECUTE="--execute"
  else
    FILES+=("$arg")
  fi
done

# If no explicit files given, discover every .py with a jupytext pairing header.
if [[ ${#FILES[@]} -eq 0 ]]; then
  while IFS= read -r f; do FILES+=("$f"); done < <(grep -l "jupytext" ./*.py)
fi

for f in "${FILES[@]}"; do
  echo "==> syncing $f ${EXECUTE:+(with execute)}"
  $PY -m jupytext --sync $EXECUTE "$f"
done

echo "Done. ${#FILES[@]} notebook(s) synced."
