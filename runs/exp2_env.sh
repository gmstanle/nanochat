#!/bin/bash
# Shared config for exp2 scripts.
# Use a raw (unexpanded) path so ~ expands on the machine that runs the script.

if [[ -z "${NANOCHAT_BASE_DIR_RAW:-}" ]]; then
  NANOCHAT_BASE_DIR_RAW='~/nanochat-exp2/.cache/nanochat'
fi
export NANOCHAT_BASE_DIR_RAW
