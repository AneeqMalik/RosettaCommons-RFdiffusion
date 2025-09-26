#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${RFD_MODEL_S3_URI:-}" ]]; then
  echo "RFD_MODEL_S3_URI not set; skipping download"
  exit 0
fi
mkdir -p "${RFD_MODEL_DIR:-/models}"
aws s3 sync "$RFD_MODEL_S3_URI" "${RFD_MODEL_DIR:-/models}"
