#!/usr/bin/env bash
# Download diarization models (pyannote segmentation-3.0 MIT + WeSpeaker embedding) into models/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p models
SEG_TBZ="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"

SEG_MODEL="models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL="models/wespeaker_en_voxceleb_resnet34_LM.onnx"
# Pinned SHA256 of the consumed artifacts (the .onnx files the diarizer loads).
SEG_SHA="220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
EMB_SHA="e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"

# verify aborts (and removes the bad file so the next run re-fetches) on mismatch.
verify() { # $1=file  $2=expected-sha256
  echo "$2  $1" | sha256sum -c - || { echo "checksum FAILED for $1"; rm -f "$1"; exit 1; }
}

# Segmentation (extract -> models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx)
if [ ! -f "$SEG_MODEL" ]; then
  echo "downloading segmentation model"; curl -fL "$SEG_TBZ" -o models/seg.tar.bz2
  tar xjf models/seg.tar.bz2 -C models && rm -f models/seg.tar.bz2
fi
verify "$SEG_MODEL" "$SEG_SHA"
# Embedding
if [ ! -f "$EMB_MODEL" ]; then
  echo "downloading embedding model"; curl -fL "$EMB_URL" -o "$EMB_MODEL"
fi
verify "$EMB_MODEL" "$EMB_SHA"
echo "SEG=$ROOT/$SEG_MODEL"
echo "EMB=$ROOT/$EMB_MODEL"
