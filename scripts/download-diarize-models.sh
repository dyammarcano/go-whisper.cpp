#!/usr/bin/env bash
# Download diarization models (pyannote segmentation-3.0 MIT + WeSpeaker embedding) into models/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p models
SEG_TBZ="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"

# Segmentation (extract -> models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx)
if [ ! -f "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
  echo "downloading segmentation model"; curl -fL "$SEG_TBZ" -o models/seg.tar.bz2
  tar xjf models/seg.tar.bz2 -C models && rm -f models/seg.tar.bz2
fi
# Embedding
if [ ! -f "models/wespeaker_en_voxceleb_resnet34_LM.onnx" ]; then
  echo "downloading embedding model"; curl -fL "$EMB_URL" -o models/wespeaker_en_voxceleb_resnet34_LM.onnx
fi
# Optional checksum verification (fill SHA once known): echo "<sha>  models/..." | sha256sum -c -
echo "SEG=$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
echo "EMB=$ROOT/models/wespeaker_en_voxceleb_resnet34_LM.onnx"
