#!/bin/bash

# Ensure PYTHONPATH includes the project root for imports
export PYTHONPATH="."

echo "=== Test 1: Generate Low-Confidence Frames ==="
# Check if test_video.mp4 exists, if not, skip this test
if [ -f "test_video.mp4" ]; then
  python scripts/inference.py \
    --config config/inference/base.yaml \
    --mode offline \
    --source test_video.mp4
else
  echo "Skipping Test 1: test_video.mp4 not found."
fi

echo -e "\n=== Test 2: Check Frames Saved ==="
ls -lah data/low_confidence_frames/ | head -10

echo -e "\n=== Test 3: Export Local (No API Key) ==="
python scripts/auto_labeling_cli.py \
  --config config/training/base.yaml \
  --provider local

echo -e "\n=== Test 4: Check Local Export ==="
if [ -f "data/auto_labeled/auto_labels.json" ]; then
  cat data/auto_labeled/auto_labels.json | python -m json.tool | head -30
else
  echo "Local export file not found."
fi

echo -e "\n=== Test 5: Upload to Roboflow ==="
if [ -z "$ROBOFLOW_API_KEY" ]; then
  echo "Error: ROBOFLOW_API_KEY is not set. Please set it before running."
  # export ROBOFLOW_API_KEY="your_api_key_here"
fi
python scripts/auto_labeling_cli.py \
  --config config/training/base.yaml \
  --provider roboflow

echo -e "\n✅ Testing complete!"