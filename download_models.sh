#!/usr/bin/env bash
set -euo pipefail

URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
DEST_DIR="models"
DEST_FILE="${DEST_DIR}/$(basename "$URL")"

mkdir -p "$DEST_DIR"

echo "Downloading to: $DEST_FILE"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 --continue-at - -o "$DEST_FILE" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "$DEST_FILE" "$URL"
else
  echo "Error: please install curl or wget." >&2
  exit 1
fi

# Basic sanity check
if [ ! -s "$DEST_FILE" ]; then
  echo "Download failed or file is empty: $DEST_FILE" >&2
  exit 1
fi
echo "Done: $DEST_FILE"


# --- Clone segment-anything-2-real-time repo ---
echo "Cloning segment-anything-2-real-time repository..."
if [ ! -d "segment-anything-2-real-time" ]; then
    git clone https://github.com/NomanAnjum09/segment-anything-2-real-time.git
else
    echo "Repository already exists. Pulling latest changes..."
    cd segment-anything-2-real-time || exit
    git pull origin main
    cd ..
fi

