#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo "=== MeghMukt — LISS-IV cloud removal ==="

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close." _; exit 1
fi
echo "Python: $($PY --version)"

if ! $PY -c "import torch, rasterio, streamlit, cv2, skimage, sklearn, yaml, omnicloudmask" 2>/dev/null; then
  echo "First run: installing the required packages (a few minutes, needs internet) ..."
  $PY -m pip install -r requirements.txt || { echo "Package installation failed."; read -r -p "Press Enter to close." _; exit 1; }
fi

missing=0
for f in checkpoints/meghgan_liss4.pt checkpoints/meghgan_real.pt checkpoints/diffusion_real.pt \
         data/processed/norm.json data/processed/harmoniser.json data/processed/mask_params.json; do
  [ -f "$f" ] || { echo "Missing: $f"; missing=1; }
done
if [ $missing = 1 ]; then
  echo "Trained models or prepared data are missing. See the scripts/ folder (01 -> 15) to rebuild them."
  read -r -p "Press Enter to close." _; exit 1
fi

echo "Starting the dashboard at http://localhost:8501  (close this window to stop it)"
$PY -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
