@echo off
cd /d "%~dp0"
echo === MeghMukt - LISS-IV cloud removal ===
where python >nul 2>nul || (echo Python was not found. Install it from https://www.python.org/downloads/ & pause & exit /b 1)
python -c "import torch, rasterio, streamlit, cv2, skimage, sklearn, yaml, omnicloudmask" 2>nul || (
  echo First run: installing the required packages ^(a few minutes, needs internet^) ...
  python -m pip install -r requirements.txt || (echo Package installation failed. & pause & exit /b 1)
)
if not exist checkpoints\meghgan_liss4.pt (echo Missing trained models - see the scripts folder. & pause & exit /b 1)
echo Starting the dashboard at http://localhost:8501  (close this window to stop it)
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
