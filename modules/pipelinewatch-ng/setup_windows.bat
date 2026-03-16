@echo off
REM ============================================================
REM  PipelineWatch-NG — Windows Anaconda Setup Script
REM  Run this once from Anaconda Prompt in the project root.
REM  Usage: setup_windows.bat
REM ============================================================

echo.
echo ===================================================
echo  PipelineWatch-NG — Environment Setup (Windows)
echo ===================================================
echo.

REM Step 1: Create conda environment
echo [1/5] Creating conda environment "pipelinewatch" (Python 3.10)...
conda create -n pipelinewatch python=3.10 -y

REM Step 2: Activate and install base geo stack via conda-forge
echo.
echo [2/5] Installing geospatial stack from conda-forge...
call conda activate pipelinewatch
conda install -c conda-forge geopandas rasterio shapely pyproj -y

REM Step 3: Install remaining packages via pip
echo.
echo [3/5] Installing remaining packages via pip...
pip install earthengine-api geemap folium
pip install pandas numpy scipy scikit-learn
pip install matplotlib streamlit streamlit-folium
pip install requests tqdm python-dotenv

REM Step 4: Register Jupyter kernel
echo.
echo [4/5] Registering Jupyter kernel...
pip install ipykernel
python -m ipykernel install --user --name pipelinewatch --display-name "Python 3 (pipelinewatch)"

REM Step 5: GEE authentication
echo.
echo [5/5] Authenticating with Google Earth Engine...
echo.
echo  IMPORTANT: This will open a browser tab.
echo  - Sign in with your Google account
echo  - Approve Earth Engine access
echo  - Copy the token and paste it back here
echo.
python -c "import ee; ee.Authenticate()"

echo.
echo ===================================================
echo  Setup complete!
echo.
echo  Next steps:
echo  1. Open Anaconda Prompt
echo  2. conda activate pipelinewatch
echo  3. cd notebooks
echo  4. jupyter notebook
echo  5. Open 01_module1_ingestion.ipynb
echo  6. Set GEE_PROJECT_ID in Cell 2 and run all cells
echo ===================================================
pause
