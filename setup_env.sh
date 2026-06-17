#!/usr/bin/env bash
# ============================================================
# Route Resilience — Environment Setup Script
# Usage: bash setup_env.sh
# ============================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        Route Resilience — Phase 1 Environment Setup     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Detect Python ────────────────────────────────────────────
PYTHON=""
for candidate in /opt/homebrew/bin/python3 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        VER=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON="$candidate"
            echo "  ✓ Using Python: $($candidate --version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ✗ Python 3.10+ not found."
    echo "  Install via Homebrew: brew install python@3.13"
    echo "  Or download from: https://www.python.org/downloads/"
    exit 1
fi

# ── Create virtual environment ────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment (.venv)…"
    "$PYTHON" -m venv .venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# ── Activate ──────────────────────────────────────────────────
source .venv/bin/activate
echo "  ✓ Virtual environment activated"

# ── Upgrade pip ───────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Install packages in groups (handles network instability) ──
echo ""
echo "  Installing dependencies (this may take a few minutes)…"
echo ""

install_group() {
    local label="$1"
    shift
    echo "  [$label]"
    pip install "$@" --quiet --resume-retries 5 2>&1 | grep -E "^(error:|ERROR|WARNING:.*failed)" || true
    echo "  ✓ $label installed"
}

install_group "Core utilities"      pyyaml numpy pandas colorlog tqdm requests Pillow click
install_group "Visualization"       matplotlib seaborn
install_group "Computer vision"     opencv-python
install_group "Augmentation"        albumentations
install_group "Geospatial (light)"  shapely pyproj
install_group "Geospatial (full)"   fiona geopandas rasterio
install_group "Image analysis"      scikit-image
install_group "Noise"               noise

echo ""
echo "  Verifying key imports…"
python -c "
packages = {
    'yaml': 'pyyaml',
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'albumentations': 'albumentations',
    'shapely': 'shapely',
    'fiona': 'fiona',
    'geopandas': 'geopandas',
    'rasterio': 'rasterio',
    'skimage': 'scikit-image',
}
missing = []
for pkg, install_name in packages.items():
    try:
        __import__(pkg)
        print(f'    ✓  {install_name}')
    except ImportError:
        print(f'    ✗  {install_name}  (failed — retry: pip install {install_name})')
        missing.append(install_name)
if missing:
    print(f'')
    print(f'    ⚠ {len(missing)} package(s) failed. Re-run: bash setup_env.sh')
else:
    print('')
    print('    All packages verified ✓')
"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                        ║"
echo "║                                                         ║"
echo "║  Activate env : source .venv/bin/activate              ║"
echo "║  Run pipeline : python run_pipeline.py                 ║"
echo "║  Download data: python src/download_utils.py --dataset check ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
