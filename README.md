# Front/Back Garden Classifier

Automatically distinguishes **front gardens** (facing the road) from **back gardens** (away from the road) using aerial imagery and OpenStreetMap data.

![Example Output](output/garden_classification.png)

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Aerial Imagery (Google Tiles API)                          │
│     └─> RGB satellite/aerial photo of the area                 │
├─────────────────────────────────────────────────────────────────┤
│  2. OpenStreetMap Data (FREE)                                  │
│     └─> Building footprints + Road network                     │
├─────────────────────────────────────────────────────────────────┤
│  3. Vegetation Detection (OpenCV)                              │
│     └─> HSV color segmentation to find green areas             │
├─────────────────────────────────────────────────────────────────┤
│  4. Geometric Classification                                    │
│     └─> For each garden pixel:                                 │
│         • Find nearest building                                │
│         • Find direction to nearest road                       │
│         • If pixel is TOWARD road → FRONT GARDEN              │
│         • If pixel is AWAY from road → BACK GARDEN            │
├─────────────────────────────────────────────────────────────────┤
│  5. Output                                                      │
│     └─> Segmentation mask + statistics + visualization         │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Google Tiles API Key

Edit `config.py` and add your API key:

```python
GOOGLE_TILES_API_KEY = "your-api-key-here"
```

> **Note:** You can run in demo mode without an API key using `--demo`

### 3. Run Classification

```bash
# Default: 67 Clonsilla Road, Blanchardstown, 500m radius
python main.py

# Demo mode (no API key required - uses placeholder image)
python main.py --demo

# Custom location
python main.py --lat 53.3917 --lon -6.3878 --radius 300

# Higher zoom for more detail
python main.py --zoom 20
```

## Output

Results are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `garden_classification.png` | 4-panel comparison figure |
| `overlay.png` | Simple overlay on aerial image |
| `original.png` | Original aerial imagery |

### Classification Colors

- 🟢 **Green** = Front garden (facing road)
- 🔵 **Blue** = Back garden (away from road)
- 🟡 **Yellow outline** = Buildings
- 🔴 **Red lines** = Roads

## Configuration

Edit `config.py` to customize:

```python
# Target location
TARGET_LAT = 53.3917
TARGET_LON = -6.3878
ANALYSIS_RADIUS_M = 500

# Imagery settings
ZOOM_LEVEL = 19  # 18-20 recommended

# Green detection thresholds (HSV)
GREEN_HSV_LOWER = (35, 25, 25)
GREEN_HSV_UPPER = (85, 255, 255)
```

## Requirements

- Python 3.10+
- macOS (tested on M3 MacBook Air)
- ~2GB disk space for dependencies
- Google Maps Platform API key (for real imagery)

## Project Structure

```
front-back-garden/
├── config.py              # Configuration and API keys
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── src/
│   ├── tiles.py          # Google Tiles API fetcher
│   ├── osm.py            # OpenStreetMap data fetcher
│   ├── garden_detector.py # Vegetation detection
│   ├── classifier.py     # Front/back classification logic
│   └── visualizer.py     # Output rendering
└── output/               # Generated results
```

## Accuracy Notes

The heuristic approach works well for:
- ✅ Typical suburban housing (semi-detached, terraced)
- ✅ Clear road layouts
- ✅ Well-maintained gardens with grass

May struggle with:
- ⚠️ Corner properties (multiple road frontages)
- ⚠️ Apartment complexes
- ⚠️ Properties with unusual layouts
- ⚠️ Dense trees obscuring ground

## Future Improvements

1. **Machine Learning**: Train a CNN on labeled data for better accuracy
2. **Property boundaries**: Integrate land registry data
3. **Multi-class**: Detect driveways, patios, pools separately
4. **Batch processing**: Process entire suburbs efficiently

## License

MIT
