# GPS Dashboard

A lightweight Python project for parsing, analyzing and visualizing GPS/IMU track data. It includes utilities to parse CSV logs, detect stops and zones, analyze IMU/gps quality, and run simulations/insights useful for building a GPS dashboard or analytics pipeline.

## Features

- CSV parsing and coordinate handling
- GPS quality checks and IMU analysis
- Stop detection and zone simulation
- Insight engine to derive analytics from tracks
- Export utilities for processed data

## Quick summary / contract

- Inputs: GPS/IMU CSV files (typical columns: timestamp, lat, lon, alt, speed, heading, acc, gyro). See your data source for exact fields.
- Outputs: Parsed tracks, stop events, zone simulation results, CSV exports and Python data structures (lists/dicts/NumPy arrays depending on the module).
- Success: Modules run on normal CSV logs and return Python-native results or write CSV exports.
- Error modes: malformed CSV, missing columns or invalid coordinate values will raise parsing errors (see `analysis/csv_parser.py`).

## Repo layout

```
app.py
requirements.txt
analysis/
    coordinates.py        # coordinate helpers
    csv_parser.py         # CSV parsing utilities
    gps_quality.py        # GPS quality checks
    imu_analyzer.py       # IMU analysis
    insight_engine.py     # High-level analytics
    stop_detector.py      # Stop detection
    zone_simulator.py     # Zone simulation
components/
utils/
    export.py             # export helpers
```

## Requirements

This project uses Python 3.x. Install dependencies from `requirements.txt`.

Install in a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you don't use a virtual env, consider using pipx or your preferred environment manager.

## Running the project

The repository includes a top-level `app.py` which can be used as an entry point for simple runs. Typical usage (from project root):

```bash
# run the app (adjust file/args as implemented in app.py)
python app.py
```

Many analysis modules are importable. Example Python usage (interactive or in a script):

```python
from analysis.csv_parser import parse_csv
from analysis.stop_detector import detect_stops

tracks = parse_csv("/path/to/track.csv")
stops = detect_stops(tracks)
print(f"Detected {len(stops)} stops")
```

Refer to module docstrings for function signatures and expected column names.

## Development notes

- Keep functions small and testable.
- Add unit tests for parsers and detectors when adding new behaviors.
- Follow the existing code style in the repository.

### Recommended edge cases to test

- Empty CSV or only header row
- Missing required columns (timestamp, lat, lon)
- Irregular timestamp ordering or duplicates
- Extremely noisy/invalid coordinates

## Tests

There are no tests in the repository by default. To add tests, create a `tests/` directory and use pytest. Example quick-start:

```bash
pip install pytest
pytest -q
```


