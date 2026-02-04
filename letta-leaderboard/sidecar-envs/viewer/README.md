# Sidecar Trajectory Viewer

Local, static viewer for sidecar environment datasets. It builds one JSON data file per category and renders scenarios in a small interactive web UI.

## Quick Start

```bash
uv run python build_data.py  # Build data files
uv run python -m http.server 5173  # Serve the viewer
```

Open <http://localhost:5173> in your browser to view the viewer.

## Output Files

The builder writes:

- `viewer/data/index.json` (category index)
- `viewer/data/<category>.json` (one per sidecar category)

Re-run `build_data.py` whenever the datasets change.
