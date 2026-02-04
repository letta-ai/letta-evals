# Sidecar Trajectory Viewer

Local, static viewer for sidecar environment datasets. It builds one JSON data file per category and renders scenarios in a small interactive web UI.

## Quick Start

```bash
uv run python serve.py  # Dev server with auto-rebuild
```

The dev server rebuilds data files on every page load, so changes to the source JSONL files are reflected immediately on refresh.

To build data files without starting the server:

```bash
uv run python build_data.py
```

## Output Files

The builder writes:

- `viewer/data/index.json` (category index)
- `viewer/data/<category>.json` (one per sidecar category)
