# Letta Leaderboard Site

Static website for displaying Letta evaluation benchmarks. Built with [Eleventy](https://www.11ty.dev/).

View the live site at https://leaderboard.letta.com/

## Running Locally

1. **Install dependencies:**
```bash
npm install
```

2. **Start the development server:**
```bash
npm run serve
```

The site will be available at `http://localhost:8080` (or another port if 8080 is in use). The development server includes live reloading, so changes to YAML files or templates will automatically refresh the browser.

3. **Build for production:**
```bash
npm run build
```

This generates static HTML files in the `_site` directory.

## How It Works

- The site automatically reads all `leaderboard_*.yaml` files from the `../letta-leaderboard` directory
- Each YAML file becomes a separate benchmark tab on the website
- The site extracts benchmark names, metrics metadata, and results from the YAML structure
- Icons for providers are loaded from `src/icons/`

## File Structure

```
leaderboard_site/
├── src/
│   ├── index.njk           # Main template (Nunjucks)
│   ├── styles.css          # Styles
│   ├── icons/              # Provider logos (png files)
│   └── letta-logo.svg      # Letta logo
├── .eleventy.js            # Eleventy configuration
├── package.json            # Dependencies and scripts
└── _site/                  # Generated output (git-ignored)
```

## Adding Provider Icons

Provider icons are displayed next to model names in the leaderboard. To add a new provider icon:

1. Add a PNG file to `src/icons/`
2. Name it after the provider prefix (e.g., `openai.png` for `openai/*` models)
3. Recommended size: 32x32 or 64x64 pixels
4. The site will automatically display the icon for all models with that provider prefix

## Troubleshooting

**Site not showing updates:**
- After updating YAML files in `letta-leaderboard/`, the development server should auto-reload
- If not, restart the server with `npm run serve`
- Check browser console for any JavaScript errors

**Provider icons not displaying:**
- Verify the PNG file exists in `src/icons/`
- Filename must match the provider prefix exactly (case-sensitive)
- Clear browser cache if you recently added/updated an icon

**Leaderboard data not loading:**
- Ensure YAML files are properly formatted (use a YAML validator if needed)
- Check that files follow the naming pattern: `leaderboard_*.yaml`
- Verify the YAML structure matches the expected format:
  ```yaml
  benchmark_name: "Benchmark Name"
  metrics:
    metric_key:
      name: "Metric Display Name"
      description: "Metric description"
  results:
    - model: provider/model-name
      average: 75.5
      total_cost: 10.5
      metric_key: 75.5
  ```

## Eleventy Configuration

The `.eleventy.js` file contains the configuration for loading leaderboard data:

- `leaderboards`: Loads all benchmark results from YAML files
- `benchmarkNames`: Extracts display names for each benchmark
- `metricsMetadata`: Loads metric definitions and descriptions
- Passthrough copy: Copies static assets (icons, CSS, SVG) to output

The configuration automatically handles both old (array-based) and new (object with metadata) YAML formats.
