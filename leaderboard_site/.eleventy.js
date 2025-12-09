const yaml = require('js-yaml');
const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');

module.exports = function(eleventyConfig) {
  // Load all leaderboard YAML files
  eleventyConfig.addGlobalData("leaderboards", function() {
    const leaderboardDir = '../letta-leaderboard';
    const leaderboards = {};

    // Find all leaderboard_*.yaml files
    if (fs.existsSync(leaderboardDir)) {
      const files = fs.readdirSync(leaderboardDir);

      files.forEach(filename => {
        // Match pattern: leaderboard_<name>.yaml or leaderboard_<name>_results.yaml
        if (filename.startsWith('leaderboard_') && filename.endsWith('.yaml')) {
          const filePath = path.join(leaderboardDir, filename);

          // Extract benchmark key from filename
          let benchmarkKey = filename.replace('leaderboard_', '').replace('.yaml', '');
          // Remove '_results' suffix if present
          benchmarkKey = benchmarkKey.replace('_results', '');

          // Skip if it's just "results" (backward compat file)
          if (benchmarkKey === 'results' || benchmarkKey === 'all') {
            return;
          }

          const fileContents = fs.readFileSync(filePath, 'utf8');
          const data = yaml.load(fileContents);

          // Check if the YAML has the new structure with metrics and results
          if (data && typeof data === 'object' && data.results && Array.isArray(data.results)) {
            // New structure: extract results array
            leaderboards[benchmarkKey] = data.results;
          } else if (Array.isArray(data)) {
            // Old structure: data is already an array
            leaderboards[benchmarkKey] = data;
          } else {
            // Fallback
            leaderboards[benchmarkKey] = data;
          }
        }
      });
    }

    // If no benchmark-specific files found, fall back to leaderboard_results.yaml
    if (Object.keys(leaderboards).length === 0) {
      const fallbackPath = path.join(leaderboardDir, 'leaderboard_results.yaml');
      if (fs.existsSync(fallbackPath)) {
        const fileContents = fs.readFileSync(fallbackPath, 'utf8');
        const data = yaml.load(fileContents);
        leaderboards['filesystem'] = Array.isArray(data) ? data : (data.results || data);
      }
    }

    return leaderboards;
  });

  // Load benchmark names from each leaderboard file
  eleventyConfig.addGlobalData("benchmarkNames", function() {
    const leaderboardDir = '../letta-leaderboard';
    const benchmarkNames = {};

    // Find all leaderboard_*.yaml files
    if (fs.existsSync(leaderboardDir)) {
      const files = fs.readdirSync(leaderboardDir);

      files.forEach(filename => {
        if (filename.startsWith('leaderboard_') && filename.endsWith('.yaml')) {
          const filePath = path.join(leaderboardDir, filename);

          // Extract benchmark key from filename
          let benchmarkKey = filename.replace('leaderboard_', '').replace('.yaml', '');
          benchmarkKey = benchmarkKey.replace('_results', '');

          if (benchmarkKey === 'results' || benchmarkKey === 'all') {
            return;
          }

          const fileContents = fs.readFileSync(filePath, 'utf8');
          const data = yaml.load(fileContents);

          // Extract benchmark_name if it exists in the new structure
          if (data && typeof data === 'object' && data.benchmark_name) {
            benchmarkNames[benchmarkKey] = data.benchmark_name;
          } else {
            // Fallback: use capitalized key
            benchmarkNames[benchmarkKey] = benchmarkKey.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          }
        }
      });
    }

    return benchmarkNames;
  });

  // Load metrics metadata from each leaderboard file
  eleventyConfig.addGlobalData("metricsMetadata", function() {
    const leaderboardDir = '../letta-leaderboard';
    const allMetrics = {};

    // Find all leaderboard_*.yaml files
    if (fs.existsSync(leaderboardDir)) {
      const files = fs.readdirSync(leaderboardDir);

      files.forEach(filename => {
        if (filename.startsWith('leaderboard_') && filename.endsWith('.yaml')) {
          const filePath = path.join(leaderboardDir, filename);

          // Extract benchmark name
          let benchmarkName = filename.replace('leaderboard_', '').replace('.yaml', '');
          benchmarkName = benchmarkName.replace('_results', '');

          if (benchmarkName === 'results' || benchmarkName === 'all') {
            return;
          }

          const fileContents = fs.readFileSync(filePath, 'utf8');
          const data = yaml.load(fileContents);

          // Extract metrics if they exist in the new structure
          if (data && typeof data === 'object' && data.metrics) {
            // Merge metrics from this file into the global metrics object
            Object.assign(allMetrics, data.metrics);
          }
        }
      });
    }

    return allMetrics;
  });

  // Load arrange_by configuration from each leaderboard file
  eleventyConfig.addGlobalData("arrangeByConfig", function() {
    const leaderboardDir = '../letta-leaderboard';
    const arrangeByConfig = {};

    // Find all leaderboard_*.yaml files
    if (fs.existsSync(leaderboardDir)) {
      const files = fs.readdirSync(leaderboardDir);

      files.forEach(filename => {
        if (filename.startsWith('leaderboard_') && filename.endsWith('.yaml')) {
          const filePath = path.join(leaderboardDir, filename);

          // Extract benchmark key from filename
          let benchmarkKey = filename.replace('leaderboard_', '').replace('.yaml', '');
          benchmarkKey = benchmarkKey.replace('_results', '');

          if (benchmarkKey === 'results' || benchmarkKey === 'all') {
            return;
          }

          const fileContents = fs.readFileSync(filePath, 'utf8');
          const data = yaml.load(fileContents);

          // Extract arrange_by if it exists (default to "average")
          if (data && typeof data === 'object' && data.arrange_by) {
            arrangeByConfig[benchmarkKey] = data.arrange_by;
          } else {
            arrangeByConfig[benchmarkKey] = "average";
          }
        }
      });
    }

    return arrangeByConfig;
  });

  // Load the latest last_updated date from all leaderboard files
  eleventyConfig.addGlobalData("lastUpdated", function() {
    const leaderboardDir = '../letta-leaderboard';
    let latestDate = null;

    // Find all leaderboard_*.yaml files
    if (fs.existsSync(leaderboardDir)) {
      const files = fs.readdirSync(leaderboardDir);

      files.forEach(filename => {
        if (filename.startsWith('leaderboard_') && filename.endsWith('.yaml')) {
          const filePath = path.join(leaderboardDir, filename);

          // Extract benchmark key from filename
          let benchmarkKey = filename.replace('leaderboard_', '').replace('.yaml', '');
          benchmarkKey = benchmarkKey.replace('_results', '');

          if (benchmarkKey === 'results' || benchmarkKey === 'all') {
            return;
          }

          const fileContents = fs.readFileSync(filePath, 'utf8');
          const data = yaml.load(fileContents);

          // Extract last_updated if it exists and compare to find latest
          if (data && typeof data === 'object' && data.last_updated) {
            const dateStr = data.last_updated;
            if (!latestDate || dateStr > latestDate) {
              latestDate = dateStr;
            }
          }
        }
      });
    }

    return latestDate;
  });

  // Load updates content from markdown file
  eleventyConfig.addGlobalData("updatesContent", function() {
    const updatesPath = path.join(__dirname, 'src', '_includes', 'updates.md');
    if (fs.existsSync(updatesPath)) {
      const md = new MarkdownIt({
        html: true,
        linkify: true,
        typographer: true
      });
      const markdown = fs.readFileSync(updatesPath, 'utf8');
      return md.render(markdown);
    }
    return '<p>No updates available.</p>';
  });

  // Keep backward compatibility with single leaderboard
  eleventyConfig.addGlobalData("leaderboard", function() {
    const fallbackPath = '../letta-leaderboard/leaderboard_results.yaml';
    if (fs.existsSync(fallbackPath)) {
      const fileContents = fs.readFileSync(fallbackPath, 'utf8');
      return yaml.load(fileContents);
    }
    // If leaderboard_results.yaml doesn't exist, return the first benchmark from leaderboards
    const leaderboardDir = '../letta-leaderboard';
    const benchmarkFiles = [
      'leaderboard_filesystem.yaml',
      'leaderboard_core-memory-read.yaml',
      'leaderboard_core-memory-update.yaml'
    ];

    for (const filename of benchmarkFiles) {
      const filePath = path.join(leaderboardDir, filename);
      if (fs.existsSync(filePath)) {
        const fileContents = fs.readFileSync(filePath, 'utf8');
        return yaml.load(fileContents);
      }
    }

    return [];
  });

  eleventyConfig.addPassthroughCopy("src/icons");
  eleventyConfig.addPassthroughCopy("src/styles.css");
  eleventyConfig.addPassthroughCopy("src/letta-logo.svg");

  return {
    dir: {
      input: "src",
      output: "_site"
    }
  };
};
