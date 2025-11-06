const yaml = require('js-yaml');
const fs = require('fs');
const path = require('path');

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

          // Extract benchmark name
          let benchmarkName = filename.replace('leaderboard_', '').replace('.yaml', '');
          // Remove '_results' suffix if present
          benchmarkName = benchmarkName.replace('_results', '');

          // Skip if it's just "results" (backward compat file)
          if (benchmarkName === 'results' || benchmarkName === 'all') {
            return;
          }

          const fileContents = fs.readFileSync(filePath, 'utf8');
          leaderboards[benchmarkName] = yaml.load(fileContents);
        }
      });
    }

    // If no benchmark-specific files found, fall back to leaderboard_results.yaml
    if (Object.keys(leaderboards).length === 0) {
      const fallbackPath = path.join(leaderboardDir, 'leaderboard_results.yaml');
      if (fs.existsSync(fallbackPath)) {
        const fileContents = fs.readFileSync(fallbackPath, 'utf8');
        leaderboards['filesystem'] = yaml.load(fileContents);
      }
    }

    return leaderboards;
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
