const yaml = require('js-yaml');
const fs = require('fs');
const path = require('path');

module.exports = function(eleventyConfig) {
  // Load all leaderboard YAML files
  eleventyConfig.addGlobalData("leaderboards", function() {
    const leaderboardDir = '../letta-leaderboard';
    const leaderboards = {};

    // Try to load benchmark-specific leaderboards
    const benchmarkFiles = [
      'leaderboard_filesystem.yaml',
      'leaderboard_core-memory-read.yaml',
      'leaderboard_core-memory-update.yaml'
    ];

    benchmarkFiles.forEach(filename => {
      const filePath = path.join(leaderboardDir, filename);
      if (fs.existsSync(filePath)) {
        const benchmarkName = filename.replace('leaderboard_', '').replace('.yaml', '');
        const fileContents = fs.readFileSync(filePath, 'utf8');
        leaderboards[benchmarkName] = yaml.load(fileContents);
      }
    });

    // If no benchmark-specific files exist, fall back to the old format
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
    const fileContents = fs.readFileSync('../letta-leaderboard/leaderboard_results.yaml', 'utf8');
    return yaml.load(fileContents);
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
