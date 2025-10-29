const yaml = require('js-yaml');
const fs = require('fs');

module.exports = function(eleventyConfig) {
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
