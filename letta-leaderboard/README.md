# Letta Leaderboard

Results from Letta's agent benchmarks, published at https://leaderboard.letta.com/. The benchmarks are no longer run. This folder keeps only what the site needs:

- `leaderboard_filesystem_results.yaml` and `leaderboard_skill_results.yaml`: the published results, one benchmark tab each.
- `updates.md`: the site's updates section.
- `leaderboard_site/`: the [Eleventy](https://www.11ty.dev/) site that renders them. `.github/workflows/deploy-leaderboard.yml` builds it and deploys it to GitHub Pages on every push to `main`. See its README to run it locally.

The suites, datasets and results script that produced these results have been removed. The last commit that has them is `21dc940`:

```bash
git checkout 21dc940 -- letta-leaderboard/
```
