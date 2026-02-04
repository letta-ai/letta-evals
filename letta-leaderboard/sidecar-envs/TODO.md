# TODO: Context-Management Envs (Essentials)

## Constraints

Response-only artifact-writing suites (RL-friendly):

- Reward must grade the provided `response` only (no environment re-run).
- No dependence on tools, trajectories, agent state/memory blocks, or filesystem state.
- Artifact contract lives entirely in the response (prefer strict JSON; sometimes Markdown).
- Deterministic and cheap grading (score in 0..1).
- Use rubric-style graders (`kind: model_judge`) and an extractor that reads the last assistant message / response text only.
- Dataset JSONL must include `input` (required). Optional: `ground_truth`, `agent_args`, `rubric_vars`, `extra_vars`.
- Stable sample IDs: `letta-evals` assigns `Sample.id` from the dataset line index; if you need stable IDs across shuffles/exports/training, include `extra_vars.sample_id` and ensure export tooling reads it.
- Suites should default to `target.base_url: https://api.letta.com/` (cloud), unless explicitly running a local server.

Full Letta Code suites (tools/state/filesystem):

- Target is `letta_code` (optionally seeded with AgentFile bootstrap via `from_af`).
- Reward may depend on tool calls, trajectory, agent state/memory blocks, and filesystem state.
- Must retrieve agent state post-run (memory blocks included) for memory grading.
- Custom extractors/graders may use trajectory + memory blocks.
- Document required skills and any backup/restore workflow that touches the filesystem.

Relevant links:

- Main doc: <https://docs.google.com/document/d/1j38hm-2e0uM00f9LhLcoIp1z4lY8kXMTEIBpew-EQ4s/edit>
- Memory subagent PR (letta-code): <https://github.com/letta-ai/letta-code/pull/498>
- Memory subagent demo (loom): <https://www.loom.com/share/88040d084931455fbc1222710f5db9cc>
- W&B project: <https://wandb.ai/letta/memory-post-training>
- SLIME tau-bench run: <https://wandb.ai/letta/memory-post-training/runs/66azk9hr>
- Skill creation docs: <https://github.com/letta-ai/letta-code/tree/main/src/skills/builtin/creating-skills>
- Memory defrag eval setup (letta-evals): <https://github.com/letta-ai/letta-evals/pull/160>
- End-to-end SLIME training (letta-train): <https://github.com/letta-ai/letta-train/pull/2>
- Letta server logprobs (letta-cloud): <https://github.com/letta-ai/letta-cloud/pull/9240>

Goals:

RL-ready environments for "learned context writing" in Letta Code:

- User model maintenance (stale facts)
- Session handoff (end-of-session -> next-session pickup)
- Next-query preparation (predict + precompute)

Targets:

- Baseline evaluation: strongest OpenAI / Anthropic / Google models (upper bound + teacher traces).
- Training: Qwen3-32B (via `letta-train` + SLIME).

Hard constraints:

- Samples must be independent (no ordering dependencies).
- Reward must be deterministic + cheap.
- Prefer JSON artifacts for robust grading.
- Length proxy for token efficiency is acceptable for response-only tasks.

## 1. Letta Code target: full-state evaluation

`LettaAgentTarget` already supports agent-state retrieval and `.af` seeding.
`LettaCodeTarget` needs parity for code-based evals that grade memory/tools/filesystem.

- [ ] **Per-sample sandboxes** (concurrency-safe). Current `sandbox` flag isolates per-model only; need per-sample working dirs for parallel runs within the same model.
- [ ] **Agent-state retrieval** after run (memory blocks included). `LettaCodeTarget.run()` currently returns `agent_state=None`; port the `retrieve_agent_state` path from `LettaAgentTarget`.
- [ ] **AgentFile seeding** (`from_af`) for deterministic bootstraps. Let `letta_code` suites specify an `.af` to pre-load before the prompt, matching `LettaAgentTarget.agent_file`.
- [ ] **Artifact ↔ memory-block sync** for memory grading workflows. Requires agent-state retrieval (above); reconcile response artifact with agent's memory blocks so graders can score both.

## 2. Diversity measurement & subset selection

When scaling to large generated pools (e.g. 10k candidates → 100 training scenarios), measure and enforce diversity.

**Metric — Vendi Score**: Embed each scenario (full input text), compute cosine similarity kernel $K$, then $\text{VS} = \exp(-\sum_i \lambda_i \log \lambda_i)$ where $\lambda_i$ are eigenvalues of $K/n$. Interpretation: effective number of distinct scenarios. Target VS ≥ 80 on a 100-sample selection.

**Selection — Greedy k-DPP**: Build L-kernel $L = \text{diag}(q) \cdot S \cdot \text{diag}(q)$ where $S$ is cosine similarity and $q_i$ is a quality score (from model judge or cheap proxy). Iteratively add the item maximizing marginal $\log\det(L_{\text{selected}})$. $O(n \cdot k^2)$ — trivial for 10k × 100.

**Verification — Behavior grid**: Define discrete axes per environment (e.g. user-model-maintenance: fact_type × conflict_type × diffusion × scope = 80 cells). After selection, check coverage ≥ 90% of cells. Empty cells flag gaps in the generator, not the selector — fill with targeted generation.

Pipeline: `generate 10k → quality filter (keep top 50%) → embed → greedy k-DPP select 100 → verify Vendi Score → check behavior grid → targeted fill for empty cells`.

- [ ] **Embedding pipeline**: embed scenarios with `text-embedding-3-small`, cache vectors.
- [ ] **Quality scorer**: cheap proxy (e.g. rubric pass/fail from existing graders) to produce per-scenario $q_i$.
- [ ] **k-DPP selection**: implement greedy k-DPP over L-kernel.
- [ ] **Vendi Score**: compute on selected subset, gate on threshold.
- [ ] **Behavior grid**: define axes per environment, verify coverage after selection.

## 3. Dataset expansion

Grow dataset coverage beyond the four hand-authored environments.

- [ ] **Inventory sources**: catalog agentfiles, reference skills, and existing datasets that can seed test-generation tasks. List candidates and licensing constraints.
- [ ] **Define extraction rules**: decide which source fields map to `input`, `ground_truth`, and `extra_vars` for each source type.
- [ ] **Build seed set** (~3 samples) and validate pipeline output quality before scaling.

## 4. SFT export: eval results → training data

Close the loop: high-scoring eval trajectories feed back into `letta-train` for fine-tuning (Qwen3-32B via SLIME).

- [ ] **`export_sft.py`**: filter evaluation results by score threshold and emit SFT-ready JSONL.
- [ ] **Schema compatibility**: decide export format compatible with `letta-train` (e.g., mapping `input` → `prompt`).
- [ ] **Stable IDs**: ensure sample identity survives shuffle/export (prefer `extra_vars.sample_id`; fall back to dataset line index).

---

## Further ideas

- **System-prompt vs. dynamic memory**: Does the agent correctly decide what belongs in core/system prompt (stable, always-loaded) vs. dynamically loaded based on task context?

- **Pre-commit over prose**: For code-enforceable preferences (e.g., "use pnpm not npm"), does the agent convert them to pre-commit/config rules rather than duplicating NL instructions in memory?

- **Context compression**: Given a long context, can the agent produce a summary that preserves task-critical info? (Measure: downstream task success with compressed vs full context)
