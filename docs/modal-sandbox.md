# Modal sandboxes

Run every sample inside a fresh Modal sandbox by adding a suite-level block
with exact application runtime pins:

```yaml
sandbox:
  kind: modal
  letta_evals_version: "0.25.0"
  letta_code_version: "0.30.5"
  secrets: [letta-api-key, openai-key]
  cpu: 2
  memory_mb: 4096
```

`image` is optional. When unset, the driver builds the system base in
`letta_evals/sandbox/Dockerfile`, then installs the requested `letta-evals`
and `@letta-ai/letta-code` releases in explicit Modal layers. Each exact pin
is part of its layer's command, so changing a pin rebuilds that layer and its
downstream layers without rebuilding the shared OS/toolchain base.

Both pins are required with the bundled base. Mutable values such as `latest`,
npm version ranges, and Git branch or tag references are rejected. For an
unreleased `letta-evals` revision, use a direct reference with the full commit
SHA:

```yaml
sandbox:
  kind: modal
  letta_evals_version: "letta-evals @ git+https://github.com/letta-ai/letta-evals.git@0123456789abcdef0123456789abcdef01234567"
  letta_code_version: "0.30.5"
```

The base checks Node `>=22.19.0` during the image build, and the application
layers smoke-test the Letta Code CLI and `typing_extensions.Sentinel`. The
runner retains its lightweight startup check for released `letta-evals` pins.

Override `image` only when you need additional system tools or a pre-built
runtime. A custom image may omit both pins because it bakes in its own
applications. If `letta_evals_version` is set with a custom image, it remains
a runtime assertion; `letta_code_version` is ignored and logs a warning.

The orchestrator (`letta-evals run`) keeps running on your host — same
sample loop, same `max_concurrent`, same JSONL output, same reward
composition. The only thing that changes is what happens *per sample*:
instead of executing in-process, the runner creates a Modal sandbox,
uploads the entire suite directory tree to `/mnt/suite/`, execs
`letta-evals run --sample ...` inside the sandbox, and round-trips the
final `SampleResult` JSON back.

## When to use this

- Different OS, controlled dependency versions, or system tools that
  shouldn't bleed into your host.
- Identical runs across CI, laptops, and leaderboard runners.
- Letta-code targets where you don't want the agent's `Bash` calls to
  touch the host filesystem.

## Setup

1. **Install letta-evals.** The Modal SDK ships as a dependency, so no
   extra is needed:

   ```sh
   pip install letta-evals
   ```

2. **Authenticate to Modal.** Either run `modal token new` or set
   `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`.

3. **Provide API keys.** No setup needed for the common ones:
   `letta-evals run` auto-loads `./.env`, and the runner forwards an
   allowlist of host env vars into the sandbox — `LETTA_API_KEY`,
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
   `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `TINKER_API_KEY` — whenever
   they're present. So a `.env` (or exported vars) is enough to run.

   - Forward **extra** variables with `forward_env: [NAME, ...]`.
   - For shared/CI use, pre-create named Modal Secrets
     (`modal secret create <name> KEY=...`) and list them under
     `secrets: [<name>]`. Only allowlisted names are forwarded — never
     your whole environment.

   The default image layers install the two pinned application runtimes,
   so most suites won't need a custom image.

### Building a custom image (optional)

If your agent invokes system tools the base image doesn't ship with
(compilers, language toolchains, project-specific binaries), build a custom
runtime and reference it via `sandbox.image`. The bundled Dockerfile is now
system-only, so a registry image must also install compatible `letta-evals`
and `@letta-ai/letta-code` application runtimes:

```dockerfile
# Copy the Python, Node >=22.19, git, and compiler setup from the bundled base.
FROM python:3.12-slim
# ... system setup plus project-specific tools ...
RUN python -m pip install --no-cache-dir "letta-evals==0.25.0"
RUN npm install -g --omit=dev "@letta-ai/letta-code@0.30.5"
```

Then push to any registry Modal can reach and set
`sandbox.image: <your-registry>/<your-image>:<tag>` in the suite YAML.

## What the runner uploads per sample

- The entire suite directory tree (`SuiteSpec.base_dir`) →
  `/mnt/suite/`. This covers the YAML, custom Python
  (`agent_setup.py`, `extractors.py`, `graders.py`), rubrics, datasets,
  and any other assets referenced by relative paths in the suite YAML.
- The single `Sample` JSON → `/mnt/sample.json`.

Inside the sandbox, the same `SuiteSpec.from_yaml(..., base_dir=Path("/mnt/suite"))`
loading path runs as on the host, so every relative path in the YAML
resolves to a real file under `/mnt/suite/`.

## What's *not* in scope for v1

- Live in-sandbox progress streaming. The host sees one `SampleResult`
  JSON per sample, not per-step token events.
- Sandbox reuse across samples. Each sample creates and destroys its
  own sandbox; cold-start is ~5–15s.
- A separate sandbox for grading. Target and graders share one sandbox,
  so extractors that read sandbox-filesystem state (e.g. agent memory
  git repos under `~/.letta/agents/<agent_id>/memory`) work without any
  artifact round-trip.
- Built-in reverse tunnels to a `localhost` server. Modal's tunnels are
  inbound-only, so the sandbox can't dial your host's `localhost` directly.
  A self-hosted/private server is still reachable — see
  [Reaching a self-hosted Letta server](#reaching-a-self-hosted-letta-server).

## Reaching a self-hosted Letta server

The sandbox reaches the Letta server over its outbound network
(`block_network: false`, the default), so any endpoint reachable from Modal
works. Set it via `target.base_url` (or the auto-forwarded `LETTA_BASE_URL`;
`target.base_url` wins if both are set):

```yaml
target:
  kind: letta_code
  base_url: https://your-letta-server.example.com
sandbox:
  kind: modal
  letta_evals_version: "0.25.0"
  letta_code_version: "0.30.5"
  block_network: false
```

- **Public URL** (cloud VM / load balancer) → use it directly.
- **Local / private** → expose it with a tunnel and use that URL, e.g.
  `cloudflared tunnel --url http://localhost:9005`; or join the sandbox to a
  mesh VPN ([Modal + Tailscale](https://modal.com/docs/examples/modal_tailscale))
  and use the tailnet address.
- **On Modal** → expose letta-server via `encrypted_ports` and use the tunnel URL.

Modal tunnels are inbound-only — no reverse tunnel to your host's `localhost`,
so a local server needs a tunnel or VPN. The server must also be recent enough
for the image's `@letta-ai/letta-code` CLI; an old one silently fails agent
creation (`No agent_id found in letta stream output`).

## Common failure modes

| Symptom | Likely cause |
|---|---|
| `Modal SDK not found` | Reinstall letta-evals (`pip install letta-evals`); the Modal SDK ships with it. |
| `Modal authentication not found` | Run `modal token new`. |
| `SandboxExecError` with `letta-evals: not found` | The image doesn't install letta-evals on `PATH`. |
| `VersionMismatch` | The image's `letta-evals --version` doesn't match the released version pinned in the YAML. Correct the pin or rebuild a custom image. |
| `ResultDeserializationError` | The in-sandbox CLI exited 0 but didn't write `/mnt/result.json`. Check sandbox stderr in the host run log. |

## Migrating from `target.sandbox` / `target.working_dir`

Both fields were removed when this feature landed. Per-sample isolation
now lives at the suite level instead of the target level.

```yaml
# Before
target:
  kind: letta_code
  working_dir: sandbox
  sandbox: true

# After
target:
  kind: letta_code
sandbox:
  kind: modal
  letta_evals_version: "0.25.0"
  letta_code_version: "0.30.5"
```

The image's `WORKDIR` (set in the Dockerfile) replaces the role of
`working_dir`; per-sample isolation is provided by the sandbox itself.
