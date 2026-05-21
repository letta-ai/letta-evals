# Base runtime image for letta-evals Modal sandboxes.
#
# This is the image referenced by ModalSandboxSpec's default `image` value
# (ghcr.io/letta-ai/letta-evals-runtime:latest). Suite authors only need
# their own Dockerfile when they need extra system tools the agent invokes
# (e.g. compilers, language toolchains, project-specific binaries).

FROM python:3.12-slim

# git is required by extractors that diff the agent's MemFS git repo
# (e.g. envs/memory_update/extractors.py reading ~/.letta/agents/<id>/memory).
# curl is used to install Node.js via NodeSource; ca-certificates so HTTPS
# certificate verification works for the NodeSource setup script.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# letta-evals provides the in-sandbox `letta-evals run --sample ...` entrypoint.
# letta-client is its Letta SDK dependency.
RUN pip install --no-cache-dir letta-evals letta-client

# letta-code is the agent harness invoked by the LettaCodeTarget.
# Install globally so the `letta` CLI is on $PATH inside the sandbox.
RUN npm install -g @letta-ai/letta-code

WORKDIR /work
