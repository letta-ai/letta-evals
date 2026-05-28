# Base runtime image for letta-evals Modal sandboxes.
#
# When a suite sets `sandbox: { kind: modal }` without an `image`, the driver
# builds this Dockerfile on demand via Modal's Image.from_dockerfile (the build
# is cached, so only the first sandbox after an edit pays the build cost).
# Suite authors only need their own image when they need extra system tools the
# agent invokes (e.g. compilers, language toolchains, project-specific binaries).

FROM python:3.12-slim

# git is required by extractors that diff the agent's MemFS git repo
# (e.g. envs/memory_update/extractors.py reading ~/.letta/agents/<id>/memory).
# curl is used to install Node.js via NodeSource; ca-certificates so HTTPS
# certificate verification works for the NodeSource setup script.
# build-essential supplies make + a C/C++ toolchain so node-gyp can compile
# @letta-ai/letta-code's native dependency (node-pty) during npm install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# letta-evals provides the in-sandbox `letta-evals run --sample ...` entrypoint.
# letta-client is its Letta SDK dependency.
RUN pip install --no-cache-dir --upgrade "letta-evals>=0.19.0" letta-client

# letta-code is the agent harness invoked by the LettaCodeTarget.
# Install globally so the `letta` CLI is on $PATH inside the sandbox.
RUN npm install -g @letta-ai/letta-code

WORKDIR /work
