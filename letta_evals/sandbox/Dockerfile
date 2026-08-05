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
# Pinning LETTA_EVALS_VERSION makes the requested runtime part of Modal's image
# cache key. A plain version installs `letta-evals==<version>`; a direct package
# spec such as `letta-evals @ git+https://...` installs that exact ref for smoke
# tests before a PyPI release exists. When unset, install the latest published
# release.
ARG LETTA_EVALS_VERSION=latest
RUN python -m pip install --no-cache-dir --upgrade pip "typing_extensions>=4.15.0" \
    && if [ "$LETTA_EVALS_VERSION" = "latest" ]; then \
        LETTA_EVALS_PACKAGE="letta-evals"; \
    elif echo "$LETTA_EVALS_VERSION" | grep -Eq '(^letta-evals[[:space:]]*@|^git\+|://)'; then \
        LETTA_EVALS_PACKAGE="$LETTA_EVALS_VERSION"; \
    else \
        LETTA_EVALS_PACKAGE="letta-evals==$LETTA_EVALS_VERSION"; \
    fi \
    && pip install --no-cache-dir --upgrade "$LETTA_EVALS_PACKAGE" \
    && python -c "from typing_extensions import Sentinel"

# letta-code is the agent harness invoked by the LettaCodeTarget.
# Install globally so the `letta` CLI is on $PATH inside the sandbox.
#
# LETTA_CODE_VERSION pins the installed npm version; it defaults to `latest`.
# The Modal driver overrides it from `sandbox.letta_code_version` via
# Image.from_dockerfile(build_args=...). Modal keys its cached image build on
# the build args, so each pinned version resolves to its own cached image.
ARG LETTA_CODE_VERSION=latest
RUN npm install -g "@letta-ai/letta-code@${LETTA_CODE_VERSION}"

WORKDIR /work
