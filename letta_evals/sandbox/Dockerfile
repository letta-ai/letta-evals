# System base for letta-evals Modal sandboxes.
#
# Application runtimes are deliberately not installed here. The Modal driver
# adds exact letta-evals and @letta-ai/letta-code versions in explicit image
# layers, so changing a suite pin changes the layer definition without
# rebuilding this stable OS/toolchain base.

FROM python:3.12-slim

# git is required by extractors that diff the agent's MemFS git repo.
# build-essential supplies the toolchain needed by node-gyp dependencies.
# Letta Code requires Node >=22.19.0; NodeSource's 22.x channel supplies it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && node -e "const [major, minor] = process.versions.node.split('.').map(Number); \
        if (major < 22 || (major === 22 && minor < 19)) process.exit(1)" \
    && rm -rf /var/lib/apt/lists/*

# Upgrade these before the versioned letta-evals layer. Keep the Sentinel check
# here and after installing letta-evals so an incompatible resolver result fails
# during the image build rather than at sample startup.
RUN python -m pip install --no-cache-dir --upgrade pip "typing_extensions>=4.15.0" \
    && python -c "from typing_extensions import Sentinel"

WORKDIR /work
