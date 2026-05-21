FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir letta-evals==0.17.0 letta-client

# Project-specific: install the letta CLI and any other binaries the target
# (and the agent) needs at runtime. The suite itself — agent_setup.py,
# extractors.py, graders.py, rubrics, datasets — is NOT baked into the image;
# the runner uploads it per-sample to /mnt/suite.

WORKDIR /work
