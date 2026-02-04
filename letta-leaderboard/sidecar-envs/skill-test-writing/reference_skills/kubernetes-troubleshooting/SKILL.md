---
name: Kubernetes Troubleshooting
description: Diagnose and mitigate common Kubernetes issues (CrashLoopBackOff, OOMKilled, probe failures, bad configs) using kubectl and a systematic runbook.
license: Proprietary. LICENSE.txt has complete terms
---

# Kubernetes Troubleshooting

## Overview

Most K8s outages are not “Kubernetes is broken” — they’re misconfigurations, missing dependencies, or resource constraints. The key is to distinguish:
- **The scheduler can’t start it** (ImagePullBackOff, Pending)
- **It starts but crashes** (CrashLoopBackOff)
- **It runs but isn’t ready** (readiness probe failures)

## CrashLoopBackOff Runbook

1. Identify scope:
   - Is it one pod, one node, or all replicas?
2. `kubectl describe pod <pod>`:
   - Look at Events: pull errors, probe failures, OOMKilled, restart count.
3. `kubectl logs <pod> -c <container>`:
   - If it restarted, also check: `kubectl logs --previous`.
4. Check exit reasons:
   - Exit code 137 often implies OOM kill.
   - Exit code 1 is generic; rely on logs and events.
5. Validate configuration:
   - Env vars, ConfigMaps, Secrets, mounted paths, command/args.
6. Probe sanity:
   - Liveness should not kill a slow-starting app.
   - Readiness should reflect “can serve traffic”, not “process is alive”.
7. Mitigate safely:
   - Roll back image, scale down, or disable liveness temporarily (with explicit follow-up).

## Multi-Container Pods

Always specify container names. Many “logs are empty” issues are because you’re reading the sidecar container.

## Common Gotchas

- Readiness depends on a downstream service that’s unavailable → never becomes ready.
- Probes hit the wrong port/path and kill the pod continuously.
- Missing Secret/ConfigMap key: app crashes instantly at startup.
- Resource limits too low → OOM kills under load spikes.

## Checklist

- [ ] Describe pod and read Events
- [ ] Logs + previous logs for the correct container
- [ ] Check exit code / OOMKilled
- [ ] Validate probes and config mounts
- [ ] Apply smallest safe mitigation and verify recovery
