---
name: React Performance Tuning
description: Identify and fix common React performance issues (unnecessary re-renders, expensive computations, list rendering) using profiling and memoization patterns.
license: Proprietary. LICENSE.txt has complete terms
---

# React Performance Tuning

## Overview

React slowness is usually caused by unnecessary renders, expensive render work, or rendering too much UI. The goal is to:
- Reduce render frequency
- Reduce work per render
- Reduce the amount of DOM rendered

## Workflow

1. Reproduce the slowdown and capture a baseline (which interaction is slow).
2. Profile:
   - Use React DevTools Profiler to find components that render often or take long.
3. Identify common causes:
   - Parent state changes re-render large subtrees
   - Inline object/array props break memoization
   - Unstable callback props cause child re-renders
   - Large lists without virtualization
4. Apply fixes:
   - `React.memo` for pure components
   - `useMemo`/`useCallback` for stable props (only when it helps)
   - Lift state appropriately or split components
   - Virtualize big lists
5. Re-profile and verify the change helped.

## Common Pitfalls

- Overusing memoization: it adds complexity and can make things worse.
- Incorrect keys in lists: causes remounts and lost state.
- Passing new objects each render: `style={{...}}`, `{}` in props.

## Checklist

- [ ] Profile before and after
- [ ] Fix unstable props (objects/functions) when causing churn
- [ ] Use correct list keys (stable IDs)
- [ ] Consider virtualization for big lists
