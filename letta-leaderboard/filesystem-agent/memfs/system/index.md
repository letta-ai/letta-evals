---
description: Memory map describing what's in this MemFS and how to navigate it.
---

# Memory map

This memory contains records on 500 synthetic people. Person details live as prose narratives in `reference/people/`. The narrative bodies do not contain machine-readable IDs — instead, the domain indexes under `reference/indexes/` map names, plates, usernames, states, employers, etc. to the matching `[[reference/people/pers-XXXX.md]]` files.

## Available indexes

- **People by name** — [[reference/indexes/people-by-name.md]]
- **Pets by name** — [[reference/indexes/pets-by-name.md]]
- **Vehicles by license plate** — [[reference/indexes/vehicles-by-plate.md]]
- **Internet usernames** — [[reference/indexes/internet-by-username.md]]
- **Addresses by state** — [[reference/indexes/addresses-by-state.md]]
- **Addresses by city** — [[reference/indexes/addresses-by-city.md]]
- **Employers (and coworkers)** — [[reference/indexes/employments-by-employer.md]]
- **Bank customers by bank** — [[reference/indexes/bank-by-bank-name.md]]
- **Insurance policies by provider** — [[reference/indexes/insurance-by-provider.md]]
- **Medical records by blood type** — [[reference/indexes/medical-by-blood-type.md]]

## How to traverse

Most questions chain through indirect relationships ("same state as the owner of X", "same employer as", "same blood type as"). The pattern:

1. Start in the relevant index to find the person whose property you need.
2. Open `reference/people/<id>.md` to read their prose narrative.
3. Use a property from that narrative (state, employer, blood type, etc.) to look up other people via the matching index.
4. Open as many `reference/people/<id>.md` files as needed to compute the answer.

The indexes are designed to be **grepped**, not loaded whole. Target the key you need — a specific pet name, plate, state heading — rather than reading the full index file. Likewise, only open the person files you need.
