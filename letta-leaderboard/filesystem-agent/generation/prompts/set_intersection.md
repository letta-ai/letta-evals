# Question Type: Set Intersection (Through Sequential Chain)

## Pattern
Find people who match a condition that ITSELF requires a chain to define. The "set" comes from tracing a relationship, not a simple grep.

## What makes this HARD (vs parallel grep + intersect)
- The set of candidates is defined by a chain, not a simple condition
- Example: "people who live in the same state as X" requires finding X first, then their state, then others in that state
- Intermediate results must be tracked across files

## Examples

**Good (set defined by chain):**
- "Among the pet owners who live in the same state as the owner of vehicle with plate 'XYZ-789', who owns the oldest pet?"
  - Step 1: vehicles.txt → find owner of plate XYZ-789 → `pers-042`
  - Step 2: addresses.txt → find their state → "Texas"
  - Step 3: addresses.txt → find ALL people in Texas → [50 person IDs]
  - Step 4: pets.txt → filter to those who own pets → [12 person IDs]
  - Step 5: pets.txt → among those, find oldest pet → owner is `pers-087`
  - Step 6: people.txt → get name

  The candidate set (Texas residents) cannot be defined until steps 1-2 complete.

- "Among people who bank at the same institution as the owner of vehicle plate 'ABC-123', who has the highest total balance?"
  - Step 1: vehicles.txt → find owner of plate → `pers-055`
  - Step 2: bank_accounts.txt → find their bank → "Chase"
  - Step 3: bank_accounts.txt → find ALL Chase customers → [30 person IDs]
  - Step 4: bank_accounts.txt → sum balance per person → find max
  - Step 5: people.txt → get name

**Bad (parallel greps):**
- "Who has a Mastercard, owns a rabbit, has O+ blood, and lives in Texas?"
  - All 4 conditions can be grepped independently and intersected
  - No condition depends on another condition's result

## Constraints
- Minimum 4 files required
- The candidate SET must be defined through a chain (same state as X, same employer as Y, same bank as Z)
- The filtering criterion must also span multiple files
- Verify intermediate set sizes: the "same X as person Y" group should be 10-50 people
- Final answer must be exactly 1 person

## Common Pitfalls
- All conditions are independent (no chain defines the set)
- Set is trivially small (3 people) making it too easy
- Not tracking intermediate candidates properly
- Using SSN (triggers safety refusals) or "neighbor" (ambiguous)
