# Question Type: Negation / Absence (Chain-Defined Group)

## Pattern
Find a person who does NOT have something, among a group that is DEFINED BY A CHAIN (not a simple grep).

## What makes this HARD
- The group to check is found through a chain: "coworkers of X", "people in the same city as Y"
- Must enumerate the chain-derived group, then check each for absence
- Proving a negative requires exhaustive search within the group

## Examples

**Good (chain-defined group):**
- "Among the coworkers of the owner of vehicle with plate 'ABC-123', who does NOT own any pets?"
  - Step 1: vehicles.txt → find owner of plate ABC-123 → `pers-012`
  - Step 2: employments.txt → find their employer → "Acme Corp"
  - Step 3: employments.txt → find ALL Acme Corp employees → [pers-012, pers-033, pers-055, pers-087]
  - Step 4: vehicles.txt → check each coworker for vehicles
  - Step 5: Find the one with NO vehicles
  
  The group (coworkers) is derived through a chain, not a simple grep.

- "Among people who live in the same state as the owner of pet 'Fluffy', who does NOT have any credit cards?"
  - Step 1: pets.txt → find owner of pet named Fluffy → `pers-045`
  - Step 2: addresses.txt → find their state → "California"
  - Step 3: addresses.txt → find people in California → [30 people]
  - Step 4: credit_cards.txt → filter to those with NO credit cards

  But 30 is too many — narrow further with another condition.

**Bad (simple group):**
- "Among the 5 people with internet accounts on jones.com, who does NOT own pets?"
  - Group is a single grep: internet_accounts.txt → jones.com users
  - No chain required

## Constraints
- Minimum 3 files required
- The group MUST be defined by a chain (coworkers, same city, same bank)
- Group size: 4-8 people (verify with SQL)
- Exactly 1 person in the group should lack the item
- Answer is the person's NAME (never "None" or "no one")

## Common Pitfalls
- Group defined by simple grep (not a chain)
- Multiple people lack the item (ambiguous)
- Group too large to enumerate
- Phrasing answer as negation instead of name
- Using SSN (triggers safety refusals) or "neighbor" (ambiguous)
