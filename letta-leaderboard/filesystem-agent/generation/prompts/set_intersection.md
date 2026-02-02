# Question Type: Set Intersection (Through Sequential Chain)

## Pattern
Find people who match a condition that ITSELF requires a chain to define. The "set" comes from tracing a relationship, not a simple grep.

## What makes this HARD (vs parallel grep + intersect)
- The set of candidates is defined by a chain, not a simple condition
- Example: "people who live in the same state as X" requires finding X first, then their state, then others in that state
- Intermediate results must be tracked across files

## Examples

**Good (aggregate-then-filter — large intermediate set):**
- "Among the 15 people with the highest total bank balance, who lives in the same state as the owner of pet 'Buddy'?"
  - Step 1: bank_accounts.txt → sum balance per person for ALL 500 people
  - Step 2: Rank and take top 15
  - Step 3: pets.txt → find owner of Buddy → `pers-045`
  - Step 4: addresses.txt → find Buddy owner's state → "California"
  - Step 5: addresses.txt → filter the 15 to those in California → should be 1

  Model must rank 500 people FIRST, then filter. Large intermediate set = more error-prone.

- "Among people who own more than 3 vehicles, who works at the same company as the owner of the cat named 'Whiskers'?"
  - Step 1: vehicles.txt → count per person → filter to those with >3 → [~20 people]
  - Step 2: pets.txt → find owner of Whiskers → `pers-033`
  - Step 3: employments.txt → find their employer → "Acme Corp"
  - Step 4: employments.txt → filter the 20 to Acme Corp employees → should be 1

**Bad (small intermediate set):**
- "Among people in Texas who own dogs, who has the highest balance?"
  - Both conditions are simple greps, no aggregation needed first

**Bad (parallel greps):**
- "Who has a Mastercard, owns a rabbit, has O+ blood, and lives in Texas?"
  - All 4 conditions can be grepped independently and intersected
  - No condition depends on another condition's result

## Constraints
- Minimum 4 files required
- Use AGGREGATE-THEN-FILTER pattern: first create a large set (top N by some metric), then filter
- The initial aggregation should rank/filter across many people (50+)
- The intermediate set should be 10-20 people before final filtering
- Final answer must be exactly 1 person

## Key Difficulty Requirement
The first step should be an AGGREGATION across many records:
- "Among the 15 people with the highest X..."
- "Among people who own more than 3 Y..."
- "Among people whose total Z exceeds $50,000..."
This forces the model to aggregate/rank before filtering, creating more room for error.

## Common Pitfalls
- All conditions are independent (no chain defines the set)
- Set is trivially small (3 people) making it too easy
- Not tracking intermediate candidates properly
- Using SSN (triggers safety refusals) or "neighbor" (ambiguous)
