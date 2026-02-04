# Question Type: Temporal Reasoning (Chain-Derived Comparison)

## Pattern
Compare dates across files where the entities/group to compare is found through a CHAIN. The temporal comparison is the final step of a multi-hop journey.

## What makes this HARD
- The dates to compare belong to people/records found through a chain
- Must parse and compare dates correctly (not string matching)
- Close dates (within days/weeks) make comparison error-prone

## Examples

**Good (chain → temporal comparison):**
- "Among the coworkers of the owner of vehicle plate 'ABC-123', whose employment started most recently?"
  - Step 1: vehicles.txt → find owner of plate ABC-123 → `pers-012`
  - Step 2: employments.txt → find their employer → "Tech Corp"
  - Step 3: employments.txt → find ALL Tech Corp employees → [5 people]
  - Step 4: employments.txt → compare start_date for each → find most recent
  - Step 5: people.txt → get name
  
  The group for temporal comparison is derived through a chain.

- "Between the owner of the oldest pet and the owner of the newest vehicle, whose insurance policy expires first?"
  - Chain A: pets.txt → find oldest pet (by age) → owner is `pers-033`
  - Chain B: vehicles.txt → find newest vehicle (by year) → owner is `pers-087`
  - Compare: insurance_policies.txt → compare expiry dates for both

- "Who started their job BEFORE their earliest insurance policy was issued, among people who bank at 'Chase'?"
  - Step 1: bank_accounts.txt → find ALL Chase customers → [20 people]
  - Step 2: For each: employments.txt → get start_date
  - Step 3: For each: insurance_policies.txt → get earliest policy date
  - Step 4: Compare: employment_start < earliest_policy → filter
  - (Narrow the 20 down further if needed)

**Bad (simple temporal):**
- "Among people with Visa cards expiring in 2026, who started their job most recently?"
  - Group is a simple grep (Visa cards expiring 2026)
  - No chain needed

## Constraints
- Minimum 4 files required
- The group/entities for comparison MUST come from a chain
- **CRITICAL: Dates must be CLOSE** — within days or weeks, not months
  - Good: 2025-06-14, 2025-06-17, 2025-06-19 (days apart)
  - Bad: 2024-01-01, 2025-06-01, 2026-12-01 (obvious winner)
- Answer is a person's name or a specific date
- Verify with SQL date comparisons
- AVOID SSN in questions (triggers safety refusals). Use license plates, usernames, pet names instead.

## Common Pitfalls
- Group is a simple grep (not chain-derived)
- Dates are far apart (obvious winner)
- Date format inconsistency causing string vs date comparison issues
- Too many candidates (narrow with a chain first)
