# Question Type: Comparison with Tiebreak (Chain-Defined Group)

## Pattern
Define a comparison group through a CHAIN (not independent conditions), then find the max/min with a tiebreak.

## What makes this HARD
- The comparison group is defined by a chain: "people who work at the same company as X", "people who live in the same city as Y"
- Cannot parallelize: must find X first, then their company, then coworkers
- Tiebreak forces checking secondary criterion even when there's a clear winner

## Examples

**Good (nested superlatives — multiple comparison steps):**
- "Among people with the highest bank balance in their state, who has the fewest credit cards? If tied, who is oldest?"
  - Step 1: addresses.txt + bank_accounts.txt → find highest balance per state → [~50 people, one per state]
  - Step 2: credit_cards.txt → count cards for each of the 50 → find minimum
  - Step 3: If tied → people.txt → compare DOB
  
  TWO superlative operations: "highest per state" then "fewest among those"

- "Among the 10 employees with the longest tenure at companies with 5+ employees, who has the most vehicles? If tied, who has the highest salary?"
  - Step 1: employments.txt → find companies with 5+ employees
  - Step 2: employments.txt → find longest tenure (earliest start) at each → [N people]
  - Step 3: Take top 10 by tenure
  - Step 4: vehicles.txt → count for each of the 10 → find max
  - Step 5: If tied → compare salaries

**Bad (single comparison):**
- "Among people at company X, who has the most pets?"
  - Only ONE comparison step — too easy

**Bad (parallel conditions):**
- "Among people with O+ blood who own dogs, who has the highest salary?"
  - Both conditions (O+ blood, owns dogs) can be grepped in parallel
  - No chain defines the group

## Constraints
- Minimum 4 files required
- The group MUST be defined by a chain: "same X as person Y"
- Group size should be 5-15 people (verify with SQL)
- **CRITICAL: Values must be CLOSE** — top 2-3 candidates should be within 5% of each other
  - For balances: e.g., $45,000 vs $44,500 vs $43,800
  - For dates: within days/weeks of each other
  - For counts: differ by only 1-2
- Always include tiebreak — it forces precise comparison
- AVOID SSN in questions (triggers safety refusals). Use license plates, usernames, pet names instead.

## Common Pitfalls
- Group defined by independent conditions (parallelizable)
- Clear winner (tiebreak never matters)
- Group too small (3 people) or too large (50+ people)
