# Question Type: Comparison with Tiebreak (Chain-Defined Group)

## Pattern
Define a comparison group through a CHAIN (not independent conditions), then find the max/min with a tiebreak.

## What makes this HARD
- The comparison group is defined by a chain: "people who work at the same company as X", "people who live in the same city as Y"
- Cannot parallelize: must find X first, then their company, then coworkers
- Tiebreak forces checking secondary criterion even when there's a clear winner

## Examples

**Good (chain-defined group):**
- "Among people who work at the same company as the owner of vehicle plate 'XYZ-789', who has the highest bank balance? If tied, who is oldest?"
  - Step 1: vehicles.txt → find owner of plate → `pers-042`
  - Step 2: employments.txt → find their employer → "Tech Corp"
  - Step 3: employments.txt → find ALL employees at Tech Corp → [pers-042, pers-055, pers-087, ...]
  - Step 4: bank_accounts.txt → sum balances per person in that group
  - Step 5: Find max; if tied → people.txt → compare DOB
  
  The comparison group (Tech Corp employees) comes from a chain.

- "Among people who bank at the same institution as the owner of pet 'Whiskers', whose insurance policy expires last? If tied, who has more pets?"
  - Step 1: pets.txt → find owner of pet Whiskers → `pers-012`
  - Step 2: bank_accounts.txt → find their bank → "Chase"
  - Step 3: bank_accounts.txt → find ALL Chase customers → [20 people]
  - Step 4: insurance_policies.txt → find latest expiry date in that group
  - Step 5: If tied → pets.txt → count pets

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
