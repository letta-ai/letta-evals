# Question Type: Multi-Entity Comparison (Convergent Chains)

## Pattern
Two PARALLEL chains that each resolve to a person, then a comparison between them. Each chain should be 2-3 hops (not just a single lookup), and both must complete before the comparison.

## What makes this HARD
- Two independent multi-step chains must both succeed
- Each chain has its own sequential dependencies
- The final comparison requires results from BOTH chains
- Models often stop at "tied" instead of applying tiebreaks

## Examples

**Good (conditional chains — different paths based on condition):**
- "If the owner of plate 'ABC-123' has more credit cards than vehicles, what is their employer? Otherwise, what is their insurance provider?"
  - Step 1: vehicles.txt → find owner of plate → `pers-042`
  - Step 2: credit_cards.txt → count cards for pers-042 → 4
  - Step 3: vehicles.txt → count vehicles for pers-042 → 3
  - Step 4: Compare: 4 > 3, so follow "employer" branch
  - Step 5: employments.txt → get employer
  
  Model must evaluate condition AND follow correct branch.

- "Between the person with the most pets among Chase bank customers and the person with the highest salary at 'Tech Corp', who has more vehicles? If the Chase person has more, what is their blood type? If the Tech Corp person has more, what is their insurance provider?"
  - Chain A: bank_accounts.txt → find Chase customers → pets.txt → count per person → find max
  - Chain B: employments.txt → find Tech Corp employees → find highest salary
  - Compare vehicles → follow correct answer branch

**Good (nested aggregations):**
- "Who has more insurance policies: the person with the highest balance among employees at the company with the most employees, or the person with the most pets among residents of California?"
  - Chain A: employments.txt → count employees per company → find max company → bank_accounts.txt → find highest balance there
  - Chain B: addresses.txt → find California residents → pets.txt → find who has most pets
  - Compare insurance_policies.txt

## Constraints
- Minimum 5 files required
- EACH chain must include an aggregation step (find max/min among a group, not just "find person with X")
- The two chains must resolve to DIFFERENT people (verify!)
- **CRITICAL: Comparison values should be CLOSE** — within 1-2 of each other for counts, 5% for amounts
- PREFERRED: Use conditional branching — "If X has more, return A. Otherwise, return B."
- Answer depends on which branch is taken
- AVOID SSN in questions (triggers safety refusals). Use license plates, usernames, pet names instead.

## Key Difficulty Requirement
Each chain should involve AGGREGATION over a group, not just a single lookup:
- "The person with the most X among Y employees..."
- "The person with the highest balance among Z bank customers..."
Conditional branching adds another layer: model must evaluate AND branch correctly.

## Common Pitfalls
- One or both chains are single-hop lookups (too easy)
- Both chains resolve to the same person
- Comparison has a clear winner with large margin (too easy — values should be close)
- Not verifying the chains are truly independent
- Using SSN which triggers model refusals
