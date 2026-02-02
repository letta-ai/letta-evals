# Question Type: Multi-Entity Comparison (Convergent Chains)

## Pattern
Two PARALLEL chains that each resolve to a person, then a comparison between them. Each chain should be 2-3 hops (not just a single lookup), and both must complete before the comparison.

## What makes this HARD
- Two independent multi-step chains must both succeed
- Each chain has its own sequential dependencies
- The final comparison requires results from BOTH chains
- Models often stop at "tied" instead of applying tiebreaks

## Examples

**Good (two non-trivial chains):**
- "Who has more vehicles: the person with the highest bank balance, or the person with the most pets?"
  - Chain A: bank_accounts.txt → sum per person → find max → `pers-042`
  - Chain B: pets.txt → count per person → find max → `pers-087`
  - Compare: vehicles.txt → count for pers-042 and pers-087 → compare

- "Between the newest employee at 'Johnson LLC' and the owner of the oldest dog, who has more credit cards?"
  - Chain A: employments.txt → find Johnson LLC employees → most recent start_date → `pers-033`
  - Chain B: pets.txt → find dogs → oldest by age → owner is `pers-055`
  - Compare: credit_cards.txt → count for each → determine winner

- "Who lives in a more expensive state (by their rent/address): the person with username 'jdoe' on gmail.com, or the person with SSN ending '9876'?"
  - Chain A: internet_accounts.txt → find jdoe → `pers-012` → addresses.txt → get state
  - Chain B: medical_records.txt → find SSN ending 9876 → `pers-045` → addresses.txt → get state
  - Compare: (would need external knowledge or derive from data patterns)

## Constraints
- Minimum 4 files required
- EACH chain must be 2+ hops (not just "find person with plate X")
- The two chains must resolve to DIFFERENT people (verify!)
- Include a tiebreak: "If tied, who [second criterion]?"
- Design the primary comparison to possibly tie, forcing tiebreak logic
- Answer is the winning person's name

## Common Pitfalls
- One or both chains are single-hop lookups (too easy)
- Both chains resolve to the same person
- Comparison has a clear winner (tiebreak never tested)
- Not verifying the chains are truly independent
