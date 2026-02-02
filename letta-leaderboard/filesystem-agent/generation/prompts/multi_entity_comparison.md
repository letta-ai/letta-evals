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

- "Between the owner of the vehicle with plate 'ABC-123' and the owner of the dog named 'Buddy', who has more bank accounts? If tied, who has the higher total balance?"
  - Chain A: vehicles.txt → find owner of plate → `pers-012` 
  - Chain B: pets.txt → find owner of dog Buddy → `pers-045`
  - Compare: bank_accounts.txt → count for each → if tied, sum balances

## Constraints
- Minimum 4 files required
- EACH chain must be 2+ hops (not just "find person with plate X")
- The two chains must resolve to DIFFERENT people (verify!)
- **CRITICAL: Comparison values should be CLOSE** — within 1-2 of each other for counts, 5% for amounts
- Include a tiebreak: "If tied, who [second criterion]?"
- Design the primary comparison to possibly tie, forcing tiebreak logic
- Answer is the winning person's name
- AVOID SSN in questions (triggers safety refusals). Use license plates, usernames, pet names instead.

## Common Pitfalls
- One or both chains are single-hop lookups (too easy)
- Both chains resolve to the same person
- Comparison has a clear winner with large margin (too easy — values should be close)
- Not verifying the chains are truly independent
- Using SSN which triggers model refusals
