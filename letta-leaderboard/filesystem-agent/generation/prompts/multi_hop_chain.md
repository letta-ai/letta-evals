# Question Type: Multi-Hop Chain (TWO PARALLEL CHAINS)

## Pattern
Create TWO independent chains that must BOTH complete, then compare or combine their results. This is the only pattern that consistently fails strong models.

Structure:
1. Chain A: 3-4 hops → finds Person/Value A
2. Chain B: 3-4 hops → finds Person/Value B  
3. Final step: Compare A vs B, or combine A and B

## What makes this HARD (vs parallelizable queries)
- Step N+1's query depends on step N's result (not just the same person ID)
- Include "indirect relationships" that must be derived: coworker (same employer), same city, same bank
- The chain should be 4-5 hops with at least one hop requiring a derived relationship

## Examples

**REQUIRED PATTERN (two parallel chains + compare):**
- "Who owns more vehicles: the person with the highest bank balance in the same state as the owner of pet 'Eduardo', OR the person with the highest bank balance in the same state as the owner of pet 'Jasmine'?"
  - Chain A: pets.txt → owner of Eduardo → addresses.txt → state → bank_accounts.txt → highest balance in that state → Person A
  - Chain B: pets.txt → owner of Jasmine → addresses.txt → state → bank_accounts.txt → highest balance in that state → Person B
  - Compare: vehicles.txt → count for A vs B → return winner's name

- "What is the combined salary of: the person with the most pets among employees at 'Tech Corp', AND the person with the most credit cards among employees at 'Acme Inc'?"
  - Chain A: employments.txt → Tech Corp employees → pets.txt → count per person → find max → get salary
  - Chain B: employments.txt → Acme Inc employees → credit_cards.txt → count per person → find max → get salary
  - Combine: sum both salaries

**BAD (single chain — too easy for strong models):**
- "What is the employer of the person with the most pets at company X?"
  - Only ONE chain, strong models solve this easily

**Bad (parallelizable):**
- "Who has a Mastercard, owns a rabbit, and lives in Texas?"
  - These are 3 independent greps that can run in parallel and intersect
  - No step depends on another step's output

## Constraints
- Minimum 4 files, 5-6 hops
- At least ONE hop must return MULTIPLE candidates (5-15 people) that require comparison
- At least ONE hop must involve an indirect relationship (coworker, same employer, same city, same bank)
- The query for step N+1 must be impossible to write without step N's result
- Start with a unique identifier (plate, username, pet name, email)
- Verify the final answer is unique
- AVOID SSN (triggers safety refusals) and "neighbor" (ambiguous)

## Key Difficulty Requirement
The chain must have a step where the model must COMPARE multiple candidates:
- "Among the N employees at company X, find who has the most Y"
- "Among people in city Z, find who has the highest balance"
This is where models fail — they can follow single-candidate chains but mess up multi-candidate comparisons.

## Common Pitfalls
- Making all conditions independent (parallelizable)
- Only using direct ID lookups (no derived relationships)
- Chain where you could theoretically skip a step
