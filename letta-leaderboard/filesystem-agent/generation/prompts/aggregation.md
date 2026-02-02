# Question Type: Aggregation (With Sequential Chain)

## Pattern
Find a person through a multi-step chain, THEN aggregate their records. The aggregation target must be discovered through the chain, not given directly.

## What makes this HARD
- The person to aggregate is found through a chain (not "person with plate X")
- Aggregation requires finding ALL records and computing correctly
- Missing one record in the sum changes the answer completely

## Examples

**Good (chain → aggregate):**
- "What is the total bank balance of the person who works at the same company as the owner of the dog named 'Buddy'?"
  - Step 1: pets.txt → find dog named Buddy → owner is `pers-033`
  - Step 2: employments.txt → find employer of pers-033 → "Acme Corp"
  - Step 3: employments.txt → find another employee at Acme Corp → `pers-087`
  - Step 4: bank_accounts.txt → find ALL accounts for pers-087 → sum balances
  
  The aggregation target (pers-087) comes from a chain, not directly given.

- "How many total insurance policies do the employees of the company where the owner of pet 'Fluffy' works have combined?"
  - Step 1: pets.txt → find owner of pet Fluffy → `pers-012`
  - Step 2: employments.txt → find their employer → "Tech Inc"
  - Step 3: employments.txt → find ALL employees at Tech Inc → [5 person IDs]
  - Step 4: insurance_policies.txt → count policies for each → sum

  Aggregation across a GROUP found through a chain.

AVOID SSN in questions (triggers safety refusals). Use license plates, usernames, pet names instead.

**Bad (direct aggregation):**
- "What is the total bank balance of the person with plate 'ABC-123'?"
  - Only 2 hops: vehicles → bank_accounts
  - The person is directly identified, no chain needed

## Constraints
- Minimum 4 files required
- The person/group to aggregate MUST be found through a 2+ hop chain
- The person should have 3+ records in the aggregated file
- Answer must be a specific number (formatted: "$145,315.33" or "7 policies")
- Verify by running SQL: if missing one record changes the answer by >10%, it's a good question

## Common Pitfalls
- Person is directly identified (no chain)
- Person only has 1-2 records (aggregation is trivial)
- Chain is optional (could skip to direct lookup)
