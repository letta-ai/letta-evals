# Question Type: Aggregation (With Sequential Chain)

## Pattern
Find a person through a multi-step chain, THEN aggregate their records. The aggregation target must be discovered through the chain, not given directly.

## What makes this HARD
- The person to aggregate is found through a chain (not "person with plate X")
- Aggregation requires finding ALL records and computing correctly
- Missing one record in the sum changes the answer completely

## Examples

**Good (multi-step aggregation with group selection):**
- "What is the total bank balance of the employee with the most credit cards at the company where the owner of pet 'Buddy' works?"
  - Step 1: pets.txt → find owner of Buddy → `pers-033`
  - Step 2: employments.txt → find employer → "Acme Corp"
  - Step 3: employments.txt → find ALL Acme Corp employees → [8 people]
  - Step 4: credit_cards.txt → count cards for EACH of the 8 → find max
  - Step 5: bank_accounts.txt → sum balances for the winner
  
  TWO aggregations: first "most credit cards among 8", then "sum balances"

- "What is the combined salary of the 3 highest-paid employees at companies with more than 10 employees?"
  - Step 1: employments.txt → count employees per company → filter to >10 employees
  - Step 2: employments.txt → for those companies, find top 3 salaries
  - Step 3: Sum the 3 salaries
  
  Nested aggregation: filter companies, then rank employees, then sum.

- "What is the average vehicle age for people who have the highest bank balance in their city?"
  - Step 1: addresses.txt + bank_accounts.txt → find highest balance per city → [~50 people]
  - Step 2: vehicles.txt → get vehicle years for each of the 50
  - Step 3: Calculate average (current year - vehicle year)

**Bad (simple aggregation):**
- "What is the total balance of person X?"
  - Just one SUM, no intermediate selection step

AVOID SSN in questions (triggers safety refusals). Use license plates, usernames, pet names instead.

**Bad (direct aggregation):**
- "What is the total bank balance of the person with plate 'ABC-123'?"
  - Only 2 hops: vehicles → bank_accounts
  - The person is directly identified, no chain needed

## Constraints
- Minimum 4 files required
- Must include TWO aggregation steps (not just "find person → sum their records"):
  - First: select from a group (e.g., "most credit cards among N employees")
  - Second: aggregate the selected person's records (e.g., "sum their balances")
- The intermediate group should have 5-15 candidates to compare
- Answer must be a specific number (formatted: "$145,315.33" or "7 policies")
- Verify by running SQL: if missing one record changes the answer by >10%, it's a good question

## Key Difficulty Requirement
Questions must have a SELECTION step before the final aggregation:
- "What is the total X of the person with the most Y among Z?"
- "What is the combined A of the top 3 B at company C?"
Single-step aggregations (just summing one person's records) are too easy.

## Common Pitfalls
- Person is directly identified (no chain)
- Person only has 1-2 records (aggregation is trivial)
- Chain is optional (could skip to direct lookup)
