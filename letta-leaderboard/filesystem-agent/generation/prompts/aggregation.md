# Question Type: Aggregation (With Sequential Chain)

## Pattern
Find a person through a multi-step chain, THEN aggregate their records. The aggregation target must be discovered through the chain, not given directly.

## What makes this HARD
- The person to aggregate is found through a chain (not "person with plate X")
- Aggregation requires finding ALL records and computing correctly
- Missing one record in the sum changes the answer completely

## Examples

**Good (multi-person aggregation):**
- "What is the COMBINED bank balance of ALL employees at the company where the owner of pet 'Buddy' works?"
  - Step 1: pets.txt → find owner of Buddy → `pers-033`
  - Step 2: employments.txt → find employer → "Acme Corp"
  - Step 3: employments.txt → find ALL Acme Corp employees → [pers-033, pers-055, pers-087, pers-099, pers-112] (5 people)
  - Step 4: bank_accounts.txt → sum balances for EACH of the 5 people
  - Step 5: Sum all 5 totals together
  
  Miss ONE employee = wrong answer. 5 people × multiple accounts each = lots of room for error.

- "What is the TOTAL number of vehicles owned by people who live in the same state as the owner of the cat named 'Whiskers'?"
  - Step 1: pets.txt → find owner of Whiskers → `pers-012`
  - Step 2: addresses.txt → find their state → "Texas"
  - Step 3: addresses.txt → find ALL people in Texas → [~30 people]
  - Step 4: vehicles.txt → count vehicles for EACH of the 30
  - Step 5: Sum all counts
  
  30 people to check, miss one = wrong total.

- "What is the AVERAGE salary of employees at companies where people with Mastercard work?"
  - Step 1: credit_cards.txt → find all Mastercard holders → [~50 people]
  - Step 2: employments.txt → find their employers → [~20 unique companies]
  - Step 3: employments.txt → find ALL employees at those companies → [~100 people]
  - Step 4: Calculate average salary across all ~100

**Bad (single-person aggregation — REJECT):**
- "What is the total balance of person X?" — just one person
- "What is the total balance of the person with the most pets?" — finds ONE person, then sums
- "How many credit cards does person Y have?" — just counting one person

AVOID SSN in questions (triggers safety refusals). Use license plates, usernames, pet names instead.

**Bad (direct aggregation):**
- "What is the total bank balance of the person with plate 'ABC-123'?"
  - Only 2 hops: vehicles → bank_accounts
  - The person is directly identified, no chain needed

## Constraints
- Minimum 5 files required
- Must aggregate across MULTIPLE PEOPLE (3+), not just one person's records
- The group to aggregate should be found through a chain
- Answer must be a specific number (formatted: "$145,315.33" or "7 policies")
- Verify by running SQL: if missing one person changes the answer by >10%, it's a good question

## Key Difficulty Requirement: MULTI-PERSON AGGREGATION
Questions must sum/count across MULTIPLE people, not just one person:

**REQUIRED (multi-person):**
- "What is the COMBINED bank balance of ALL employees at company X?"
- "What is the TOTAL number of pets owned by people in the same city as person Y?"
- "What is the AVERAGE salary of the 5 highest-paid employees at companies with 10+ employees?"

**REJECT (single-person):**
- "What is the total balance of person X?" — just summing one person's accounts
- "How many pets does person Y own?" — just counting one person's records

Multi-person aggregations are hard because missing ONE person ruins the answer.

## Common Pitfalls
- Person is directly identified (no chain)
- Person only has 1-2 records (aggregation is trivial)
- Chain is optional (could skip to direct lookup)
