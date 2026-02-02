# Question Type: Multi-Hop Chain (TRUE Sequential)

## Pattern
Create a chain where **each step's output is the next step's query input**. The agent CANNOT parallelize these queries — they must complete step N before they can even formulate the query for step N+1.

## What makes this HARD (vs parallelizable queries)
- Step N+1's query depends on step N's result (not just the same person ID)
- Include "indirect relationships" that must be derived: coworker (same employer), same city, same bank
- The chain should be 4-5 hops with at least one hop requiring a derived relationship

## Examples

**Good (multi-candidate intermediate steps):**
- "What is the insurance provider of the employee with the most credit cards at the company where the owner of plate '999-KUZJ' works?"
  - Step 1: vehicles.txt → find owner of plate → `pers-042`
  - Step 2: employments.txt → find employer of pers-042 → "Acme Corp"
  - Step 3: employments.txt → find ALL employees at Acme Corp → [pers-042, pers-055, pers-087, pers-099, pers-112] (5 people)
  - Step 4: credit_cards.txt → count cards for EACH of the 5 → find max
  - Step 5: insurance_policies.txt → get winner's insurer
  
  Step 3 returns MULTIPLE candidates. Step 4 must compare all of them.

- "Among the coworkers of the owner of pet 'Buddy', who has the highest bank balance? What is their employer's name?"
  - Step 1: pets.txt → find owner of Buddy → `pers-045`
  - Step 2: employments.txt → find employer → "Tech Inc"
  - Step 3: employments.txt → find ALL Tech Inc employees → [6 people]
  - Step 4: bank_accounts.txt → sum balances for EACH of the 6 → find max
  - Step 5: Return winner's employer (Tech Inc, but model must verify)

**Bad (single candidate per step):**
- "What pet does the coworker of Morgan Hunter own?"
  - If Morgan's company has only 2 employees, there's only 1 coworker
  - No comparison needed, model just follows breadcrumbs

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
