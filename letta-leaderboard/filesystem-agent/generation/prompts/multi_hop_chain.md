# Question Type: Multi-Hop Chain (TRUE Sequential)

## Pattern
Create a chain where **each step's output is the next step's query input**. The agent CANNOT parallelize these queries — they must complete step N before they can even formulate the query for step N+1.

## What makes this HARD (vs parallelizable queries)
- Step N+1's query depends on step N's result (not just the same person ID)
- Include "indirect relationships" that must be derived: coworker (same employer), same city, same bank
- The chain should be 4-5 hops with at least one hop requiring a derived relationship

## Examples

**Good (true sequential):**
- "What is the insurance provider of the newest employee at the company where the owner of license plate '999-KUZJ' works?"
  - Step 1: vehicles.txt → find owner of plate '999-KUZJ' → get `pers-042`
  - Step 2: employments.txt → find employer of pers-042 → get "Acme Corp"
  - Step 3: employments.txt → find ALL employees at "Acme Corp" → get list [pers-042, pers-087, ...]
  - Step 4: employments.txt → find who started most recently → get `pers-087`
  - Step 5: insurance_policies.txt → find pers-087's insurer
  
  You cannot write step 3's query until step 2 returns the employer name.

- "What pet does the coworker of Morgan Hunter own? (Morgan works at a company with exactly 2 employees)"
  - Step 1: people.txt → find Morgan Hunter → get `pers-007`
  - Step 2: employments.txt → find employer of pers-007 → get "Small Corp"
  - Step 3: employments.txt → find OTHER employees at "Small Corp" → get `pers-099`
  - Step 4: pets.txt → find pers-099's pet

  "Coworker" is an indirect relationship — not a field you can grep.

**Bad (parallelizable):**
- "Who has a Mastercard, owns a rabbit, and lives in Texas?"
  - These are 3 independent greps that can run in parallel and intersect
  - No step depends on another step's output

## Constraints
- Minimum 4 files, 4-5 hops
- At least ONE hop must involve an indirect relationship (coworker, same employer, same city, same bank)
- The query for step N+1 must be impossible to write without step N's result
- Start with a unique identifier (plate, username, pet name, email)
- Verify the final answer is unique
- AVOID SSN (triggers safety refusals) and "neighbor" (ambiguous)

## Common Pitfalls
- Making all conditions independent (parallelizable)
- Only using direct ID lookups (no derived relationships)
- Chain where you could theoretically skip a step
