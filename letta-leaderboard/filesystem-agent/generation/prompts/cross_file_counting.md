# Question Type: Cross-File Counting (Chain-Derived Target)

## Pattern
Count records across multiple files for a person/group that is found through a CHAIN, not directly identified.

## What makes this HARD
- The counting target is discovered through a multi-hop chain
- Must count across 3+ different record types
- Missing one file or one record type changes the answer

## Examples

**Good (multi-person cross-file counting):**
- "How many total items (pets + vehicles + credit cards) do ALL employees at the company where the owner of pet 'Buddy' works own COMBINED?"
  - Step 1: pets.txt → find owner of Buddy → `pers-045`
  - Step 2: employments.txt → find their employer → "Tech Inc"
  - Step 3: employments.txt → find ALL Tech Inc employees → [pers-045, pers-087, pers-099, pers-112, pers-156] (5 people)
  - Step 4: For EACH of the 5 employees:
    - pets.txt → count pets
    - vehicles.txt → count vehicles  
    - credit_cards.txt → count cards
  - Step 5: Sum ALL counts across ALL 5 people
  
  5 people × 3 file types = 15 lookups. Miss one = wrong total.

- "What is the TOTAL number of bank accounts held by people who live in the same state as the owner of vehicle plate 'ABC-123'?"
  - Step 1: vehicles.txt → find owner of plate → `pers-042`
  - Step 2: addresses.txt → find their state → "California"
  - Step 3: addresses.txt → find ALL people in California → [~40 people]
  - Step 4: bank_accounts.txt → count accounts for EACH of the 40
  - Step 5: Sum all counts
  
  40 people to check across a single file type.

**Bad (single-person counting — REJECT):**
- "How many total items does the person with plate 'XYZ-123' own?"
  - Just ONE person's records
- "How many items does the highest-paid employee at company X own?"
  - Finds ONE person, then counts their records

## Constraints
- Minimum 5 files required
- Must count across MULTIPLE PEOPLE (5+), not just one person
- The group to count should be found through a chain (same employer, same state, same city)
- Count across at least 2-3 different record types per person
- Answer is a specific number (total across all people)

## Key Difficulty Requirement: MULTI-PERSON COUNTING
Questions must count records across a GROUP of people:
- "How many total X do ALL employees at company Y own?"
- "What is the combined count of A + B + C for people in the same state as Z?"

Single-person counting is too easy — model just needs to find one person and count.
Multi-person counting requires finding ALL members of a group, counting for EACH, summing.

## Common Pitfalls
- Counting for just ONE person (too easy)
- Group too small (2-3 people) — need 5+ people
- Using SSN (triggers safety refusals) or "neighbor" (ambiguous)
