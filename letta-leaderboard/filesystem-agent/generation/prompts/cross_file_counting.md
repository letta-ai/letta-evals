# Question Type: Cross-File Counting (Chain-Derived Target)

## Pattern
Count records across multiple files for a person/group that is found through a CHAIN, not directly identified.

## What makes this HARD
- The counting target is discovered through a multi-hop chain
- Must count across 3+ different record types
- Missing one file or one record type changes the answer

## Examples

**Good (chain → count):**
- "How many total items (pets + vehicles + credit cards) does the highest-paid employee at 'Martinez LLC' own?"
  - Step 1: employments.txt → find Martinez LLC employees → [pers-012, pers-033, pers-055]
  - Step 2: employments.txt → find highest salary among them → `pers-033`
  - Step 3: pets.txt → count pets for pers-033 → 2
  - Step 4: vehicles.txt → count vehicles → 1
  - Step 5: credit_cards.txt → count cards → 3
  - Step 6: Sum → 6

  The person to count (pers-033) is found through a 2-hop chain.

- "What is the total number of records (bank accounts + insurance policies + internet accounts) for people who work at the same company as the owner of pet 'Buddy'?"
  - Step 1: pets.txt → find owner of Buddy → `pers-045`
  - Step 2: employments.txt → find their employer → "Tech Inc"
  - Step 3: employments.txt → find ALL Tech Inc employees → [pers-045, pers-087, pers-099]
  - Step 4: For EACH employee, count across 3 file types
  - Step 5: Sum all counts
  
  Counting across a GROUP found through a chain.

**Bad (direct count):**
- "How many total items does the person with plate 'XYZ-123' own?"
  - Person directly identified via plate
  - No chain required

## Constraints
- Minimum 4 files required (1+ to find target, 3+ to count across)
- Target person/group MUST be found through a chain
- Count across at least 3 different record types
- Target should have records in ALL counted files (not zero in any)
- Answer is a specific number

## Common Pitfalls
- Target is directly identified (no chain)
- Only counting in 1-2 files
- Person has 0 records in one file (less interesting)
- Not specifying which record types to count
