# Question Type: Cross-File Counting

## Pattern
Count records for a person across multiple different files and combine the counts. The answer is a total count spanning different data domains.

## What makes this HARD
- The agent must search 3+ files for the same person and count records in each
- Requires addition across file boundaries
- Easy to miss one file or double-count

## Examples

**Good (4-5 files):**
- "How many total financial products (bank accounts + credit cards + insurance policies) does the person with license plate 'XYZ-789' own?"
  - vehicles.txt -> find person
  - bank_accounts.txt -> count accounts
  - credit_cards.txt -> count cards
  - insurance_policies.txt -> count policies
  - Sum all counts

- "How many total records (pets + vehicles + internet accounts) does the person with SSN ending 4567 have?"
  - medical_records.txt -> find person by SSN
  - pets.txt -> count pets
  - vehicles.txt -> count vehicles
  - internet_accounts.txt -> count accounts
  - Sum all

## Constraints
- Minimum 4 files required (1 to identify person + 3 to count across)
- Count across at least 3 different record types
- The answer must be a specific number
- Verify the count by running individual SQL queries for each file
- The person should have records in ALL counted files (not zero in any)

## Common Pitfalls
- Person has 0 records in one of the files (less interesting)
- Only counting in 1-2 files (not really cross-file)
- Ambiguous starting identifier
- Not specifying which record types to count (vague "how many records")
