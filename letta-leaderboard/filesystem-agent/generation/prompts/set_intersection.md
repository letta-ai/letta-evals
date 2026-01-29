# Question Type: Set Intersection

## Pattern
Find a person (or people) who satisfy conditions across multiple independent files simultaneously. The agent must search multiple files and intersect the candidate sets.

## What makes this HARD
- Not a linear chain (A -> B -> C). Instead: find set from A, find set from B, find set from C, intersect.
- The agent must track candidates across files and eliminate non-matches
- More files = harder intersection

## Examples

**Good (4-5 files):**
- "Which person has a Mastercard expiring in 2026, owns a pet rabbit, lives in Texas, AND works as an engineer?"
  - credit_cards.txt -> people with Mastercard expiring 2026
  - pets.txt -> people who own rabbits
  - addresses.txt -> people in Texas
  - employments.txt -> people who are engineers
  - Intersect all 4 sets -> should be exactly 1 person

- "Who has both an internet account on smith.com AND a vehicle made by Bailey Inc AND O+ blood type?"
  - internet_accounts.txt -> smith.com users
  - vehicles.txt -> Bailey Inc owners
  - medical_records.txt -> O+ blood type
  - Intersect -> 1 person

## Constraints
- Minimum 4 files required
- Each condition should independently match 5-30 people (not too broad, not too narrow)
- The intersection must yield exactly 1 person
- Verify each individual condition's match count before combining
- The answer should be the person's name or a specific attribute of theirs

## Common Pitfalls
- One condition already uniquely identifies the person (other conditions are irrelevant)
- Conditions are too broad (e.g. "has a credit card" matches everyone)
- Conditions are too narrow (e.g. "born on 1985-03-15" only matches 1 person already)
- Not verifying the intersection is exactly 1
