# Question Type: Multi-Entity Comparison

## Pattern
Identify two specific people through independent chains, then compare an attribute between them. The answer states which person "wins" the comparison.

## What makes this HARD
- Two independent resolution chains must both succeed
- The agent must track two people simultaneously
- Requires comparing values between two separate entities

## Examples

**Good (4-5 files):**
- "Between the owner of vehicle with plate 'ABC-123' and the person with SSN ending 4567, who has more credit cards?"
  - Chain 1: vehicles.txt -> find owner of plate ABC-123
  - Chain 2: medical_records.txt -> find person with SSN ending 4567
  - Compare: credit_cards.txt -> count cards for each person

- "Between the person with internet username 'jdoe' on smith.com and the employee at Palmer and Sons with the highest salary, who owns more pets?"
  - Chain 1: internet_accounts.txt -> find jdoe
  - Chain 2: employments.txt -> find Palmer and Sons employees -> highest salary
  - Compare: pets.txt -> count pets for each

## Constraints
- Minimum 4 files required
- Each person must be identifiable through a unique chain (not ambiguous)
- The two people must be DIFFERENT people (verify!)
- The comparison attribute must have a clear winner (not tied, or specify tiebreak)
- The answer should be the winning person's name
- Frame naturally: "Between X and Y, who has more Z?"

## Common Pitfalls
- Both chains resolve to the same person
- The comparison is tied with no tiebreak
- One chain is ambiguous (matches multiple people)
- Overly complex chains that make the question unreadable
