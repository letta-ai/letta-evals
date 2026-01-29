# Question Type: Comparison with Tiebreak

## Pattern
Among a filtered group of people, find who has the highest/lowest value for some attribute. Include a tiebreak criterion in case multiple people share the top value.

## What makes this HARD
- The agent must compare values across multiple people
- Tiebreak adds a second dimension of comparison
- Requires careful reading of multiple records and numerical/date comparison

## Examples

**Good (4-5 files):**
- "Among the people with O+ blood type who own dogs, who has the highest total bank balance? If tied, who is oldest?"
  - medical_records.txt -> O+ blood type people
  - pets.txt -> filter to dog owners
  - bank_accounts.txt -> sum balances per person
  - people.txt -> get DOB for tiebreak

- "Among people who work as engineers and have a Visa card, whose insurance policy expires last? If tied, who has the most pets?"
  - employments.txt -> engineers
  - credit_cards.txt -> Visa holders
  - insurance_policies.txt -> latest expiry
  - pets.txt -> count pets for tiebreak

## Constraints
- Minimum 4 files required
- The filtering step should produce a group of 3-8 people
- The comparison attribute must be quantitative (balance, date, count)
- Include a clear tiebreak: "If tied, who has [second criterion]?"
- Verify there IS actually a unique winner (with or without tiebreak)
- The answer is a person's name

## Common Pitfalls
- Group too large (15+ people makes comparison infeasible with grep)
- No tiebreak needed (clear winner) — still include it for difficulty, it forces the agent to check
- The tiebreak itself is tied (must verify uniqueness through both levels)
- Comparing non-numeric fields
