# Question Type: Temporal Reasoning

## Pattern
Compare dates across different files to answer a question about temporal ordering or duration. The answer requires understanding "before/after", "earliest/latest", or computing time differences.

## What makes this HARD
- Dates appear in different formats across files
- The agent must parse and compare dates, not just string-match
- May require computing duration between two dates

## Examples

**Good (3-4 files):**
- "Among the 4 people with Life insurance policies expiring in 2025, whose policy was issued closest to their date of birth?"
  - insurance_policies.txt -> find Life policies expiring 2025
  - people.txt -> get DOB for each
  - Compare: policy start date vs DOB -> find smallest gap

- "Among people whose credit cards expire in March 2026, who started their current job most recently?"
  - credit_cards.txt -> find people with cards expiring 03/26
  - employments.txt -> get employment start dates
  - Compare: most recent start date

- "Which person started working at their employer BEFORE their earliest insurance policy was issued?"
  - employments.txt -> get start dates
  - insurance_policies.txt -> get earliest policy date per person
  - Compare: employment_start < earliest_policy_date
  - (Narrow to a small group first via another condition)

## Constraints
- Minimum 3 files required
- Questions must involve comparing or ordering dates from different files
- The answer must be a concrete value (person's name, a date, a duration)
- Verify temporal comparisons with SQL (use date functions)
- Narrow the candidate group to 3-8 people before doing temporal comparison

## Common Pitfalls
- Date format inconsistency making comparison ambiguous
- Too many candidates to compare (need a filtering step first)
- Asking about duration without specifying units
- No clear winner in the temporal comparison (ties)
