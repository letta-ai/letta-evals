# Question Type: Negation / Absence

## Pattern
Find a person who does NOT have something, among a small, well-defined group. The agent must search a file and confirm the ABSENCE of a record.

## What makes this HARD
- Proving a negative requires exhaustive search within a group
- The agent must check every member of the group, not just find one match
- Easy to make false assumptions about absence

## Examples

**Good:**
- "Among the 3 employees at Palmer and Sons, who does NOT own any vehicles?"
  - employments.txt -> find the 3 employees
  - people.txt -> get their names/IDs
  - vehicles.txt -> check each person, find who has NO vehicles

- "Which of the 4 people with internet accounts on jones.com does NOT have any insurance policies?"
  - internet_accounts.txt -> find the 4 people
  - insurance_policies.txt -> check each, find who has none

## Constraints
- Minimum 3 files required
- The group MUST be small (3-6 people). Verify the group size with SQL first.
- Exactly 1 person in the group should lack the item (verify this!)
- The answer MUST be a concrete value (the person's name), NOT "no one" or "none"
- NEVER create questions where the GT answer is "None", "No one", or "does not own X"
- Frame as "who does NOT have X?" not "does person Y have X?"

## Common Pitfalls
- Group too large (checking 20+ people for absence is infeasible with grep)
- Multiple people lack the item (ambiguous answer)
- Everyone in the group has the item (no valid answer)
- Phrasing the answer as a negation instead of a name
