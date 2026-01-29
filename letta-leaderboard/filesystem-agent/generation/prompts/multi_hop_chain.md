# Question Type: Multi-Hop Chain

## Pattern
Follow a chain of references across 3-4 files. Start with a unique identifier in one file, resolve the person, then look up an attribute in a different file.

## What makes this HARD (not a simple 2-hop)
- Require 3-4 hops minimum, not just "find person -> get attribute"
- Include a filtering step in the middle of the chain (not just at the start)
- The answer should require the LAST hop to be non-trivial (e.g. looking up a specific record among multiple)

## Examples

**Good (3-4 hops):**
- "What is the employer of the person whose pet rabbit is named 'Kenneth' and who has a Mastercard expiring in 02/30?"
  - Hop 1: pets.txt -> find owner of rabbit named Kenneth
  - Hop 2: credit_cards.txt -> verify they have Mastercard expiring 02/30
  - Hop 3: employments.txt -> get their employer

**Bad (too simple, 2 hops):**
- "What pet does the person with license plate XYZ own?"
  - Only 2 hops: vehicles -> people -> pets

## Constraints
- Minimum 3 files required
- Start with a specific, unique identifier (license plate, account number ending, SSN ending, username)
- Each hop must narrow the candidate set
- Verify exactly 1 person matches the full chain
- If asking about pets/employments, ensure the person has exactly 1 of that type (or specify which one)

## Common Pitfalls
- Starting with a condition that matches too many people (e.g. "people with O+ blood type")
- Only requiring 2 hops and calling it multi-hop
- Not verifying the answer is unique
