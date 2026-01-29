# Question Type: Aggregation

## Pattern
Find a person through some chain, then compute an aggregate value across their records in another file. The answer is a computed value (total, sum, count), not a simple lookup.

## What makes this HARD
- The agent must find ALL records for a person in a file and aggregate them
- Requires arithmetic (summing balances, counting records)
- Easy to miss records if the agent stops searching too early

## Examples

**Good:**
- "What is the total bank account balance of the person who owns the vehicle with plate '999-KUZJ'?"
  - vehicles.txt -> find owner
  - people.txt -> confirm person
  - bank_accounts.txt -> find ALL accounts, sum balances

- "How much total salary does the person with SSN ending in 4567 earn across all their jobs?"
  - medical_records.txt -> find person by SSN
  - employments.txt -> find ALL employments, sum salaries

- "What is the combined credit limit of all credit cards held by the person with internet username 'jdoe' on smith.com?"
  - internet_accounts.txt -> find person
  - credit_cards.txt -> find ALL cards, sum limits

## Constraints
- Minimum 3 files required
- The answer MUST be a specific number (e.g. "$145,315.33", "3 accounts")
- Verify the aggregate by running the SQL yourself
- The target person must have 2+ records in the aggregated file (otherwise it's just a lookup)
- Format currency answers with $ and commas

## Common Pitfalls
- Person only has 1 bank account (not really aggregation)
- Not summing ALL records (missing one because of a different owner ID format)
- Ambiguous when person has multiple of the same type and you ask for "the" one
