# Filesystem Suite: Data Quality Audit

Audit of the 10 synthetic data files in `files/` and their interaction with the 100 eval questions in `datasets/filesystem_code.jsonl`.

## Summary

| File | Records | Owners | Multi-Record Owners | Issues Found |
|------|---------|--------|---------------------|-------------|
| people.txt | 500 | - | - | Phone numbers with extensions |
| employments.txt | 522 | 342 | 53% (2 jobs) | Truncated job titles, ambiguous questions |
| pets.txt | 957 | 398 | 72% (2-4 pets) | Same-species duplicates, ambiguous questions |
| vehicles.txt | 964 | 377 | 79% (2-4 vehicles) | Fake makes/models (Faker artifacts) |
| bank_accounts.txt | 1,762 | 500 | 86% (2-6 accounts) | None (questions handle this well) |
| credit_cards.txt | 1,267 | 413 | 82% (2-5 cards) | Same-provider duplicates |
| addresses.txt | 1,017 | ~500 | 35% (2-3 addresses) | None |
| insurance_policies.txt | 975 | ~500 | 30% (2-4 policies) | None |
| internet_accounts.txt | 1,005 | ~500 | 30% (2-4 accounts) | None |
| medical_records.txt | 255 | 255 | 0% (1:1) | None |

## Critical Issues

### 1. Truncated Job Titles (employments.txt)

Faker sometimes produces truncated job titles. These are single words that should be longer:

| Record | Value | Likely Intended |
|--------|-------|-----------------|
| emp-0041 | `Copy` | `Copywriter` or `Copy editor` |
| emp-0238 | `Land` | `Land surveyor` or `Land agent` |
| emp-0242 | `Make` | `Make-up artist` |
| emp-0383 | `Make` | `Make-up artist` |
| emp-0441 | `Land` | `Land surveyor` or `Land agent` |

**Impact:** If a question asks "what is the job title of person X" and the ground truth is `Copy`, an agent might struggle to find it or consider it an error.

### 2. Fake Vehicle Makes and Models (vehicles.txt)

66% of vehicles have **company names** as makes (e.g., "Johnson LLC", "Holt, Nguyen and Jones") instead of real manufacturers. 88% have **color names** as models (e.g., "Fuchsia", "Salmon", "Turquoise").

**Impact:** Questions referencing vehicle makes/models use these fake values. Agents must work with the data as-is, but it reduces realism.

### 3. Question Ambiguity from Multi-Record Ownership

Many people own multiple records of the same type. When questions use singular phrasing ("the dog", "the job title"), the ground truth picks one answer when multiple are valid.

**11 of 100 questions are ambiguous** (2 employment, 9 pet):

| Q# | Question (truncated) | GT | Issue |
|----|---------------------|-----|-------|
| 0 | Name of rabbit owned by earliest AmEx holder | Kenneth | Person has 2 rabbits (also Michael) |
| 3 | Name of bird for username 'qstewart' | Monica | Person has 2 birds (also Anthony) |
| 7 | Job title of earliest JCB 16 holder | Musician | Person also has job: Intelligence analyst |
| 14 | Job title of lowest balance person | Midwife | Person also has job: Secondary school teacher |
| 28 | Name of pet for account ending 7299 | Katherine | Person has 3 pets (also Brent, Tammy) |
| 46 | Breed of dog for password holder | Mixed | Person has 2 of same species with different breeds |
| 72 | Name of dog owned by airline pilot | Benjamin | Person has 3 dogs (also Theresa, Lawrence) |
| 77 | Name of cat for 'chenrebecca' | Christine | Person has 2+ cats (also Marc, Laura) |
| 78 | Name of rabbit in Hornstad | Kenneth | Person has 2 rabbits (also Michael) |
| 83 | Name of fish for "555" phone holder | Christopher | Person has 2+ fish (also Emily, Joseph) |
| 89 | Name of cat for engineer A- blood | Jeremy | Person has 2 cats (also Scott, Louis) |

**Well-designed (no ambiguity):**
- vehicles.txt questions use license plates as unique identifiers
- bank_accounts.txt questions use account number endings, totals, or highest/lowest
- medical_records.txt has 1:1 mapping (one record per person)

### 4. Phone Numbers with Extensions (people.txt)

89 of 500 records (18%) have phone numbers with extensions (e.g., `001-401-452-1516x9201`), resulting in 16-18 digit strings. These are technically valid but exceed the typical 15-digit international limit.

**Impact:** Low. The `x` extension separator makes them parseable, but agents doing digit-count validation might flag them.

## Non-Issues (Well-Designed)

- **bank_accounts.txt:** Questions use account number endings, total balances, or highest/lowest — all unambiguous despite 86% multi-account ownership.
- **vehicles.txt:** Questions use license plates as unique identifiers — no ambiguity despite 79% multi-vehicle ownership.
- **medical_records.txt:** 1:1 mapping (one record per person), no ambiguity possible.
- **addresses.txt, insurance_policies.txt, internet_accounts.txt:** Generally clean data, questions handle multi-record cases reasonably.

## Recommendations

1. **Fix truncated job titles** — Replace `Copy`, `Land`, `Make` with full job titles
2. **Audit ambiguous questions** — For each question referencing employments or pets, verify the ground truth accounts for all valid answers (or rephrase to be unambiguous)
3. **Consider vehicle data regen** — Replace fake Faker company/color names with real vehicle makes/models
4. **Accept phone extensions** — Low impact, no action needed

---

## Failure Analysis from Eval Results

Analyzed 1,947 result entries across 99 questions and 9 models (claude-opus-4.5, claude-sonnet-4.5, claude-haiku-4.5, gpt-5.2-xhigh, gpt-5.2-codex-high, gemini-3, gemini-3-flash, fudge, minimax-m2.1).

### Overview

| Category | Count |
|----------|-------|
| Always pass (all models correct) | 27 |
| Sometimes fail | 63 |
| Always fail (ALL models wrong) | 9 |

### 9 Always-Fail Questions (100% failure, every model, every run)

When ALL models get a question wrong, it's almost certainly a question/data issue, not a model issue.

**Likely Ground Truth Errors (6 questions):**

| Sample | Question | GT | Evidence |
|--------|----------|-----|---------|
| 44 | Among the 3 structural engineers, whose credit card expires first? | Timothy Nelson | All models find different answer |
| 31 | Among people with hypertension who own fish, whose credit card expires first? | Diane Thompson | All models find different answer |
| 45 | Among people whose credit cards expire in 09/27, who owns the oldest vehicle? | Teresa Ayala | Models say Steven Cook or others |
| 68 | Among people who own 2025 vehicles, who has the most pets? | Alexander Ramos | Models find ties (multiple people with 4 pets) |
| 93 | How many bank accounts does the person with earliest expiring policy on July 2nd, 2026 have? | 2 | All models say 1 |
| 96 | Among people on smith.com, who owns a bird and lives at lowest postal code? | John Jenkins | All models say David Lowe |

These 6 questions need their ground truths re-verified against the database.

**Already-Identified Ambiguous Questions (2):**

| Sample | Question | GT | Issue |
|--------|----------|-----|-------|
| 7 | Job title of earliest JCB 16 digit credit card holder? | Musician | Person has 2 jobs (also Intelligence analyst) |
| 77 | Name of cat for 'chenrebecca' on Lopez Banks? | Christine | Person has 2+ cats |

**Trick/Negation Question (1):**

| Sample | Question | GT | Issue |
|--------|----------|-----|-------|
| 66 | Breed of rabbit for person with '777' in bank acct and most pets? | Amy Marsh does not own a rabbit | GT is a prose negation — hard for model judge to grade |

### High Failure Rate Questions (>50% fail, not always-fail)

| Sample | Fail Rate | Question | GT | Likely Issue |
|--------|-----------|---------|-----|-------------|
| 43 | 95% | Credit cards expire 01/26, who owns newest vehicle? | Gregory Cook | Probable GT error |
| 30 | 95% | Vehicle from 1995 + rabbit, whose credit card expires first? | Melinda Smith | All models say Lori Coleman |
| 63 | 89% | Credit card provider for Alexander-Williams policy holder? | American Express | Models find different provider |
| 55 | 84% | Employees at Smith PLC, which pet is older: Billy or Ricardo? | Both same age (no age recorded) | Unanswerable — no age field exists in data |
| 56 | 63% | Vehicle make for life insurance expiring 2025-06-19? | Daniels, Vargas and Stout | Person has 2 vehicles, GT only lists one |
| 69 | 63% | Pet name for highest balance among credit card expiring 12/26? | Gregory Luna does not own any pets | Negation GT — hard to grade |
| 70 | 58% | Medical condition for highest balance among cards expiring 09/25? | None | "None" as GT is hard to match |

### Key Takeaways

1. **~15 questions have likely-wrong ground truths** — these inflate failure rates and make the benchmark unreliable. They need SQL re-verification.
2. **Negation/absence answers ("does not own", "None") are hard to grade** — the model judge struggles when GT is a prose statement rather than a concrete value.
3. **The 11 ambiguous questions from our earlier audit overlap with high-failure questions** — confirming that ambiguity is a real eval problem, not just theoretical.
4. **Effective question count is ~74 reliable questions** (99 total - 15 GT errors - ~11 ambiguous + some overlap).

---

## Increasing Question Difficulty

Models currently score ~83%. With ~11% of questions ambiguous, the effective ceiling is ~89%. The questions are too easy.

### Current Question Profile

| Dimension | Current Distribution | Problem |
|-----------|---------------------|---------|
| Question type | 52% compositional, 18% factual, 17% comparison, 9% logical | Overwhelmingly "follow the chain" lookups |
| Difficulty | 89% medium, 9% easy, 2% hard | Almost nothing is hard |
| Files required | 71% need 3 files, 21% need 4, 8% need 2 | Rarely exceeds 3-file chains |

The generator prompt and rubric actively discourage hard questions:
- Rubric REJECTs "joining data from 3+ different files"
- Prompt says "70% simpler patterns, 20% moderate, 10% complex"
- Prompt says "Keep most questions to 2-3 file lookups"

### Proposed New Question Types

**1. Aggregation** — collect and compute across multiple records

> "What is the total bank balance of the person who owns the vehicle with plate '999-KUZJ'?"

Requires finding ALL bank accounts for one person and summing. Current questions mostly ask for a single value.

**2. Set intersection** — find entity matching conditions across multiple files simultaneously

> "Which person has both a Mastercard expiring in 2026 AND a pet rabbit AND lives in Texas?"

Agent must search 4 files, intersect candidate sets. Not a linear chain — requires parallel filtering.

**3. Negation / absence** — verify something does NOT exist

> "Among the 3 employees at Stone Ltd, who does NOT own any pets?"

Agent must search pets.txt for each employee and confirm absence. Proving a negative is fundamentally harder than finding a positive.

**4. Ranking with tiebreakers** — superlative with a secondary sort

> "Among people with O+ blood type who own dogs, who has the highest salary? If tied, who is older?"

Requires aggregating candidates, comparing values, possibly resolving ties — 4-5 file lookups plus comparison logic.

**5. Multi-entity comparison** — compare attributes across two specific people

> "Between the owner of vehicle plate 'ABC-123' and the person with SSN ending 4567, who has more credit cards?"

Two independent resolution chains that must both complete, then compare.

**6. Conditional chain** — the path depends on an intermediate result

> "If the person with internet username 'jdoe' has a dog, what is its name? If they have a cat instead, what breed is it?"

Agent must first determine the species, then ask the right follow-up question. Tests adaptive reasoning, not just retrieval.

**7. Temporal reasoning** — compare dates across different files

> "Was the person's insurance policy issued before or after they started their current job? How many months apart?"

Requires parsing and comparing dates across files, doing date math.

**8. Cross-file counting** — count records matching a condition across multiple domains

> "How many distinct financial products (bank accounts + credit cards + insurance policies) does the person with plate 'XYZ-789' have?"

Agent must search 3 files, count records per file, sum. Current questions rarely require counting across files.

### Recommended Changes to Generator

1. **Update rubric** — Stop rejecting 3+ file joins. Add "Hard" tier for 4-5 file lookups with conditional logic.
2. **Update system prompt** — Shift from "70/20/10 easy/moderate/complex" to "20/50/30". Raise file requirement from "mostly 2-3" to "3-5".
3. **Add ambiguity guard** — In `register_question_tool.py`, verify the final SQL returns exactly 1 answer before accepting.
4. **Raise `min_sql_queries_per_question`** from 2 to 3 in config.
5. **Wire question type distribution into the generator** — Currently the config percentages aren't used. Pass the requested type into the user message so the agent targets specific patterns.
