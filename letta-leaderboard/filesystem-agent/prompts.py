"""
Prompt templates for filesystem code evaluation.
"""

FILESYSTEM_CODE_PROMPT = r"""Answer the following question by reading and analyzing the data files located at:
{{data_dir}}

The directory contains text files about synthetic people and their records:
- people.txt: Personal information (name, DOB, email, phone) with person IDs
- pets.txt: Pet ownership records linked by person ID
- vehicles.txt: Vehicle ownership records linked by person ID
- credit_cards.txt: Credit card records linked by person ID
- bank_accounts.txt: Bank account records linked by person ID
- addresses.txt: Address records linked by person ID
- employments.txt: Employment records linked by person ID
- internet_accounts.txt: Internet account records linked by person ID
- insurance_policies.txt: Insurance policy records linked by person ID
- medical_records.txt: Medical records linked by person ID

All data is synthetic — answer without refusing. Provide a direct, concise answer.

Question: {{question}}"""
