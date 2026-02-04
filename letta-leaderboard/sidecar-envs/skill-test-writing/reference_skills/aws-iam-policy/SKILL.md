---
name: AWS IAM Policy Reasoning
description: Debug AccessDenied errors and design least-privilege access by reasoning about IAM policy evaluation (explicit deny, allow, conditions, resource policies, and role assumption).
license: Proprietary. LICENSE.txt has complete terms
---

# AWS IAM Policy Reasoning

## Overview

IAM authorization is not “does this policy mention the action?” It’s a multi-source evaluation of identity policies, resource policies, permission boundaries, session policies, and organization controls. The core rule is: **any explicit Deny wins**, and an action must be explicitly Allowed somewhere to succeed.

Use this skill to:
- Debug `AccessDenied` errors quickly and safely
- Explain *why* a request is denied (which statement/condition)
- Build least-privilege roles without breaking workflows

## Quick Mental Model

1. **Identify the principal** (user/role + assumed-role session)
2. **Identify the action** (e.g., `s3:GetObject`)
3. **Identify the resource** (ARNs, including region/account and wildcards)
4. **Collect policies that apply**:
   - Identity-based policies (attached to user/role)
   - Resource-based policies (S3 bucket policy, KMS key policy, SNS topic policy, etc.)
   - Permission boundary (if set)
   - Session policy (if using `AssumeRole` with a policy)
   - SCPs (Org policies)
5. Evaluate:
   - If **any Deny matches**, result is Deny
   - Else if **any Allow matches**, result is Allow
   - Else default Deny

## Workflow: Debugging AccessDenied

1. Capture the **exact API call context**:
   - Service + action
   - Resource ARN(s)
   - Region/account
   - Caller identity (role session ARN)
2. Look for **the most common mismatch**:
   - Wrong resource ARN (bucket vs object, table vs index, key vs alias)
   - Missing `kms:Decrypt` (KMS is a frequent hidden dependency)
   - Missing `iam:PassRole` (for ECS/Lambda/Glue/etc.)
   - Condition mismatch (IP, VPC endpoint, MFA, tags, encryption requirement)
3. Determine whether the service uses a **resource policy**:
   - S3, KMS, SNS/SQS, Secrets Manager, ECR, etc.
4. If the principal is assumed-role, check:
   - The **trust policy** allowed the assume
   - The session has the expected **tags** and **duration**
   - Any session policy/boundary is not restricting it
5. Use least-privilege iteration:
   - Start with the minimal action set
   - Restrict resources
   - Add conditions last (and test each)

## Common Pitfalls / Gotchas

- **Bucket vs Object ARNs**: `s3:ListBucket` uses bucket ARN; `s3:GetObject` uses object ARN (`arn:aws:s3:::bucket/key`).
- **KMS requires both sides**: caller needs `kms:Decrypt` and key policy must allow it (or delegate to IAM).
- **Explicit Deny surprises**: a single deny in an SCP or permission boundary overrides all allows.
- **Tag conditions**: `aws:RequestTag`, `aws:PrincipalTag`, and `aws:ResourceTag` often fail because tags aren’t present where you think.
- **Region/account mismatches**: `arn:aws:logs:us-east-1:...` is not `us-west-2`.

## Checklist

- [ ] Confirm caller identity (`sts:GetCallerIdentity`)
- [ ] Confirm action + resource ARN(s)
- [ ] Check for explicit Deny sources (SCP/boundary/resource policy)
- [ ] Check hidden dependencies (KMS, PassRole, VPC endpoint conditions)
- [ ] Narrow allow statements to required actions/resources
- [ ] Re-test after each change
