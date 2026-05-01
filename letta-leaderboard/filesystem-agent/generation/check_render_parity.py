"""Parity check between rendered prose corpus and the source SQLite DB.

For each person, verify that every fact present in the DB rows also appears
verbatim somewhere in the rendered narrative file. We use literal string
containment, which is sufficient because the renderer always writes the raw
field values (no abbreviation, no rewording of identifiers like phone/email/
license plate/etc.).

Reports:
- people with missing facts (per field, per person)
- index files that don't reference an expected person_id
- summary counts

Run:

    python generation/check_render_parity.py
    python generation/check_render_parity.py --memfs /tmp/memfs_smoke --limit 5
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
DB_DEFAULT = HERE / "data" / "letta_file_bench.db"
MEMFS_DEFAULT = HERE.parent / "memfs"


def fetch_all(conn: sqlite3.Connection, sql: str) -> list[dict]:
    return [dict(r) for r in conn.execute(sql).fetchall()]


def check_person(person: dict, body: str, db: dict) -> list[str]:
    """Return a list of 'missing fact' descriptions, empty if all good."""
    pid = person["person_id"]
    misses: list[str] = []

    # identity fields
    for field in ("full_name", "email", "phone"):
        v = person.get(field)
        if v and str(v) not in body:
            misses.append(f"{pid}: missing {field}={v!r}")

    # group child rows by table
    domain_specs = [
        ("addresses", ("street", "city", "state", "postal_code")),
        ("pets", ("name",)),
        ("vehicles", ("license_plate", "make", "model")),
        ("employments", ("employer", "job_title")),
        ("bank_accounts", ("bank_name", "account_no", "routing")),
        ("credit_cards", ("provider", "number", "expire")),
        ("insurance_policies", ("insurer", "policy_number", "policy_type")),
        ("medical_records", ("blood_type", "ssn")),
        ("internet_accounts", ("username", "url")),
    ]
    for table, fields in domain_specs:
        for row in db[table].get(pid, []):
            for field in fields:
                v = row.get(field)
                if v in (None, ""):
                    continue
                if str(v) not in body:
                    misses.append(f"{pid}: missing {table}.{field}={v!r}")
    return misses


def check_indexes(memfs_dir: Path, db: dict, limit_pids: Optional[set[str]] = None) -> list[str]:
    """Sanity check that every person referenced in the DB shows up in the
    appropriate domain indexes."""
    misses: list[str] = []

    # Read all index files once
    index_dir = memfs_dir / "reference" / "indexes"
    idx = {p.name: p.read_text(encoding="utf-8") for p in index_dir.glob("*.md")}

    # Every person should appear in people-by-name
    for p in db["people"]:
        pid = p["person_id"]
        if limit_pids and pid not in limit_pids:
            continue
        link = f"[[reference/people/{pid}.md]]"
        if link not in idx["people-by-name.md"]:
            misses.append(f"{pid}: missing from people-by-name.md")

    # Every owner of a pet should appear in pets-by-name
    pet_owners = {pid for pid in db["pets"].keys()}
    for pid in pet_owners:
        if limit_pids and pid not in limit_pids:
            continue
        link = f"[[reference/people/{pid}.md]]"
        if link not in idx["pets-by-name.md"]:
            misses.append(f"{pid}: missing from pets-by-name.md")

    # Every owner of an address should appear in addresses-by-state
    addr_owners = {pid for pid in db["addresses"].keys()}
    for pid in addr_owners:
        if limit_pids and pid not in limit_pids:
            continue
        link = f"[[reference/people/{pid}.md]]"
        if link not in idx["addresses-by-state.md"]:
            misses.append(f"{pid}: missing from addresses-by-state.md")

    return misses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_DEFAULT)
    parser.add_argument("--memfs", type=Path, default=MEMFS_DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="Check only first N people")
    parser.add_argument("--max-misses", type=int, default=20, help="Print at most this many miss lines")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    people = fetch_all(conn, "SELECT * FROM people ORDER BY person_id")
    if args.limit:
        people = people[: args.limit]
    keep = {p["person_id"] for p in people}

    db: dict = {"people": people}
    for tbl in (
        "addresses", "pets", "vehicles", "employments",
        "bank_accounts", "credit_cards", "insurance_policies",
        "medical_records", "internet_accounts",
    ):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for r in fetch_all(conn, f"SELECT * FROM {tbl}"):
            if r["owner_id"] in keep:
                grouped[r["owner_id"]].append(r)
        db[tbl] = grouped
    conn.close()

    all_misses: list[str] = []
    for p in people:
        path = args.memfs / "reference" / "people" / f"{p['person_id']}.md"
        if not path.exists():
            all_misses.append(f"{p['person_id']}: file not found at {path}")
            continue
        body = path.read_text(encoding="utf-8")
        all_misses.extend(check_person(p, body, db))

    all_misses.extend(check_indexes(args.memfs, db, limit_pids=keep))

    print(f"Checked {len(people)} people. Misses: {len(all_misses)}")
    for line in all_misses[: args.max_misses]:
        print("  " + line)
    if len(all_misses) > args.max_misses:
        print(f"  ... and {len(all_misses) - args.max_misses} more")

    if all_misses:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
