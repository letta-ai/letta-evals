"""Render the synthetic-people SQLite DB as a prose-style MemFS tree.

Emits:

    <out>/system/index.md                       # always-visible navigation map
    <out>/reference/indexes/<domain>.md         # name/key -> person-file lookup tables
    <out>/reference/people/pers-XXXX.md         # one prose narrative per person

The narrative bodies do **not** contain `pers-XXXX` style IDs; those identifiers
survive only as filenames and as the targets of `[[reference/people/...]]` links
inside the index files. This is deliberate: it means a Bash regex over the
prose corpus can't reliably count or join across people without first walking
the indexes, so the eval rewards models that follow the in-context map rather
than ones that scrape everything.

Run from anywhere — paths are anchored relative to this file:

    python generation/render_memfs.py                              # full corpus -> ../memfs
    python generation/render_memfs.py --limit 5 --out /tmp/sample  # smoke render
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).parent
DB_DEFAULT = HERE / "data" / "letta_file_bench.db"
OUT_DEFAULT = HERE.parent / "memfs"

MONTH = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------- formatting helpers ----------


def fmt_date(iso: Optional[str]) -> str:
    """`YYYY-MM-DD` -> `Month D, YYYY`. Pass-through on parse failure."""
    if not iso:
        return ""
    try:
        d = datetime.strptime(iso, "%Y-%m-%d")
    except ValueError:
        return iso
    return f"{MONTH[d.month - 1]} {d.day}, {d.year}"


def fmt_money(amount: Optional[float], currency: str = "USD") -> str:
    if amount is None:
        return ""
    if currency == "USD":
        return f"${amount:,.2f}"
    return f"{amount:,.2f} {currency}"


def first_name(full: str) -> str:
    return full.split()[0] if full else ""


def join_oxford(items: list[str]) -> str:
    """`['a','b','c']` -> `'a, b, and c'`. Empty -> ''."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def pluralize(n: int, singular: str, plural: Optional[str] = None) -> str:
    if n == 1:
        return f"1 {singular}"
    return f"{n} {plural or singular + 's'}"


def aan(word: str) -> str:
    """Pick "a" or "an" based on the first letter of the next word."""
    if not word:
        return "a"
    return "an" if word.lstrip()[:1].lower() in "aeiou" else "a"


# ---------- DB load ----------


def load_db(db_path: Path) -> dict:
    """Load all rows for all tables into dicts grouped by person.

    Returns a dict with keys: people (list), and per-domain dict
    {person_id: [rows]} for each child table.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    def fetch_all(sql: str) -> list[dict]:
        return [dict(row) for row in conn.execute(sql).fetchall()]

    people = fetch_all("SELECT * FROM people ORDER BY person_id")

    def group_by_owner(table: str, order_by: str) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in fetch_all(f"SELECT * FROM {table} ORDER BY {order_by}"):
            grouped[row["owner_id"]].append(row)
        return grouped

    out = {
        "people": people,
        "addresses": group_by_owner("addresses", "address_id"),
        "pets": group_by_owner("pets", "pet_id"),
        "vehicles": group_by_owner("vehicles", "vehicle_id"),
        "employments": group_by_owner("employments", "employment_id"),
        "bank_accounts": group_by_owner("bank_accounts", "account_id"),
        "credit_cards": group_by_owner("credit_cards", "card_id"),
        "insurance_policies": group_by_owner("insurance_policies", "policy_id"),
        "medical_records": group_by_owner("medical_records", "record_id"),
        "internet_accounts": group_by_owner("internet_accounts", "account_id"),
    }
    conn.close()
    return out


# ---------- per-person prose ----------


def render_person(person: dict, db: dict) -> tuple[str, str]:
    """Return (frontmatter_description, body_markdown) for one person.

    Body sections appear in a stable order. Each section is omitted when the
    person has no rows in that domain. No `pers-XXXX` IDs leak into the body.
    """
    pid = person["person_id"]
    name = person["full_name"]
    fn = first_name(name)

    addrs = db["addresses"].get(pid, [])
    pets = db["pets"].get(pid, [])
    vehs = db["vehicles"].get(pid, [])
    emps = db["employments"].get(pid, [])
    banks = db["bank_accounts"].get(pid, [])
    cards = db["credit_cards"].get(pid, [])
    ins = db["insurance_policies"].get(pid, [])
    meds = db["medical_records"].get(pid, [])
    nets = db["internet_accounts"].get(pid, [])

    # --- frontmatter description (terse, ~100 chars target) ---
    states = sorted({a["state"] for a in addrs if a.get("state")})
    qualifiers: list[str] = []
    if person.get("dob"):
        qualifiers.append(f"b. {person['dob']}")
    if states:
        qualifiers.append(", ".join(states))
    if emps:
        employers = sorted({e["employer"] for e in emps if e.get("employer")})
        if len(employers) == 1:
            qualifiers.append(f"employed at {employers[0]}")
        elif len(employers) == 2:
            qualifiers.append(f"employed at {employers[0]} and {employers[1]}")
        elif employers:
            qualifiers.append(f"employed at {len(employers)} employers")
    counts = []
    if pets:
        counts.append(pluralize(len(pets), "pet"))
    if vehs:
        counts.append(pluralize(len(vehs), "vehicle"))
    if cards:
        counts.append(pluralize(len(cards), "card"))
    if counts:
        qualifiers.append("; ".join(counts))
    description = name if not qualifiers else f"{name} — " + "; ".join(qualifiers)

    # --- body sections ---
    parts: list[str] = [f"# {name}", ""]

    # identity line
    id_bits: list[str] = []
    if person.get("dob"):
        id_bits.append(f"was born on {fmt_date(person['dob'])}")
    contact: list[str] = []
    if person.get("email"):
        contact.append(f"by email at {person['email']}")
    if person.get("phone"):
        contact.append(f"by phone at {person['phone']}")
    if contact:
        id_bits.append("can be reached " + " or ".join(contact))
    if id_bits:
        parts.append(f"{name} {join_oxford(id_bits)}.")
        parts.append("")

    # addresses
    if addrs:
        parts.append(f"## Where {fn} lives")
        parts.append("")
        if len(addrs) == 1:
            a = addrs[0]
            line = (
                f"{fn} lives at {a['street']} in {a['city']}, {a['state']} "
                f"({a['postal_code']}, {a['country']})."
            )
            parts.append(line)
        else:
            parts.append(f"{fn} keeps {pluralize(len(addrs), 'address', 'addresses')}:")
            parts.append("")
            for a in addrs:
                parts.append(
                    f"- {a['street']} in {a['city']}, {a['state']} "
                    f"({a['postal_code']}, {a['country']})"
                )
        parts.append("")
        parts.append(
            f"To find others living in the same state, see "
            f"[[reference/indexes/addresses-by-state.md]]."
        )
        parts.append("")

    # pets
    if pets:
        parts.append("## Pets")
        parts.append("")
        if len(pets) == 1:
            pt = pets[0]
            breed = (pt.get("breed") or "").strip().lower()
            species = (pt.get("species") or "").strip().lower()
            head = breed or species
            descriptor = f"{aan(head)} {breed} {species}".strip() if breed else f"{aan(species)} {species}"
            descriptor = " ".join(descriptor.split())
            parts.append(f"{fn} has one pet, {descriptor} named {pt['name']}.")
        else:
            parts.append(f"{fn} has {pluralize(len(pets), 'pet')}:")
            parts.append("")
            for pt in pets:
                breed = (pt.get("breed") or "").strip().lower()
                species = (pt.get("species") or "").strip().lower()
                head = breed or species
                if breed:
                    parts.append(f"- {pt['name']}, {aan(head)} {breed} {species}")
                else:
                    parts.append(f"- {pt['name']}, {aan(species)} {species}")
        parts.append("")

    # vehicles
    if vehs:
        parts.append("## Vehicles")
        parts.append("")
        if len(vehs) == 1:
            v = vehs[0]
            line = f"{fn} owns one vehicle: {aan(str(v['year']))} {v['year']} {v['make']} {v['model']}"
            if v.get("license_plate"):
                line += f" (license plate {v['license_plate']})"
            line += "."
            parts.append(line)
        else:
            parts.append(f"{fn} owns {pluralize(len(vehs), 'vehicle')}:")
            parts.append("")
            for v in vehs:
                line = f"- {aan(str(v['year']))} {v['year']} {v['make']} {v['model']}"
                if v.get("license_plate"):
                    line += f" (license plate {v['license_plate']})"
                parts.append(line)
        parts.append("")

    # employment
    if emps:
        parts.append("## Work")
        parts.append("")
        if len(emps) == 1:
            e = emps[0]
            jt = e['job_title']
            parts.append(
                f"{fn} works as {aan(jt)} {jt} at {e['employer']}, since "
                f"{fmt_date(e['start_date'])}, earning {fmt_money(e['salary'], e.get('currency') or 'USD')} per year."
            )
        else:
            parts.append(f"{fn} has {pluralize(len(emps), 'employment record')}:")
            parts.append("")
            for e in emps:
                parts.append(
                    f"- {e['job_title']} at {e['employer']} since "
                    f"{fmt_date(e['start_date'])}, "
                    f"{fmt_money(e['salary'], e.get('currency') or 'USD')} per year"
                )
        parts.append("")
        parts.append(
            "For coworkers at the same employer, see "
            "[[reference/indexes/employments-by-employer.md]]."
        )
        parts.append("")

    # banking
    if banks:
        parts.append("## Banking")
        parts.append("")
        if len(banks) == 1:
            b = banks[0]
            parts.append(
                f"{fn} banks with {b['bank_name']} (account {b['account_no']}, "
                f"routing {b['routing']}). Current balance: "
                f"{fmt_money(b['balance'], b.get('currency') or 'USD')}."
            )
        else:
            total = sum(b["balance"] or 0 for b in banks)
            currency = banks[0].get("currency") or "USD"
            parts.append(
                f"{fn} holds {pluralize(len(banks), 'bank account')} "
                f"with a combined balance of {fmt_money(total, currency)}:"
            )
            parts.append("")
            for b in banks:
                parts.append(
                    f"- {b['bank_name']} account {b['account_no']} "
                    f"(routing {b['routing']}), balance "
                    f"{fmt_money(b['balance'], b.get('currency') or 'USD')}"
                )
        parts.append("")

    # credit cards
    if cards:
        parts.append("## Credit cards")
        parts.append("")
        if len(cards) == 1:
            c = cards[0]
            parts.append(
                f"{fn} carries one credit card: {aan(c['provider'])} {c['provider']} "
                f"({c['number']}, expires {c['expire']})."
            )
        else:
            parts.append(f"{fn} carries {pluralize(len(cards), 'credit card')}:")
            parts.append("")
            for c in cards:
                parts.append(
                    f"- {c['provider']} ({c['number']}, expires {c['expire']})"
                )
        parts.append("")

    # insurance
    if ins:
        parts.append("## Insurance")
        parts.append("")
        if len(ins) == 1:
            p = ins[0]
            pt = p['policy_type']
            parts.append(
                f"{fn} holds {aan(pt)} {pt} policy with {p['insurer']} "
                f"(policy number {p['policy_number']}), expiring "
                f"{fmt_date(p['expires'])}."
            )
        else:
            parts.append(
                f"{fn} holds {pluralize(len(ins), 'insurance policy', 'insurance policies')}:"
            )
            parts.append("")
            for p in ins:
                parts.append(
                    f"- {p['policy_type']} with {p['insurer']} "
                    f"(policy number {p['policy_number']}), expires "
                    f"{fmt_date(p['expires'])}"
                )
        parts.append("")

    # medical
    if meds:
        parts.append("## Medical")
        parts.append("")
        if len(meds) == 1:
            m = meds[0]
            bits: list[str] = []
            if m.get("blood_type"):
                bits.append(f"blood type {m['blood_type']}")
            if m.get("condition"):
                bits.append(f"condition: {m['condition']}")
            if m.get("ssn"):
                bits.append(f"SSN {m['ssn']}")
            parts.append(f"{fn}'s medical record notes " + "; ".join(bits) + ".")
        else:
            parts.append(f"{fn} has {pluralize(len(meds), 'medical record')}:")
            parts.append("")
            for m in meds:
                bits = []
                if m.get("blood_type"):
                    bits.append(f"blood type {m['blood_type']}")
                if m.get("condition"):
                    bits.append(f"condition {m['condition']}")
                if m.get("ssn"):
                    bits.append(f"SSN {m['ssn']}")
                parts.append("- " + "; ".join(bits))
        parts.append("")

    # internet accounts
    if nets:
        parts.append("## Internet accounts")
        parts.append("")
        if len(nets) == 1:
            n = nets[0]
            parts.append(
                f"{fn} has one online account: username {n['username']} "
                f"({n['email']}) at {n['url']}."
            )
        else:
            parts.append(f"{fn} has {pluralize(len(nets), 'online account')}:")
            parts.append("")
            for n in nets:
                parts.append(
                    f"- {n['username']} ({n['email']}) at {n['url']}"
                )
        parts.append("")

    # Trim trailing blank line(s)
    while parts and parts[-1] == "":
        parts.pop()
    body = "\n".join(parts) + "\n"
    return description, body


# ---------- index files ----------


def _index_header(description: str) -> str:
    return f"---\ndescription: {description}\n---\n\n"


def _person_link(pid: str, name: str) -> str:
    return f"[[reference/people/{pid}.md]] ({name})"


def render_unique_index(
    title: str,
    description: str,
    rows: Iterable[tuple[str, str, str, str, Optional[str]]],
) -> str:
    """Render a name-keyed lookup with one entry per line.

    `rows` yields (sort_key, key_text, person_id, full_name, optional_extra).
    The sort_key drives ordering; key_text is what's displayed.
    """
    lines = [_index_header(description), f"# {title}\n"]
    # Total sort key — include pid + extra as tiebreakers so re-renders are stable.
    for _sort_key, key_text, pid, name, extra in sorted(
        rows, key=lambda r: (r[0].lower(), r[1], r[2], r[4] or "")
    ):
        suffix = f" — {extra}" if extra else ""
        lines.append(f"- **{key_text}** → {_person_link(pid, name)}{suffix}")
    return "\n".join(lines) + "\n"


def render_grouped_index(
    title: str,
    description: str,
    groups: dict[str, list[tuple[str, str, Optional[str]]]],
) -> str:
    """Render a multi-key lookup grouped by section heading.

    `groups[section_name] = list of (person_id, full_name, optional_extra)`.
    Sorted alphabetically by section, then by name within section.
    """
    lines = [_index_header(description), f"# {title}\n"]
    for section in sorted(groups.keys(), key=str.lower):
        # Total sort key — pid + extra as tiebreakers so re-renders are stable.
        entries = sorted(
            set(groups[section]), key=lambda e: (e[1].lower(), e[0], e[2] or "")
        )
        if not entries:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for pid, name, extra in entries:
            suffix = f" — {extra}" if extra else ""
            lines.append(f"- {_person_link(pid, name)}{suffix}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_indexes(db: dict) -> dict[str, str]:
    """Return a dict of relative_path -> file_content for every index file."""
    pid_to_name = {p["person_id"]: p["full_name"] for p in db["people"]}
    indexes: dict[str, str] = {}

    # people-by-name (unique key — full name; collisions become near-unique with DOB extra)
    rows = []
    for p in db["people"]:
        extra = f"b. {p['dob']}" if p.get("dob") else None
        rows.append((p["full_name"], p["full_name"], p["person_id"], p["full_name"], extra))
    # Use signature compatible with render_unique_index: (sort_key, key_text, pid, name, extra)
    indexes["reference/indexes/people-by-name.md"] = render_unique_index(
        "People by name",
        "Roster of all people in this memory. Grep by full name to find someone's reference file.",
        ((r[0], r[1], r[2], r[3], r[4]) for r in rows),
    )

    # pets-by-name (unique-ish; some pet names may repeat across owners — list all)
    pet_groups: dict[str, list[tuple[str, str, Optional[str]]]] = defaultdict(list)
    for owner_pid, pet_rows in db["pets"].items():
        for pt in pet_rows:
            label = pt["name"]
            extra_bits = []
            if pt.get("species"):
                extra_bits.append(pt["species"])
            if pt.get("breed"):
                extra_bits.append(pt["breed"].lower())
            extra = ", ".join(extra_bits) or None
            pet_groups[label].append((owner_pid, pid_to_name[owner_pid], extra))
    indexes["reference/indexes/pets-by-name.md"] = render_grouped_index(
        "Pets by name",
        "Pet name → owner. Grep for the pet name to find its owner's reference file.",
        pet_groups,
    )

    # vehicles-by-plate (unique key — plate)
    vp_rows = []
    for owner_pid, veh_rows in db["vehicles"].items():
        for v in veh_rows:
            plate = v.get("license_plate")
            if not plate:
                continue
            extra = f"{v['year']} {v['make']} {v['model']}"
            vp_rows.append((plate, plate, owner_pid, pid_to_name[owner_pid], extra))
    indexes["reference/indexes/vehicles-by-plate.md"] = render_unique_index(
        "Vehicles by license plate",
        "License plate → owner. Grep the plate to find the owner's reference file.",
        ((r[0], r[1], r[2], r[3], r[4]) for r in vp_rows),
    )

    # internet-by-username (unique key)
    ui_rows = []
    for owner_pid, net_rows in db["internet_accounts"].items():
        for n in net_rows:
            uname = n.get("username")
            if not uname:
                continue
            extra = f"at {n['url']}" if n.get("url") else None
            ui_rows.append((uname, uname, owner_pid, pid_to_name[owner_pid], extra))
    indexes["reference/indexes/internet-by-username.md"] = render_unique_index(
        "Internet usernames",
        "Username → owner. Grep the username to find the owner's reference file.",
        ((r[0], r[1], r[2], r[3], r[4]) for r in ui_rows),
    )

    # addresses-by-state (grouped)
    state_groups: dict[str, list[tuple[str, str, Optional[str]]]] = defaultdict(list)
    city_groups: dict[str, list[tuple[str, str, Optional[str]]]] = defaultdict(list)
    for owner_pid, addr_rows in db["addresses"].items():
        for a in addr_rows:
            if a.get("state"):
                state_groups[a["state"]].append(
                    (owner_pid, pid_to_name[owner_pid], a.get("city"))
                )
            if a.get("city"):
                city_groups[a["city"]].append(
                    (owner_pid, pid_to_name[owner_pid], a.get("state"))
                )
    indexes["reference/indexes/addresses-by-state.md"] = render_grouped_index(
        "Addresses by state",
        "Each section lists residents of that state. Grep for the state name to find everyone living there.",
        state_groups,
    )
    indexes["reference/indexes/addresses-by-city.md"] = render_grouped_index(
        "Addresses by city",
        "Each section lists residents of that city. Grep the city name to find everyone living there.",
        city_groups,
    )

    # employments-by-employer (grouped — powers coworker queries)
    emp_groups: dict[str, list[tuple[str, str, Optional[str]]]] = defaultdict(list)
    for owner_pid, emp_rows in db["employments"].items():
        for e in emp_rows:
            if e.get("employer"):
                emp_groups[e["employer"]].append(
                    (owner_pid, pid_to_name[owner_pid], e.get("job_title"))
                )
    indexes["reference/indexes/employments-by-employer.md"] = render_grouped_index(
        "Employments by employer",
        "Each section lists employees of that employer. Grep the employer name to find coworkers.",
        emp_groups,
    )

    # bank-by-bank-name (grouped — powers same-bank queries)
    bank_groups: dict[str, list[tuple[str, str, Optional[str]]]] = defaultdict(list)
    for owner_pid, bank_rows in db["bank_accounts"].items():
        for b in bank_rows:
            if b.get("bank_name"):
                bank_groups[b["bank_name"]].append(
                    (owner_pid, pid_to_name[owner_pid], None)
                )
    indexes["reference/indexes/bank-by-bank-name.md"] = render_grouped_index(
        "Bank customers by bank",
        "Each section lists customers of that bank. Grep the bank name to find others banking there.",
        bank_groups,
    )

    # insurance-by-provider (grouped)
    ins_groups: dict[str, list[tuple[str, str, Optional[str]]]] = defaultdict(list)
    for owner_pid, ins_rows in db["insurance_policies"].items():
        for ip in ins_rows:
            if ip.get("insurer"):
                ins_groups[ip["insurer"]].append(
                    (owner_pid, pid_to_name[owner_pid], ip.get("policy_type"))
                )
    indexes["reference/indexes/insurance-by-provider.md"] = render_grouped_index(
        "Insurance policies by provider",
        "Each section lists policyholders of that insurer. Grep the insurer name to find others.",
        ins_groups,
    )

    # medical-by-blood-type (grouped)
    blood_groups: dict[str, list[tuple[str, str, Optional[str]]]] = defaultdict(list)
    for owner_pid, med_rows in db["medical_records"].items():
        for m in med_rows:
            if m.get("blood_type"):
                blood_groups[m["blood_type"]].append(
                    (owner_pid, pid_to_name[owner_pid], m.get("condition"))
                )
    indexes["reference/indexes/medical-by-blood-type.md"] = render_grouped_index(
        "Medical records by blood type",
        "Each section lists people with that blood type. Grep the blood type to find matches.",
        blood_groups,
    )

    return indexes


# ---------- system index ----------


SYSTEM_INDEX_TEMPLATE = """---
description: Memory map describing what's in this MemFS and how to navigate it.
---

# Memory map

This memory contains records on {n_people} synthetic people. Person details live as prose narratives in `reference/people/`. The narrative bodies do not contain machine-readable IDs — instead, the domain indexes under `reference/indexes/` map names, plates, usernames, states, employers, etc. to the matching `[[reference/people/pers-XXXX.md]]` files.

## Available indexes

- **People by name** — [[reference/indexes/people-by-name.md]]
- **Pets by name** — [[reference/indexes/pets-by-name.md]]
- **Vehicles by license plate** — [[reference/indexes/vehicles-by-plate.md]]
- **Internet usernames** — [[reference/indexes/internet-by-username.md]]
- **Addresses by state** — [[reference/indexes/addresses-by-state.md]]
- **Addresses by city** — [[reference/indexes/addresses-by-city.md]]
- **Employers (and coworkers)** — [[reference/indexes/employments-by-employer.md]]
- **Bank customers by bank** — [[reference/indexes/bank-by-bank-name.md]]
- **Insurance policies by provider** — [[reference/indexes/insurance-by-provider.md]]
- **Medical records by blood type** — [[reference/indexes/medical-by-blood-type.md]]

## How to traverse

Most questions chain through indirect relationships ("same state as the owner of X", "same employer as", "same blood type as"). The pattern:

1. Start in the relevant index to find the person whose property you need.
2. Open `reference/people/<id>.md` to read their prose narrative.
3. Use a property from that narrative (state, employer, blood type, etc.) to look up other people via the matching index.
4. Open as many `reference/people/<id>.md` files as needed to compute the answer.

The indexes are designed to be **grepped**, not loaded whole. Target the key you need — a specific pet name, plate, state heading — rather than reading the full index file. Likewise, only open the person files you need.
"""


def render_system_index(db: dict) -> str:
    return SYSTEM_INDEX_TEMPLATE.format(n_people=len(db["people"]))


# ---------- driver ----------


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_all(db_path: Path, out_dir: Path, limit: Optional[int] = None) -> None:
    db = load_db(db_path)
    if limit:
        keep_ids = {p["person_id"] for p in db["people"][:limit]}
        db["people"] = [p for p in db["people"] if p["person_id"] in keep_ids]
        for table in (
            "addresses", "pets", "vehicles", "employments",
            "bank_accounts", "credit_cards", "insurance_policies",
            "medical_records", "internet_accounts",
        ):
            db[table] = {pid: rows for pid, rows in db[table].items() if pid in keep_ids}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # system/index.md
    write_file(out_dir / "system" / "index.md", render_system_index(db))

    # reference/indexes/*
    for rel_path, content in build_indexes(db).items():
        write_file(out_dir / rel_path, content)

    # reference/people/*
    for person in db["people"]:
        description, body = render_person(person, db)
        front = f"---\ndescription: {description}\n---\n\n"
        write_file(
            out_dir / "reference" / "people" / f"{person['person_id']}.md",
            front + body,
        )

    n_people = len(db["people"])
    n_indexes = len(build_indexes(db))
    print(f"Rendered {n_people} person files + {n_indexes} indexes + 1 system file -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_DEFAULT, help="Path to SQLite DB")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Output directory for the memfs tree")
    parser.add_argument("--limit", type=int, default=None, help="Only render the first N people (smoke testing)")
    args = parser.parse_args()
    render_all(args.db, args.out, args.limit)


if __name__ == "__main__":
    main()
