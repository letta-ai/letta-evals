#!/usr/bin/env python3
"""Audit filesystem benchmark datasets against the source DB."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Callable

MetricSpec = tuple[str, str, str | None]


@dataclass
class AuditResult:
    index: int
    status: str
    question_type: str
    ground_truth: str
    valid_answers: list[str]
    question: str
    note: str = ""


class FilesystemData:
    def __init__(self, db_path: Path):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        self.people = {}
        self.all_people = set()
        self.pet_owners_by_name = defaultdict(set)
        self.pet_species_by_name = defaultdict(set)
        self.pets_by_owner = defaultdict(list)
        self.vehicle_owners_by_plate = defaultdict(set)
        self.vehicles_by_owner = defaultdict(list)
        self.credit_cards_by_owner = defaultdict(list)
        self.bank_accounts_by_owner = defaultdict(list)
        self.addresses_by_owner = defaultdict(list)
        self.insurance_by_owner = defaultdict(list)
        self.internet_by_owner = defaultdict(list)
        self.username_to_owner = defaultdict(set)
        self.employments_by_owner = defaultdict(list)
        self.medical_by_owner = defaultdict(list)

        for row in conn.execute("SELECT * FROM people"):
            self.people[row["person_id"]] = {
                "name": row["full_name"],
                "dob": date.fromisoformat(row["dob"]),
            }
            self.all_people.add(row["person_id"])

        for row in conn.execute("SELECT * FROM pets"):
            entry = dict(row)
            self.pets_by_owner[row["owner_id"]].append(entry)
            self.pet_owners_by_name[row["name"]].add(row["owner_id"])
            self.pet_species_by_name[row["name"]].add(row["species"])

        for row in conn.execute("SELECT * FROM vehicles"):
            entry = dict(row)
            self.vehicles_by_owner[row["owner_id"]].append(entry)
            self.vehicle_owners_by_plate[row["license_plate"]].add(row["owner_id"])

        for row in conn.execute("SELECT * FROM credit_cards"):
            self.credit_cards_by_owner[row["owner_id"]].append(dict(row))

        for row in conn.execute("SELECT * FROM bank_accounts"):
            self.bank_accounts_by_owner[row["owner_id"]].append(dict(row))

        for row in conn.execute("SELECT * FROM addresses"):
            self.addresses_by_owner[row["owner_id"]].append(dict(row))

        for row in conn.execute("SELECT * FROM insurance_policies"):
            self.insurance_by_owner[row["owner_id"]].append(dict(row))

        for row in conn.execute("SELECT * FROM internet_accounts"):
            entry = dict(row)
            self.internet_by_owner[row["owner_id"]].append(entry)
            self.username_to_owner[row["username"]].add(row["owner_id"])

        for row in conn.execute("SELECT * FROM employments"):
            self.employments_by_owner[row["owner_id"]].append(dict(row))

        for row in conn.execute("SELECT * FROM medical_records"):
            self.medical_by_owner[row["owner_id"]].append(dict(row))

        conn.close()

        self.residents_by_state = defaultdict(set)
        self.residents_by_city = defaultdict(set)
        self.people_by_blood_type = defaultdict(set)
        self.people_by_condition = defaultdict(set)
        self.people_by_cc_provider = defaultdict(set)
        self.people_by_cc_expire = defaultdict(set)
        self.people_by_insurance_type = defaultdict(set)
        self.people_by_job_title = defaultdict(set)
        self.people_by_vehicle_make = defaultdict(set)
        self.people_by_pet_species = defaultdict(set)
        self.people_by_url = defaultdict(set)

        self.states_by_owner = defaultdict(set)
        self.cities_by_owner = defaultdict(set)
        self.blood_types_by_owner = defaultdict(set)
        self.conditions_by_owner = defaultdict(set)
        self.cc_providers_by_owner = defaultdict(set)
        self.cc_expires_by_owner = defaultdict(set)
        self.insurance_types_by_owner = defaultdict(set)
        self.job_titles_by_owner = defaultdict(set)
        self.vehicle_makes_by_owner = defaultdict(set)
        self.pet_species_owned_by_owner = defaultdict(set)
        self.urls_by_owner = defaultdict(set)

        for owner_id, rows in self.addresses_by_owner.items():
            for row in rows:
                self.states_by_owner[owner_id].add(row["state"])
                self.cities_by_owner[owner_id].add(row["city"])
                self.residents_by_state[row["state"]].add(owner_id)
                self.residents_by_city[row["city"]].add(owner_id)

        for owner_id, rows in self.medical_by_owner.items():
            for row in rows:
                self.blood_types_by_owner[owner_id].add(row["blood_type"])
                self.conditions_by_owner[owner_id].add(row["condition"])
                self.people_by_blood_type[row["blood_type"]].add(owner_id)
                self.people_by_condition[row["condition"]].add(owner_id)

        for owner_id, rows in self.credit_cards_by_owner.items():
            for row in rows:
                self.cc_providers_by_owner[owner_id].add(row["provider"])
                self.cc_expires_by_owner[owner_id].add(row["expire"])
                self.people_by_cc_provider[row["provider"]].add(owner_id)
                self.people_by_cc_expire[row["expire"]].add(owner_id)

        for owner_id, rows in self.insurance_by_owner.items():
            for row in rows:
                self.insurance_types_by_owner[owner_id].add(row["policy_type"])
                self.people_by_insurance_type[row["policy_type"]].add(owner_id)

        for owner_id, rows in self.employments_by_owner.items():
            for row in rows:
                self.job_titles_by_owner[owner_id].add(row["job_title"])
                self.people_by_job_title[row["job_title"]].add(owner_id)

        for owner_id, rows in self.vehicles_by_owner.items():
            for row in rows:
                self.vehicle_makes_by_owner[owner_id].add(row["make"])
                self.people_by_vehicle_make[row["make"]].add(owner_id)

        for owner_id, rows in self.pets_by_owner.items():
            for row in rows:
                self.pet_species_owned_by_owner[owner_id].add(row["species"])
                self.people_by_pet_species[row["species"]].add(owner_id)

        for owner_id, rows in self.internet_by_owner.items():
            for row in rows:
                self.urls_by_owner[owner_id].add(row["url"])
                self.people_by_url[row["url"]].add(owner_id)

        self.total_balance_by_owner = {
            owner_id: round(sum(row["balance"] for row in rows), 2)
            for owner_id, rows in self.bank_accounts_by_owner.items()
        }
        self.max_salary_by_owner = {
            owner_id: max(row["salary"] for row in rows)
            for owner_id, rows in self.employments_by_owner.items()
        }
        self.latest_employment_by_owner = {
            owner_id: max(date.fromisoformat(row["start_date"]) for row in rows)
            for owner_id, rows in self.employments_by_owner.items()
        }
        self.earliest_policy_expiry_by_owner = {
            owner_id: min(date.fromisoformat(row["expires"]) for row in rows)
            for owner_id, rows in self.insurance_by_owner.items()
        }
        self.oldest_vehicle_year_by_owner = {
            owner_id: min(row["year"] for row in rows)
            for owner_id, rows in self.vehicles_by_owner.items()
        }

    def count(self, owner_id: str, kind: str, qualifier: str | None = None) -> int:
        if kind == "pets":
            return len(self.pets_by_owner[owner_id])
        if kind == "vehicles":
            return len(self.vehicles_by_owner[owner_id])
        if kind == "credit_cards":
            if qualifier is None:
                return len(self.credit_cards_by_owner[owner_id])
            return sum(1 for row in self.credit_cards_by_owner[owner_id] if row["provider"].lower() == qualifier.lower())
        if kind == "bank_accounts":
            return len(self.bank_accounts_by_owner[owner_id])
        if kind == "insurance_policies":
            if qualifier is None:
                return len(self.insurance_by_owner[owner_id])
            return sum(1 for row in self.insurance_by_owner[owner_id] if row["policy_type"].lower() == qualifier.lower())
        if kind == "internet_accounts":
            return len(self.internet_by_owner[owner_id])
        raise ValueError(f"Unknown count kind: {kind}")

    def metric_value(self, owner_id: str, metric: MetricSpec):
        name, direction, qualifier = metric
        if name == "total_balance":
            return self.total_balance_by_owner.get(owner_id, 0.0)
        if name == "max_salary":
            return self.max_salary_by_owner.get(owner_id)
        if name == "num_credit_cards":
            return self.count(owner_id, "credit_cards", qualifier)
        if name == "num_vehicles":
            return self.count(owner_id, "vehicles")
        if name == "num_insurance_policies":
            return self.count(owner_id, "insurance_policies", qualifier)
        if name == "num_internet_accounts":
            return self.count(owner_id, "internet_accounts")
        if name == "num_pets":
            return self.count(owner_id, "pets")
        if name == "num_bank_accounts":
            return self.count(owner_id, "bank_accounts")
        if name == "dob":
            return self.people[owner_id]["dob"]
        if name == "latest_employment_start":
            return self.latest_employment_by_owner.get(owner_id)
        if name == "earliest_policy_expiry":
            return self.earliest_policy_expiry_by_owner.get(owner_id)
        if name == "oldest_vehicle_year":
            return self.oldest_vehicle_year_by_owner.get(owner_id)
        raise ValueError(f"Unknown metric: {metric}")

    def person_name(self, owner_id: str) -> str:
        return self.people[owner_id]["name"]

    def resolve_ref_owners(self, phrase: str) -> set[str]:
        patterns = [
            (r"owner of the pet named '([^']+)'", lambda m: self.pet_owners_by_name[m.group(1)]),
            (r"owner of pet named '([^']+)'", lambda m: self.pet_owners_by_name[m.group(1)]),
            (r"owner of the pet '([^']+)'", lambda m: self.pet_owners_by_name[m.group(1)]),
            (r"owner of pet '([^']+)'", lambda m: self.pet_owners_by_name[m.group(1)]),
            (r"owner of the vehicle with license plate '([^']+)'", lambda m: self.vehicle_owners_by_plate[m.group(1)]),
            (r"owner of vehicle with license plate '([^']+)'", lambda m: self.vehicle_owners_by_plate[m.group(1)]),
            (r"owner of license plate '([^']+)'", lambda m: self.vehicle_owners_by_plate[m.group(1)]),
            (r"owner of internet username '([^']+)'", lambda m: self.username_to_owner[m.group(1)]),
            (r"user with internet username '([^']+)'", lambda m: self.username_to_owner[m.group(1)]),
            (r"the user with internet username '([^']+)'", lambda m: self.username_to_owner[m.group(1)]),
            (r"person with internet username '([^']+)'", lambda m: self.username_to_owner[m.group(1)]),
            (r"owner of the internet account with username '([^']+)'", lambda m: self.username_to_owner[m.group(1)]),
        ]
        for pattern, resolver in patterns:
            match = re.search(pattern, phrase)
            if match:
                return set(resolver(match))
        raise ValueError(f"Unsupported reference phrase: {phrase}")


class DatasetAuditor:
    def __init__(self, data: FilesystemData):
        self.data = data

    def parse_money(self, text: str) -> float:
        return float(text.replace("$", "").replace(",", ""))

    def format_money(self, value: float) -> str:
        return f"${value:,.2f}"

    def normalize_answer(self, answer: str) -> str:
        text = answer.strip().lower()
        text = text.replace("total bank account balance", "").replace("total bank balance", "")
        text = text.replace("insurance policies", "").replace("insurance policy", "")
        text = text.replace("bank accounts", "").replace("bank account", "")
        text = text.replace("credit cards", "").replace("credit card", "")
        text = text.replace("internet accounts", "").replace("internet account", "")
        text = text.replace("pets", "").replace("pet", "")
        text = text.replace("vehicles", "").replace("vehicle", "")
        text = text.replace("records", "").replace("record", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def dedupe_groups(self, groups: list[set[str]]) -> list[set[str]]:
        unique = {}
        for group in groups:
            if not group:
                continue
            key = frozenset(group)
            unique[key] = set(group)
        return list(unique.values())

    def same_state_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for state in self.data.states_by_owner[owner_id]:
                groups.append(set(self.data.residents_by_state[state]))
        return self.dedupe_groups(groups)

    def same_city_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for city in self.data.cities_by_owner[owner_id]:
                groups.append(set(self.data.residents_by_city[city]))
        return self.dedupe_groups(groups)

    def same_provider_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for provider in self.data.cc_providers_by_owner[owner_id]:
                groups.append(set(self.data.people_by_cc_provider[provider]))
        return self.dedupe_groups(groups)

    def same_expire_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for expire in self.data.cc_expires_by_owner[owner_id]:
                groups.append(set(self.data.people_by_cc_expire[expire]))
        return self.dedupe_groups(groups)

    def same_insurance_type_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for policy_type in self.data.insurance_types_by_owner[owner_id]:
                groups.append(set(self.data.people_by_insurance_type[policy_type]))
        return self.dedupe_groups(groups)

    def same_blood_type_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for blood_type in self.data.blood_types_by_owner[owner_id]:
                groups.append(set(self.data.people_by_blood_type[blood_type]))
        return self.dedupe_groups(groups)

    def same_condition_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for condition in self.data.conditions_by_owner[owner_id]:
                groups.append(set(self.data.people_by_condition[condition]))
        return self.dedupe_groups(groups)

    def same_job_title_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for title in self.data.job_titles_by_owner[owner_id]:
                groups.append(set(self.data.people_by_job_title[title]))
        return self.dedupe_groups(groups)

    def same_url_groups(self, ref_phrase: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.resolve_ref_owners(ref_phrase):
            for url in self.data.urls_by_owner[owner_id]:
                groups.append(set(self.data.people_by_url[url]))
        return self.dedupe_groups(groups)

    def same_pet_type_groups(self, pet_name: str, exclude_species: str | None = None) -> list[set[str]]:
        groups = []
        for species in self.data.pet_species_by_name[pet_name]:
            group = set(self.data.people_by_pet_species[species])
            if exclude_species:
                group = {
                    owner_id
                    for owner_id in group
                    if exclude_species.lower() not in {item.lower() for item in self.data.pet_species_owned_by_owner[owner_id]}
                }
            groups.append(group)
        return self.dedupe_groups(groups)

    def same_vehicle_make_groups_from_pet_owner(self, pet_name: str) -> list[set[str]]:
        groups = []
        for owner_id in self.data.pet_owners_by_name[pet_name]:
            for make in self.data.vehicle_makes_by_owner[owner_id]:
                groups.append(set(self.data.people_by_vehicle_make[make]))
        return self.dedupe_groups(groups)

    def auto_insurance_same_state_groups(self, ref_phrase: str) -> list[set[str]]:
        return self.filter_groups(self.same_state_groups(ref_phrase), lambda owner_id: self.data.count(owner_id, "insurance_policies", "Auto") > 0)

    def filter_groups(self, groups: list[set[str]], predicate: Callable[[str], bool]) -> list[set[str]]:
        return self.dedupe_groups([{owner_id for owner_id in group if predicate(owner_id)} for group in groups])

    def metric(self, name: str, direction: str = "desc", qualifier: str | None = None) -> MetricSpec:
        return (name, direction, qualifier)

    def best_candidates(self, group: set[str], metric: MetricSpec, tiebreakers: list[MetricSpec]) -> set[str]:
        candidates = set(group)
        for metric_spec in [metric, *tiebreakers]:
            values = {
                owner_id: self.data.metric_value(owner_id, metric_spec)
                for owner_id in candidates
                if self.data.metric_value(owner_id, metric_spec) is not None
            }
            if not values:
                return set()
            if metric_spec[1] == "desc":
                best_value = max(values.values())
            else:
                best_value = min(values.values())
            candidates = {owner_id for owner_id, value in values.items() if value == best_value}
            if len(candidates) <= 1:
                break
        return candidates

    def select_people(self, groups: list[set[str]], metric: MetricSpec, tiebreakers: list[MetricSpec] | None = None) -> list[set[str]]:
        winners = []
        for group in groups:
            selected = self.best_candidates(group, metric, tiebreakers or [])
            if selected:
                winners.append(selected)
        return self.dedupe_groups(winners)

    def top_n_groups(self, base_group: set[str], metric: MetricSpec, n: int) -> list[set[str]]:
        values = {
            owner_id: self.data.metric_value(owner_id, metric)
            for owner_id in base_group
            if self.data.metric_value(owner_id, metric) is not None
        }
        if len(values) < n:
            return [set(values)]
        sorted_values = sorted(values.items(), key=lambda item: item[1], reverse=(metric[1] == "desc"))
        threshold_value = sorted_values[n - 1][1]
        if metric[1] == "desc":
            required = {owner_id for owner_id, value in values.items() if value > threshold_value}
            tied = sorted(owner_id for owner_id, value in values.items() if value == threshold_value)
        else:
            required = {owner_id for owner_id, value in values.items() if value < threshold_value}
            tied = sorted(owner_id for owner_id, value in values.items() if value == threshold_value)
        slots = n - len(required)
        if slots <= 0:
            return [required]
        if len(tied) <= 10:
            return [set(required) | set(combo) for combo in combinations(tied, slots)]
        return [set(required) | set(tied)]

    def parse_outer_compare(self, question: str) -> tuple[str, str, str]:
        prefixes = [
            ("Who owns more vehicles: ", "num_vehicles"),
            ("Who has more vehicles: ", "num_vehicles"),
            ("Who owns more insurance policies: ", "num_insurance_policies"),
            ("Who has more credit cards: ", "num_credit_cards"),
            ("Who has more insurance policies: ", "num_insurance_policies"),
            ("Who has more internet accounts: ", "num_internet_accounts"),
            ("Who has more bank accounts: ", "num_bank_accounts"),
            ("Who has more pets: ", "num_pets"),
            ("Who owns more pets: ", "num_pets"),
            ("Who has a higher total bank balance: ", "total_balance"),
            ("Who has a higher total bank account balance: ", "total_balance"),
            ("Who has the higher total bank balance: ", "total_balance"),
        ]
        for prefix, measure in prefixes:
            if question.startswith(prefix):
                body = question[len(prefix):].rstrip("?")
                parts = re.split(r", (?:OR|or) ", body, maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Could not split comparison body: {body}")
                left, right = parts
                return measure, left.strip(), right.strip()
        raise ValueError(f"Unsupported comparison question: {question}")

    def parse_measure_metric(self, measure_name: str) -> MetricSpec:
        mapping = {
            "num_vehicles": self.metric("num_vehicles"),
            "num_credit_cards": self.metric("num_credit_cards"),
            "num_insurance_policies": self.metric("num_insurance_policies"),
            "num_internet_accounts": self.metric("num_internet_accounts"),
            "num_bank_accounts": self.metric("num_bank_accounts"),
            "num_pets": self.metric("num_pets"),
            "total_balance": self.metric("total_balance"),
        }
        return mapping[measure_name]

    def answers_for_comparison(self, measure_name: str, left_groups: list[set[str]], right_groups: list[set[str]]) -> set[str]:
        measure = self.parse_measure_metric(measure_name)
        answers = set()
        for left_group in left_groups:
            for right_group in right_groups:
                for left_person in left_group:
                    for right_person in right_group:
                        left_value = self.data.metric_value(left_person, measure)
                        right_value = self.data.metric_value(right_person, measure)
                        if left_value is None or right_value is None:
                            continue
                        if left_value > right_value:
                            answers.add(self.data.person_name(left_person))
                        elif left_value < right_value:
                            answers.add(self.data.person_name(right_person))
                        else:
                            answers.add("Tie")
        return answers

    def person_selector(self, text: str, forced_tiebreaker: MetricSpec | None = None) -> list[set[str]]:
        selector_text = text.strip().rstrip(".")
        selector_core = re.sub(r" \([^)]*tiebreaker[^)]*\)", "", selector_text, flags=re.IGNORECASE)
        tiebreakers = self.tiebreakers_from_text(selector_text)
        if forced_tiebreaker and forced_tiebreaker not in tiebreakers:
            tiebreakers.append(forced_tiebreaker)

        patterns = [
            (
                r"^the person with the highest total bank balance among residents of the same state as (.+)$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("total_balance"), tiebreakers),
            ),
            (
                r"^the person with the highest salary among residents of the same state as (.+)$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("max_salary"), tiebreakers),
            ),
            (
                r"^the person with the most credit cards among residents of the same state as (.+)$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_credit_cards"), tiebreakers),
            ),
            (
                r"^the person with the most insurance policies among residents of the same state as (.+)$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_insurance_policies"), tiebreakers),
            ),
            (
                r"^the person with the most vehicles among residents of the same state as (.+)$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_vehicles"), tiebreakers),
            ),
            (
                r"^the person with the highest total bank balance among all holders of the same credit card provider as (.+)$",
                lambda m: self.select_people(self.same_provider_groups(m.group(1)), self.metric("total_balance"), tiebreakers),
            ),
            (
                r"^the person with the highest total bank balance among holders of the same credit card provider as (.+)$",
                lambda m: self.select_people(self.same_provider_groups(m.group(1)), self.metric("total_balance"), tiebreakers),
            ),
            (
                r"^the person with the highest salary among holders of the same credit card provider as (.+)$",
                lambda m: self.select_people(self.same_provider_groups(m.group(1)), self.metric("max_salary"), tiebreakers),
            ),
            (
                r"^the person with the highest salary among all holders of the same insurance type as (.+)$",
                lambda m: self.select_people(self.same_insurance_type_groups(m.group(1)), self.metric("max_salary"), tiebreakers),
            ),
            (
                r"^the person with the highest salary among those with the same blood type as (.+)$",
                lambda m: self.select_people(self.same_blood_type_groups(m.group(1)), self.metric("max_salary"), tiebreakers),
            ),
            (
                r"^the person with the highest total bank balance among those who share the same medical condition as (.+)$",
                lambda m: self.select_people(self.same_condition_groups(m.group(1)), self.metric("total_balance"), tiebreakers),
            ),
            (
                r"^the person with the highest total bank balance among people who share the same medical condition as (.+)$",
                lambda m: self.select_people(self.same_condition_groups(m.group(1)), self.metric("total_balance"), tiebreakers),
            ),
            (
                r"^the person with the highest total bank balance among people who share the same blood type as (.+)$",
                lambda m: self.select_people(self.same_blood_type_groups(m.group(1)), self.metric("total_balance"), tiebreakers),
            ),
            (
                r"^the person with the highest salary among those who share the same job title as (.+)$",
                lambda m: self.select_people(self.same_job_title_groups(m.group(1)), self.metric("max_salary"), tiebreakers),
            ),
            (
                r"^the person with the most vehicles among all holders of the same credit card provider as (.+)$",
                lambda m: self.select_people(self.same_provider_groups(m.group(1)), self.metric("num_vehicles"), tiebreakers),
            ),
            (
                r"^the person with the most internet accounts among holders of the same credit card provider as (.+)$",
                lambda m: self.select_people(self.same_provider_groups(m.group(1)), self.metric("num_internet_accounts"), tiebreakers),
            ),
            (
                r"^the person with the most insurance policies among those who have the same job title as (.+)$",
                lambda m: self.select_people(self.same_job_title_groups(m.group(1)), self.metric("num_insurance_policies"), tiebreakers),
            ),
            (
                r"^the person with the most credit cards among people who share the same blood type as (.+)$",
                lambda m: self.select_people(self.same_blood_type_groups(m.group(1)), self.metric("num_credit_cards"), tiebreakers),
            ),
            (
                r"^the person with the most credit cards among residents of the same state as (.+)$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_credit_cards"), tiebreakers),
            ),
            (
                r"^the person with the most credit cards among Mastercard holders who live in the same state as (.+)$",
                lambda m: self.select_people(
                    self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "credit_cards", "Mastercard") > 0),
                    self.metric("num_credit_cards", qualifier="Mastercard"),
                    tiebreakers,
                ),
            ),
            (
                r"^the person with the most credit cards among Discover cardholders who live in the same state as (.+)$",
                lambda m: self.select_people(
                    self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "credit_cards", "Discover") > 0),
                    self.metric("num_credit_cards", qualifier="Discover"),
                    tiebreakers,
                ),
            ),
            (
                r"^the person with the most vehicles among people who have a credit card with the same expiry date as (.+)$",
                lambda m: self.select_people(self.same_expire_groups(m.group(1)), self.metric("num_vehicles"), tiebreakers),
            ),
            (
                r"^the person with the highest salary among owners of the same type of pet as '([^']+)'(?: .+)?$",
                lambda m: self.select_people(
                    self.same_pet_type_groups(
                        m.group(1),
                        "birds" if "excluding people who also own birds" in selector_text else "fish" if "excluding people who also own fish" in selector_text else None,
                    ),
                    self.metric("max_salary"),
                    tiebreakers,
                ),
            ),
            (
                r"^the person with the oldest vehicle among residents of the same state as (.+)$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("oldest_vehicle_year", "asc"), tiebreakers),
            ),
            (
                r"^the person with the oldest vehicle among those who share the same blood type as (.+)$",
                lambda m: self.select_people(self.same_blood_type_groups(m.group(1)), self.metric("oldest_vehicle_year", "asc"), tiebreakers),
            ),
            (
                r"^the person with the oldest vehicle among Auto insurance policy holders who live in the same state as (.+)$",
                lambda m: self.select_people(self.auto_insurance_same_state_groups(m.group(1)), self.metric("oldest_vehicle_year", "asc"), tiebreakers),
            ),
            (
                r"^the oldest person \(by date of birth\) among holders of the same credit card provider as (.+)$",
                lambda m: self.select_people(self.same_provider_groups(m.group(1)), self.metric("dob", "asc"), tiebreakers),
            ),
            (
                r"^the oldest person among holders of the same credit card provider as (.+)$",
                lambda m: self.select_people(self.same_provider_groups(m.group(1)), self.metric("dob", "asc"), tiebreakers),
            ),
            (
                r"^the person with the most pets among those who share the same medical condition as (.+)$",
                lambda m: self.select_people(self.same_condition_groups(m.group(1)), self.metric("num_pets"), tiebreakers),
            ),
        ]

        for pattern, handler in patterns:
            match = re.match(pattern, selector_core)
            if match:
                return handler(match)

        raise ValueError(f"Unsupported selector: {selector_text}")

    def tiebreakers_from_text(self, text: str) -> list[MetricSpec]:
        lower = text.lower()
        tiebreakers = []
        if "highest total bank balance as a tiebreaker" in lower or "highest total bank balance as tiebreaker" in lower:
            tiebreakers.append(self.metric("total_balance"))
        if "total bank balance as a tiebreaker" in lower or "total bank balance as tiebreaker" in lower:
            tiebreakers.append(self.metric("total_balance"))
        if "highest salary as a tiebreaker" in lower:
            tiebreakers.append(self.metric("max_salary"))
        return tiebreakers

    def comparison_question(self, row: dict) -> tuple[set[str], str]:
        question = row["input"]
        if question.startswith("What is the combined total bank balance of "):
            match = re.match(r"^What is the combined total bank balance of (.+), AND (.+)\? \((.+)\)$", question)
            if not match:
                raise ValueError(f"Unsupported combined-balance question: {question}")
            forced = self.metric("total_balance") if "tiebreaker" in match.group(3).lower() else None
            left = self.person_selector(match.group(1), forced)
            right = self.person_selector(match.group(2), forced)
            answers = {
                self.format_money(
                    self.data.total_balance_by_owner.get(left_person, 0.0) + self.data.total_balance_by_owner.get(right_person, 0.0)
                )
                for left_group in left
                for right_group in right
                for left_person in left_group
                for right_person in right_group
            }
            return answers, ""

        measure_name, left_text, right_text = self.parse_outer_compare(question)
        left = self.person_selector(left_text)
        right = self.person_selector(right_text)
        return self.answers_for_comparison(measure_name, left, right), ""

    def comparison_tiebreak_question(self, row: dict) -> tuple[set[str], str]:
        question = row["input"]
        patterns = [
            (
                r"^Among all people who live in the same state as (.+), who owns the most vehicles\? If there's a tie, who among them is the oldest\?$",
                self.metric("num_vehicles"),
                [self.metric("dob", "asc")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who has the most credit cards\? If there's a tie, who among them has the highest total bank balance\?$",
                self.metric("num_credit_cards"),
                [self.metric("total_balance")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who has the most pets\? If there's a tie, who among them has the most insurance policies\? If still tied, who has the most vehicles\?$",
                self.metric("num_pets"),
                [self.metric("num_insurance_policies"), self.metric("num_vehicles")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who owns the most vehicles\? If there's a tie, who among them has the most bank accounts\? If still tied, who has the highest total bank balance\?$",
                self.metric("num_vehicles"),
                [self.metric("num_bank_accounts"), self.metric("total_balance")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who has the most credit cards\? If there's a tie, who among them has the most vehicles\? If still tied, who has the most bank accounts\?$",
                self.metric("num_credit_cards"),
                [self.metric("num_vehicles"), self.metric("num_bank_accounts")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who owns the most vehicles\? If there's a tie, who has the most credit cards\? If still tied, who has the highest total bank_balance\?$",
                self.metric("num_vehicles"),
                [self.metric("num_credit_cards"), self.metric("total_balance")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who owns the most vehicles\? If there's a tie, who has the most credit cards\? If still tied, who has the highest total bank balance\?$",
                self.metric("num_vehicles"),
                [self.metric("num_credit_cards"), self.metric("total_balance")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who among them owns the most vehicles\? If there's a tie, who has the highest total bank balance\?$",
                self.metric("num_vehicles"),
                [self.metric("total_balance")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who among them owns the most vehicles\? If there's a tie, who among them has the most vehicles\?$",
                self.metric("num_vehicles"),
                [],
            ),
            (
                r"^Among all people who live in the same state as (.+), who has the most credit cards\? If there's a tie, who among them owns the most vehicles\?$",
                self.metric("num_credit_cards"),
                [self.metric("num_vehicles")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who has the most credit cards\? If there's a tie, who among them has the highest total bank account balance\?$",
                self.metric("num_credit_cards"),
                [self.metric("total_balance")],
            ),
            (
                r"^Among all residents of the same state as (.+), who owns the most vehicles\? If there's a tie, who has the highest salary\?$",
                self.metric("num_vehicles"),
                [self.metric("max_salary")],
            ),
            (
                r"^Among all people who live in the same state as (.+), who has the most bank accounts\? If there's a tie, who has the highest total bank account balance\?$",
                self.metric("num_bank_accounts"),
                [self.metric("total_balance")],
            ),
        ]

        for pattern, metric, tiebreakers in patterns:
            match = re.match(pattern, question)
            if match:
                winners = self.select_people(self.same_state_groups(match.group(1)), metric, tiebreakers)
                return {self.data.person_name(owner_id) for group in winners for owner_id in group}, ""

        raise ValueError(f"Unsupported comparison_tiebreak question: {question}")

    def negation_question(self, row: dict) -> tuple[set[str], str]:
        question = row["input"]
        patterns = [
            (
                r"^Among all people who live in the same state as (.+), who does NOT own any pets\?$",
                lambda m: self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "pets") == 0),
            ),
            (
                r"^Among all people who live in the same state as (.+), who does NOT have any credit cards\?$",
                lambda m: self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "credit_cards") == 0),
            ),
            (
                r"^Among all people who live in the same state as (.+), who does NOT have any insurance policies\?$",
                lambda m: self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "insurance_policies") == 0),
            ),
            (
                r"^Among all people who have an internet account at the same URL as (.+), who does NOT own any vehicles\?$",
                lambda m: self.filter_groups(self.same_url_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "vehicles") == 0),
            ),
        ]
        for pattern, builder in patterns:
            match = re.match(pattern, question)
            if match:
                groups = builder(match)
                return {self.data.person_name(owner_id) for group in groups for owner_id in group}, ""
        raise ValueError(f"Unsupported negation question: {question}")

    def temporal_question(self, row: dict) -> tuple[set[str], str]:
        question = row["input"]
        patterns = [
            (
                r"^Among the people who live in the same state as (.+), who started their job most recently\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("latest_employment_start")),
            ),
            (
                r"^Among all people who live in the same state as (.+), who started their employment most recently\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("latest_employment_start")),
            ),
            (
                r"^Among all people who live in the same state as (.+), whose insurance policy expires the soonest\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("earliest_policy_expiry", "asc")),
            ),
            (
                r"^Among all people who live in the same state as (.+) and own at least one vehicle, whose employment started most recently\?$",
                lambda m: self.select_people(
                    self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "vehicles") > 0),
                    self.metric("latest_employment_start"),
                ),
            ),
            (
                r"^Among all vehicle owners who live in the same state as (.+), whose insurance policy expires the soonest\?$",
                lambda m: self.select_people(
                    self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "vehicles") > 0),
                    self.metric("earliest_policy_expiry", "asc"),
                ),
            ),
        ]
        for pattern, builder in patterns:
            match = re.match(pattern, question)
            if match:
                winners = builder(match)
                return {self.data.person_name(owner_id) for group in winners for owner_id in group}, ""
        raise ValueError(f"Unsupported temporal question: {question}")

    def aggregation_question(self, row: dict) -> tuple[set[str], str]:
        question = row["input"]
        patterns = [
            (
                r"^What is the total bank balance of the person who has the most credit cards among all residents of the same state as (.+)\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_credit_cards")),
                "total_balance",
            ),
            (
                r"^What is the total bank account balance of the person with the most credit cards among all residents of the same state as (.+)\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_credit_cards")),
                "total_balance",
            ),
            (
                r"^What is the total bank account balance of the person who has the most credit cards among all residents of the same state as (.+)\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_credit_cards")),
                "total_balance",
            ),
            (
                r"^What is the total bank balance of the oldest person \(by birth date\) among those who live in the same state as (.+) and have at least one insurance policy\?$",
                lambda m: self.select_people(
                    self.filter_groups(self.same_state_groups(m.group(1)), lambda owner_id: self.data.count(owner_id, "insurance_policies") > 0),
                    self.metric("dob", "asc"),
                ),
                "total_balance",
            ),
            (
                r"^Among all people who live in the same state as (.+), who has the most bank accounts\? If there is a tie, consider the one with the highest total bank balance\. How many insurance policies does this person have\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_bank_accounts"), [self.metric("total_balance")]),
                "num_insurance_policies",
            ),
            (
                r"^What is the total bank balance of the person with the most insurance policies among all residents of the same state as (.+)\? If there's a tie for most insurance policies, consider the one with the highest total bank balance\.$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_insurance_policies"), [self.metric("total_balance")]),
                "total_balance",
            ),
            (
                r"^What is the total bank balance of the person with the most internet accounts among all residents of the same state as (.+)\? If there's a tie for most internet accounts, consider the one with the highest total bank balance\.$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_internet_accounts"), [self.metric("total_balance")]),
                "total_balance",
            ),
            (
                r"^What is the total bank balance of the person with the most internet accounts among all residents of the same state as (.+)\? If there's a tie for most internet accounts, use the highest total bank balance as a tiebreaker\.$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_internet_accounts"), [self.metric("total_balance")]),
                "total_balance",
            ),
            (
                r"^What is the total bank balance of the person with the most vehicles among residents of the same state as (.+)\? If there's a tie for most vehicles, use highest total bank balance as the tiebreaker\.$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_vehicles"), [self.metric("total_balance")]),
                "total_balance",
            ),
        ]
        for pattern, builder, answer_kind in patterns:
            match = re.match(pattern, question)
            if match:
                winners = builder(match)
                if answer_kind == "total_balance":
                    return {
                        self.format_money(self.data.total_balance_by_owner.get(owner_id, 0.0))
                        for group in winners
                        for owner_id in group
                    }, ""
                return {str(self.data.count(owner_id, "insurance_policies")) for group in winners for owner_id in group}, ""

        match = re.match(
            r"^What is the combined bank balance of the 3 people with the most insurance policies among residents of the same state as (.+)\?$",
            question,
        )
        if match:
            groups = self.same_state_groups(match.group(1))
            answers = set()
            for group in groups:
                top_groups = self.top_n_groups(group, self.metric("num_insurance_policies"), 3)
                for top_group in top_groups:
                    answers.add(self.format_money(sum(self.data.total_balance_by_owner.get(owner_id, 0.0) for owner_id in top_group)))
            return answers, ""

        raise ValueError(f"Unsupported aggregation question: {question}")

    def combined_count(self, owner_id: str, record_kinds: list[str]) -> int:
        mapping = {
            "bank accounts": ("bank_accounts", None),
            "credit cards": ("credit_cards", None),
            "vehicles": ("vehicles", None),
            "insurance policies": ("insurance_policies", None),
            "pets": ("pets", None),
            "internet accounts": ("internet_accounts", None),
        }
        total = 0
        for kind in record_kinds:
            key, qualifier = mapping[kind]
            total += self.data.count(owner_id, key, qualifier)
        return total

    def cross_file_question(self, row: dict) -> tuple[set[str], str]:
        question = row["input"]
        patterns = [
            (
                r"^How many total financial and property records \((.+)\) does the person with the highest total bank balance among all residents of the same state as (.+) have\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(2)), self.metric("total_balance")),
                1,
            ),
            (
                r"^Among all people who live in the same state as (.+), who owns the most pets\? How many combined (.+) does that person have\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(1)), self.metric("num_pets")),
                2,
            ),
            (
                r"^How many total records \((.+)\) does the person with the most vehicles among all residents of the same state as (.+) have\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(2)), self.metric("num_vehicles")),
                1,
            ),
            (
                r"^How many total records \((.+)\) does the person with the highest total bank balance among residents of the same state as (.+) have\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(2)), self.metric("total_balance")),
                1,
            ),
            (
                r"^How many total records \((.+)\) does the person with the most credit cards among all residents of the same state as (.+) have\?$",
                lambda m: self.select_people(self.same_state_groups(m.group(2)), self.metric("num_credit_cards")),
                1,
            ),
        ]
        for pattern, builder, record_group_index in patterns:
            match = re.match(pattern, question)
            if match:
                record_kinds = re.findall(
                    r"bank accounts|credit cards|vehicles|insurance policies|pets|internet accounts",
                    match.group(record_group_index),
                )
                winners = builder(match)
                return {
                    str(self.combined_count(owner_id, record_kinds))
                    for group in winners
                    for owner_id in group
                }, ""
        raise ValueError(f"Unsupported cross_file_counting question: {question}")

    def set_intersection_question(self, row: dict) -> tuple[set[str], str]:
        question = row["input"]

        match = re.match(r"^Among the (\d+) people with the highest total bank balance, who lives in the same state as (.+)\? How many pets does this person own\?$", question)
        if match:
            n = int(match.group(1))
            answers = set()
            for top_group in self.top_n_groups(set(self.data.all_people), self.metric("total_balance"), n):
                for state_group in self.same_state_groups(match.group(2)):
                    answers |= {str(self.data.count(owner_id, "pets")) for owner_id in top_group & state_group}
            return answers, ""

        match = re.match(r"^Among the (\d+) people with the highest total bank balance, who lives in the same state as (.+)\?$", question)
        if match:
            n = int(match.group(1))
            answers = set()
            for top_group in self.top_n_groups(set(self.data.all_people), self.metric("total_balance"), n):
                for state_group in self.same_state_groups(match.group(2)):
                    answers |= {self.data.person_name(owner_id) for owner_id in top_group & state_group}
            return answers, ""

        match = re.match(
            r"^Among people with 4 or more internet accounts who have the same blood type as (.+), who lives in the same state as (.+)\?$",
            question,
        )
        if match:
            base = self.filter_groups(self.same_blood_type_groups(match.group(1)), lambda owner_id: self.data.count(owner_id, "internet_accounts") >= 4)
            answers = {self.data.person_name(owner_id) for group in base for state_group in self.same_state_groups(match.group(2)) for owner_id in group & state_group}
            return answers, ""

        match = re.match(
            r"^Among people who own 3 or more vehicles and have the same blood type as (.+), who lives in the same state as (.+)\?$",
            question,
        )
        if match:
            base = self.filter_groups(self.same_blood_type_groups(match.group(1)), lambda owner_id: self.data.count(owner_id, "vehicles") >= 3)
            answers = {self.data.person_name(owner_id) for group in base for state_group in self.same_state_groups(match.group(2)) for owner_id in group & state_group}
            return answers, ""

        match = re.match(r"^Among people who own 3 or more pets, who lives in the same city as (.+)\?$", question)
        if match:
            pet_owners = {owner_id for owner_id in self.data.all_people if self.data.count(owner_id, "pets") >= 3}
            answers = {self.data.person_name(owner_id) for city_group in self.same_city_groups(match.group(1)) for owner_id in pet_owners & city_group}
            return answers, ""

        match = re.match(r"^Among the top (\d+) people by total bank balance who own at least one vehicle, who lives in the same state as (.+)\?$", question)
        if match:
            n = int(match.group(1))
            vehicle_owners = {owner_id for owner_id in self.data.all_people if self.data.count(owner_id, "vehicles") > 0}
            answers = set()
            for top_group in self.top_n_groups(vehicle_owners, self.metric("total_balance"), n):
                for state_group in self.same_state_groups(match.group(2)):
                    answers |= {self.data.person_name(owner_id) for owner_id in top_group & state_group}
            return answers, ""

        match = re.match(
            r"^Among people who have at least 4 internet accounts and at least 3 credit cards, who lives in the same state as (.+)\?$",
            question,
        )
        if match:
            eligible = {
                owner_id
                for owner_id in self.data.all_people
                if self.data.count(owner_id, "internet_accounts") >= 4 and self.data.count(owner_id, "credit_cards") >= 3
            }
            answers = {self.data.person_name(owner_id) for state_group in self.same_state_groups(match.group(1)) for owner_id in eligible & state_group}
            return answers, ""

        match = re.match(r"^Among the (\d+) people with the highest total bank account balance, who has the same blood type as (.+)\?$", question)
        if match:
            n = int(match.group(1))
            answers = set()
            for top_group in self.top_n_groups(set(self.data.all_people), self.metric("total_balance"), n):
                for blood_group in self.same_blood_type_groups(match.group(2)):
                    answers |= {self.data.person_name(owner_id) for owner_id in top_group & blood_group}
            return answers, ""

        match = re.match(r"^Among people who have exactly 6 bank accounts, who owns a vehicle with the same make as a vehicle owned by the owner of the pet named '([^']+)'\?$", question)
        if match:
            eligible = {owner_id for owner_id in self.data.all_people if self.data.count(owner_id, "bank_accounts") == 6}
            answers = {
                self.data.person_name(owner_id)
                for make_group in self.same_vehicle_make_groups_from_pet_owner(match.group(1))
                for owner_id in eligible & make_group
                if self.data.count(owner_id, "vehicles") > 0
            }
            return answers, ""

        raise ValueError(f"Unsupported set_intersection question: {question}")

    def evaluate(self, index: int, row: dict) -> AuditResult:
        question_type = row["agent_args"]["extra"]["question_type"]
        question = row["input"]
        ground_truth = row["ground_truth"]

        try:
            if question_type in {"multi_hop_chain", "multi_entity_comparison"}:
                valid_answers, note = self.comparison_question(row)
            elif question_type == "comparison_tiebreak":
                valid_answers, note = self.comparison_tiebreak_question(row)
            elif question_type == "aggregation":
                valid_answers, note = self.aggregation_question(row)
            elif question_type == "set_intersection":
                valid_answers, note = self.set_intersection_question(row)
            elif question_type == "temporal_reasoning":
                valid_answers, note = self.temporal_question(row)
            elif question_type == "cross_file_counting":
                valid_answers, note = self.cross_file_question(row)
            elif question_type == "negation":
                valid_answers, note = self.negation_question(row)
            else:
                raise ValueError(f"Unsupported question type: {question_type}")
        except Exception as exc:
            return AuditResult(index, "unparsed", question_type, ground_truth, [], question, str(exc))

        valid_answers = sorted(valid_answers)
        normalized_valid = {self.normalize_answer(answer) for answer in valid_answers}
        normalized_gt = self.normalize_answer(ground_truth)

        if ground_truth in valid_answers:
            status = "correct" if len(valid_answers) == 1 else "ambiguous"
        elif normalized_gt in normalized_valid:
            status = "format_issue" if len(valid_answers) == 1 else "ambiguous_format"
        elif len(valid_answers) > 1:
            status = "wrong_and_ambiguous"
        else:
            status = "wrong"

        return AuditResult(index, status, question_type, ground_truth, valid_answers, question, note)


def load_dataset_rows(path: str | Path) -> list[dict]:
    """Load parsed dataset rows from JSONL."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_dataset_rows(rows: list[dict], db_path: str | Path) -> list[AuditResult]:
    """Audit parsed dataset rows against the benchmark DB."""
    auditor = DatasetAuditor(FilesystemData(Path(db_path)))
    return [auditor.evaluate(index, row) for index, row in enumerate(rows)]


def summarize_audit_results(results: list[AuditResult]) -> dict[str, int]:
    """Count audit statuses."""
    summary: dict[str, int] = defaultdict(int)
    for result in results:
        summary[result.status] += 1
    return dict(summary)


def print_audit_report(results: list[AuditResult], dataset_label: str):
    """Print a human-readable audit report."""
    summary = summarize_audit_results(results)
    print("FILESYSTEM DATASET AUDIT")
    print("=" * 80)
    print(f"Dataset: {dataset_label}")
    print(f"Rows: {len(results)}")
    print()
    for status in sorted(summary):
        print(f"{status:20s} {summary[status]}")

    interesting = [result for result in results if result.status != "correct"]
    if interesting:
        print()
        print("Issues")
        print("-" * 80)
        for result in interesting:
            valid = ", ".join(result.valid_answers[:8]) if result.valid_answers else "<none>"
            print(f"[{result.index}] {result.status} | GT={result.ground_truth} | valid={valid}")
            print(f"  {result.question}")
            if result.note:
                print(f"  note: {result.note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "filesystem_cloud.jsonl"),
        help="Path to dataset JSONL",
    )
    parser.add_argument(
        "--db",
        default=str(Path(__file__).resolve().parent / "data" / "letta_file_bench.db"),
        help="Path to SQLite DB",
    )
    args = parser.parse_args()

    rows = load_dataset_rows(args.dataset)
    results = audit_dataset_rows(rows, args.db)
    print_audit_report(results, dataset_label=f"{args.dataset}\nDB: {args.db}")
    return 0 if all(result.status == "correct" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
