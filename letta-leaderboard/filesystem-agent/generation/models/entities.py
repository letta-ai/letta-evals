"""
Data models for synthetic person entities and related records.
"""

import random
from dataclasses import asdict, dataclass, field
from typing import List

from faker import Faker
from generation.utils.id_generator import generate_unique_id
from generation.utils.uniqueness import ensure_unique_value


@dataclass
class Address:
    address_id: str
    street: str
    city: str
    state: str
    postal_code: str
    country: str

    @staticmethod
    def fake(fk: Faker):
        return Address(
            generate_unique_id("addr"), fk.street_address(), fk.city(), fk.state(), fk.postcode(), fk.current_country()
        )

    def flatten(self):
        return asdict(self)


@dataclass
class BankAccount:
    account_id: str
    bank_name: str
    routing: str
    account_no: str
    balance: float
    currency: str

    @staticmethod
    def fake(fk: Faker):
        bank_name = ensure_unique_value(lambda: fk.company() + " Bank", "company_names")
        account_no = ensure_unique_value(lambda: fk.bban(), "account_numbers")
        return BankAccount(
            generate_unique_id("acct"), bank_name, fk.aba(), account_no, round(random.uniform(100, 50_000), 2), "USD"
        )

    def flatten(self):
        d = asdict(self)
        d["balance"] = f"{self.balance:.2f}"
        return d


@dataclass
class Employment:
    employment_id: str
    employer: str
    job_title: str
    start_date: str
    salary: float
    currency: str

    @staticmethod
    def fake(fk: Faker):
        employer = ensure_unique_value(lambda: fk.company(), "company_names")
        return Employment(
            generate_unique_id("emp"),
            employer,
            fk.job(),
            str(fk.date_between(start_date="-10y", end_date="today")),
            round(random.uniform(40_000, 250_000), 2),
            "USD",
        )

    def flatten(self):
        d = asdict(self)
        d["salary"] = f"{self.salary:.2f}"
        return d


@dataclass
class CreditCard:
    card_id: str
    provider: str
    number: str
    expire: str
    cvc: str

    @staticmethod
    def fake(fk: Faker):
        card_number = ensure_unique_value(lambda: fk.credit_card_number(), "credit_card_numbers")
        return CreditCard(
            generate_unique_id("card"),
            fk.credit_card_provider(),
            card_number,
            fk.credit_card_expire(),
            fk.credit_card_security_code(),
        )

    def flatten(self):
        return asdict(self)


@dataclass
class Vehicle:
    vehicle_id: str
    make: str
    model: str
    year: int
    license_plate: str

    @staticmethod
    def fake(fk: Faker):
        make = fk.vehicle_make() if hasattr(fk, "vehicle_make") else fk.company()
        model = fk.vehicle_model() if hasattr(fk, "vehicle_model") else fk.color_name()
        license_plate = ensure_unique_value(lambda: fk.license_plate(), "license_plates")
        return Vehicle(generate_unique_id("veh"), make, model, random.randint(1995, 2025), license_plate)

    def flatten(self):
        return asdict(self)


@dataclass
class Pet:
    pet_id: str
    name: str
    species: str
    breed: str

    @staticmethod
    def fake(fk: Faker):
        return Pet(
            generate_unique_id("pet"),
            fk.first_name(),
            random.choice(["Dog", "Cat", "Bird", "Fish", "Rabbit"]),
            random.choice(["Mixed", "Purebred", "Unknown"]),
        )

    def flatten(self):
        return asdict(self)


@dataclass
class InternetAccount:
    net_id: str
    username: str
    email: str
    url: str
    password: str

    @staticmethod
    def fake(fk: Faker):
        username = ensure_unique_value(lambda: fk.user_name(), "usernames")
        email = ensure_unique_value(lambda: fk.free_email(), "emails")
        return InternetAccount(generate_unique_id("net"), username, email, fk.url(), fk.password(length=12))

    def flatten(self):
        return asdict(self)


@dataclass
class InsurancePolicy:
    policy_id: str
    insurer: str
    policy_type: str
    policy_number: str
    expires: str

    @staticmethod
    def fake(fk: Faker):
        insurer = ensure_unique_value(lambda: fk.company(), "company_names")
        policy_number = ensure_unique_value(lambda: fk.bothify(text="???-########"), "policy_numbers")
        return InsurancePolicy(
            generate_unique_id("ins"),
            insurer,
            random.choice(["Home", "Auto", "Health", "Life"]),
            policy_number,
            str(fk.future_date(end_date="+5y")),
        )

    def flatten(self):
        return asdict(self)


@dataclass
class MedicalRecord:
    record_id: str
    ssn: str
    blood_type: str
    condition: str

    @staticmethod
    def fake(fk: Faker):
        ssn = ensure_unique_value(lambda: fk.ssn(), "ssns")
        return MedicalRecord(
            generate_unique_id("med"),
            ssn,
            random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
            random.choice(["None", "Diabetes", "Hypertension", "Asthma", "Allergy"]),
        )

    def flatten(self):
        return asdict(self)


@dataclass
class Person:
    person_id: str
    first: str
    last: str
    dob: str
    email: str
    phone: str
    locale: str
    addresses: List[Address] = field(default_factory=list)
    accounts: List[BankAccount] = field(default_factory=list)
    employments: List[Employment] = field(default_factory=list)
    credit_cards: List[CreditCard] = field(default_factory=list)
    vehicles: List[Vehicle] = field(default_factory=list)
    pets: List[Pet] = field(default_factory=list)
    net_accounts: List[InternetAccount] = field(default_factory=list)
    insurances: List[InsurancePolicy] = field(default_factory=list)
    medical_records: List[MedicalRecord] = field(default_factory=list)

    @property
    def full_name(self):
        return f"{self.first} {self.last}"

    @staticmethod
    def fake(fk: Faker, locale: str):
        """Generate a Person compatible with any Faker version."""
        if hasattr(fk, "localcontext"):
            with fk.localcontext(locale):
                active_fk = fk
        else:
            active_fk = Faker(locale)

        # Generate unique full name
        def generate_person_name():
            first = active_fk.first_name()
            last = active_fk.last_name()
            return f"{first} {last}"

        full_name = ensure_unique_value(generate_person_name, "person_names")
        first_name, last_name = full_name.split(" ", 1)

        email = ensure_unique_value(lambda: active_fk.email(), "emails")
        phone = ensure_unique_value(lambda: active_fk.phone_number(), "phone_numbers")

        return Person(
            generate_unique_id("pers"),
            first_name,
            last_name,
            str(active_fk.date_of_birth(minimum_age=18, maximum_age=90)),
            email,
            phone,
            locale,
        )

    def flatten(self):
        return {
            "person_id": self.person_id,
            "full_name": self.full_name,
            "dob": self.dob,
            "email": self.email,
            "phone": self.phone,
            "locale": self.locale,
            "addresses": [a.flatten() for a in self.addresses],
            "bank_accounts": [a.flatten() for a in self.accounts],
            "employments": [e.flatten() for e in self.employments],
            "credit_cards": [c.flatten() for c in self.credit_cards],
            "vehicles": [v.flatten() for v in self.vehicles],
            "pets": [p.flatten() for p in self.pets],
            "internet_accounts": [n.flatten() for n in self.net_accounts],
            "insurances": [i.flatten() for i in self.insurances],
            "medical_records": [m.flatten() for m in self.medical_records],
        }


# Entity mapping for file generation
ENTITY_MAP = {
    "addresses": ("addresses.txt", Address, "address_id"),
    "accounts": ("bank_accounts.txt", BankAccount, "account_id"),
    "employments": ("employments.txt", Employment, "employment_id"),
    "credit_cards": ("credit_cards.txt", CreditCard, "card_id"),
    "vehicles": ("vehicles.txt", Vehicle, "vehicle_id"),
    "pets": ("pets.txt", Pet, "pet_id"),
    "net_accounts": ("internet_accounts.txt", InternetAccount, "net_id"),
    "insurances": ("insurance_policies.txt", InsurancePolicy, "policy_id"),
    "medical_records": ("medical_records.txt", MedicalRecord, "record_id"),
}
