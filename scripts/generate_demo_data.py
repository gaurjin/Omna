import polars as pl
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
random.seed(42)

N = 500_000

print(f"Generating {N:,} rows...")

ISSUES = [
    "I was charged twice for my subscription and need a refund.",
    "Cannot log in even after resetting my password.",
    "The dashboard takes 30 seconds to load.",
    "My data from the CSV import is missing.",
    "I need to delete customer data under GDPR.",
    "API response times have gone from 200ms to 4 seconds.",
    "I was auto-renewed without a reminder email.",
    "Two-factor authentication is not sending the SMS code.",
    "Bulk export is timing out for anything over 10,000 rows.",
    "Changes are not syncing across devices.",
    "I need a SOC 2 report for vendor onboarding.",
    "The invoice shows a different amount than quoted.",
    "My account is locked after too many login attempts.",
    "The webhook is firing duplicate events.",
    "I cancelled but am still being charged.",
]

CATEGORIES = ["billing", "authentication", "performance", "data_sync", "compliance"]
PRIORITIES = ["low", "medium", "high", "critical"]

df = pl.DataFrame({
    "ticket_id":     list(range(100_000, 100_000 + N)),
    "customer_name": [fake.name()                          for _ in range(N)],
    "email":         [fake.email()                         for _ in range(N)],
    "phone":         [fake.numerify("(###) ###-####")      for _ in range(N)],
    "ssn":           [fake.ssn()                           for _ in range(N)],
    "issue":         [random.choice(ISSUES)                for _ in range(N)],
    "category":      [random.choice(CATEGORIES)            for _ in range(N)],
    "priority":      [random.choice(PRIORITIES)            for _ in range(N)],
    "resolved":      [random.random() > 0.35               for _ in range(N)],
})

import os
os.makedirs("data", exist_ok=True)
df.write_parquet("data/demo_500k.parquet")
print(f"✓ Saved 500,000 rows to data/demo_500k.parquet")
