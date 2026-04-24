"""
Generate a realistic demo dataset for Omna smoke testing.
Run with: python scripts/demo_data.py
"""
import polars as pl

# A fake employee dataset with PII + varied text descriptions
df = pl.DataFrame({
    "id": list(range(1, 21)),
    "name": [
        "Alice Johnson", "Bob Smith", "Carol White", "David Brown",
        "Eve Davis", "Frank Miller", "Grace Wilson", "Henry Moore",
        "Irene Taylor", "Jack Anderson", "Karen Thomas", "Leo Jackson",
        "Mia Harris", "Nate Martin", "Olivia Garcia", "Paul Martinez",
        "Quinn Robinson", "Ruth Clark", "Sam Rodriguez", "Tina Lewis",
    ],
    "email": [
        "alice@acme.com", "bob.smith@gmail.com", "carol@acme.com", "david@corp.io",
        "eve@acme.com", "frank@gmail.com", "grace@corp.io", "henry@acme.com",
        "irene@corp.io", "jack@acme.com", "karen@gmail.com", "leo@acme.com",
        "mia@corp.io", "nate@acme.com", "olivia@gmail.com", "paul@corp.io",
        "quinn@acme.com", "ruth@gmail.com", "sam@corp.io", "tina@acme.com",
    ],
    "phone": [
        "555-867-5309", "555-123-4567", "555-234-5678", "555-345-6789",
        "555-456-7890", "555-567-8901", "555-678-9012", "555-789-0123",
        "555-890-1234", "555-901-2345", "555-012-3456", "555-111-2222",
        "555-222-3333", "555-333-4444", "555-444-5555", "555-555-6666",
        "555-666-7777", "555-777-8888", "555-888-9999", "555-999-0000",
    ],
    "role": [
        "data engineer", "machine learning engineer", "backend developer",
        "frontend developer", "devops engineer", "data scientist",
        "security engineer", "product manager", "UX designer", "data analyst",
        "backend developer", "machine learning engineer", "data engineer",
        "devops engineer", "data scientist", "frontend developer",
        "security engineer", "product manager", "UX designer", "data analyst",
    ],
    "bio": [
        "Builds ETL pipelines and data infrastructure at scale.",
        "Trains neural networks for NLP and computer vision tasks.",
        "Designs REST APIs and microservices in Python and Go.",
        "Builds React dashboards and data visualization tools.",
        "Manages Kubernetes clusters and CI/CD pipelines.",
        "Applies statistical modelling to business problems.",
        "Protects systems from intrusion and data breaches.",
        "Defines product roadmaps and works with engineering teams.",
        "Conducts user research and designs intuitive interfaces.",
        "Turns raw data into actionable business insights.",
        "Works on high-throughput backend systems and caching layers.",
        "Fine-tunes large language models for domain-specific tasks.",
        "Automates data ingestion and transformation workflows.",
        "Builds infrastructure as code and manages cloud deployments.",
        "Runs A/B tests and builds predictive models for growth.",
        "Creates accessible UI components and design systems.",
        "Performs penetration testing and vulnerability assessments.",
        "Coordinates cross-functional teams and manages stakeholders.",
        "Runs usability studies and prototypes new product concepts.",
        "Creates dashboards and self-serve analytics for teams.",
    ],
    "salary_usd": [
        95000, 140000, 115000, 105000, 120000, 130000,
        125000, 110000, 100000, 90000, 118000, 145000,
        98000, 122000, 135000, 108000, 128000, 112000,
        102000, 92000,
    ],
})

print(df)
df.write_parquet("scripts/employees.parquet")
print("\n✅ Saved to scripts/employees.parquet")
