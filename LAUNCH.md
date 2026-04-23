# Omna — Launch Strategy

## The one user we are building for
An ML engineer at a startup or scale-up building
an AI feature on a Polars DataFrame who hits the
vector database wall. They want semantic search on
100k rows and don't want to spin up Chroma, Pinecone,
or any external infrastructure just to do it.

## The hook — Version 2 (recommended)
Your DataFrame has 500,000 rows of customer feedback.

Finding the unhappy ones used to mean:
- 50 lines of keyword code
- Missing slang, typos, other languages
- Rebuilding every time schema changes

Now it's one line.
Searched in 9ms on a MacBook Air.
PII never leaves your machine.

pip install omna

## HackerNews post template
Title: Show HN: Omna — df.omna.search("angry customer")
on any Polars DataFrame, no vector DB needed

Body:
I got tired of writing 50-line keyword filters to find
unhappy customers in DataFrames. Still missing half of them.

So I built Omna:
  df.omna.search("angry customer", on="reviews", top_k=50)

No Chroma. No Pinecone. No infrastructure.
Searches 500k rows in 9ms on a MacBook Air.
PII masking built in for compliance teams.
Runs fully offline. Open source. MIT license.

GitHub: github.com/gaurjin/omna
pip install omna

Happy to answer questions.

## Launch sequence — Day 7
9am  → Push to PyPI, verify pip install works
10am → Post on HackerNews
11am → Post in Polars Discord
12pm → Post on r/Python and r/dataengineering
2pm  → Tweet benchmark video on personal account
4pm  → LinkedIn post with personal story

## All platforms
- GitHub        ← home base, most important
- HackerNews    ← biggest launch moment
- Polars Discord ← join NOW before launch
- r/Python      ← 900k members
- r/dataengineering ← most relevant community
- Twitter/X     ← benchmark video
- LinkedIn      ← personal network, finance angle
- dev.to        ← write the why I built this article
- Product Hunt  ← second wave after HackerNews

## Star trajectory
Week 1    200-500 stars   (HackerNews front page)
Month 1   1,000 stars     (first VC emails arrive)
Month 3   3,000 stars     (serious VC conversations)
Month 6   10,000 stars    (term sheets)

## Risk 3 prevention
- Reply to every GitHub issue within hours on launch day
- Get 3 beta users before launch with real testimonials
- Post real benchmark numbers — verifiable, not marketing
- Be present in communities 2 weeks before launch

## Phase 2 viral moment
df1.omna.join(df2) — semantic joins
No library does this natively on DataFrames.
This is the second HackerNews post after first 3,000 stars.

## The long term vision (for investors)
"Omna is the universal semantic layer between
enterprise data and any AI model. We started with
Polars because that is where the fastest growing
data engineering community is. Our roadmap covers
every format and every model. Think Confluent —
but for AI-ready data instead of streaming."
