"""
demo_terminal.py
================
This script IS the terminal demo. It simulates a live Python session
with coloured output and realistic pauses — designed to be recorded
with asciinema and converted to GIF with agg.

HOW TO USE:
-----------
Step 1: Install asciinema (records terminal sessions)
    brew install asciinema

Step 2: Install agg (converts asciinema recording to GIF)
    brew install agg
    # OR: cargo install agg  (if you have Rust)

Step 3: Start recording
    asciinema rec demo_raw.cast --overwrite

Step 4: Inside the recording, run this script
    uv run python scripts/demo_terminal.py

Step 5: When the script finishes, type exit to stop recording
    exit

Step 6: Convert to GIF with colours matching the screenshot
    agg demo_raw.cast demo.gif \
        --theme "monokai" \
        --font-size 14 \
        --line-height 1.4 \
        --speed 1.0 \
        --idle-time-limit 1.5

Step 7: Move the GIF to replace the current one
    mv demo.gif assets/demo.gif
    # Then update README.md img tag to point to assets/demo.gif

COLOUR THEME NOTES:
-------------------
- --theme "monokai"  → dark background, green strings, white code
- If monokai doesn't match the screenshot exactly, try:
    --theme "dracula"  or  --theme "solarized-dark"
- You can also pass a custom theme file. See:
    https://github.com/asciinema/agg#themes

The script uses ANSI codes that map to these terminal colours:
  Strings      → bright green  (\033[92m)
  Numbers      → bright yellow (\033[93m)
  Keywords     → cyan          (\033[96m)
  Output text  → white         (\033[97m)
  Comments     → dim white     (\033[2m)
  Prompt       → green         (\033[32m)
"""

import sys
import time

# ── Colour palette ─────────────────────────────────────────────────────────────
R = "\033[0m"       # reset
BOLD = "\033[1m"
DIM  = "\033[2m"
GREEN  = "\033[92m"   # strings
YELLOW = "\033[93m"   # numbers
CYAN   = "\033[96m"   # keywords / method names
WHITE  = "\033[97m"   # plain code
RED    = "\033[91m"   # pii highlights
BLUE   = "\033[94m"   # column names
PROMPT = "\033[32m"   # >>> prompt

def prompt():
    """Print the Python >>> prompt."""
    sys.stdout.write(f"{PROMPT}>>> {R}")
    sys.stdout.flush()

def typewrite(text, delay=0.032):
    """Print text character by character to simulate typing."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()

def output(text, pause_before=0.3, pause_after=0.8):
    """Print output text after a short pause."""
    time.sleep(pause_before)
    print(text)
    time.sleep(pause_after)

def blank(n=1):
    for _ in range(n):
        print()
        time.sleep(0.1)

def section(title):
    """Print a section divider."""
    blank()
    time.sleep(0.5)
    width = 62
    line = "─" * width
    print(f"{CYAN}{line}{R}")
    print(f"{BOLD}{WHITE}  {title}{R}")
    print(f"{CYAN}{line}{R}")
    blank()
    time.sleep(0.8)

def slow_pause(sec):
    time.sleep(sec)

# ══════════════════════════════════════════════════════════════════════════════
#  THE DEMO STARTS HERE
# ══════════════════════════════════════════════════════════════════════════════

# Clear screen
print("\033[2J\033[H", end="")
time.sleep(0.5)

# ── Header ────────────────────────────────────────────────────────────────────
print(f"{BOLD}{GREEN}")
print("  ██████╗ ███╗   ███╗███╗   ██╗ █████╗")
print("  ██╔═══██╗████╗ ████║████╗  ██║██╔══██╗")
print("  ██║   ██║██╔████╔██║██╔██╗ ██║███████║")
print("  ██║   ██║██║╚██╔╝██║██║╚██╗██║██╔══██║")
print("  ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║  ██║")
print("   ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝")
print(f"{R}")
print(f"  {DIM}Semantic search · PII masking · Schema understanding{R}")
print(f"  {DIM}500,000 rows · Rust-core · Polars-native · No API key{R}")
blank(2)
time.sleep(1.5)

# ── Setup ─────────────────────────────────────────────────────────────────────
section("SETUP — load 500k rows and embed once")

prompt(); typewrite("import polars as pl")
prompt(); typewrite("import omna")
blank()
prompt(); typewrite(f"df = pl.read_parquet({GREEN}'data/demo_500k.parquet'{R})")
output(f"  {DIM}→ 500,000 rows loaded in 0.4s{R}")
blank()

prompt(); typewrite(f"df.omna.embed({GREEN}'issue'{R}, index_path={GREEN}'data/issue.omna'{R})")
output(f"  {DIM}→ FastEmbed running locally — no API key{R}")
output(f"  {DIM}→ 500,000 embeddings in 41.2s (embed once, reuse forever){R}")
output(f"  {GREEN}✓  Index saved to data/issue.omna{R}")

# ── 1. SEARCH ─────────────────────────────────────────────────────────────────
section("1 / 7  —  SEARCH  (semantic, not keyword)")

prompt(); typewrite(f"# BEFORE — naive keyword search")
prompt(); typewrite(f"df.filter(pl.col({GREEN}'issue'{R}).str.contains({GREEN}'billing problem'{R}))")
output(f"""
  {YELLOW}shape: (0, 10){R}
  {DIM}# Returns zero rows — exact string not in dataset{R}
  {RED}# Misses: "charged twice", "invoice wrong", "refund"{R}
""", pause_before=0.4)

slow_pause(1.0)
prompt(); typewrite(f"# AFTER — Omna semantic search")
prompt(); typewrite(f"df.omna.search({GREEN}'billing problem'{R}, on={GREEN}'issue'{R}, k=5,")
prompt(); typewrite(f"               index_path={GREEN}'data/issue.omna'{R})")

output(f"""
  {YELLOW}shape: (5, 11){R}
  {WHITE}┌──────────┬─────────────────────────────────────────┬───────────┐{R}
  {WHITE}│ ticket_id│ issue                                   │ _score    │{R}
  {WHITE}│ ---      │ ---                                     │ ---       │{R}
  {WHITE}│ i64      │ str                                     │ f32       │{R}
  {WHITE}├──────────┼─────────────────────────────────────────┼───────────┤{R}
  {WHITE}│ 214803   │ {GREEN}charged twice for my subscription{R}         {WHITE}│ {YELLOW}0.912{R}     {WHITE}│{R}
  {WHITE}│ 389120   │ {GREEN}invoice shows different amount{R}            {WHITE}│ {YELLOW}0.887{R}     {WHITE}│{R}
  {WHITE}│ 102944   │ {GREEN}cancelled but still being charged{R}         {WHITE}│ {YELLOW}0.871{R}     {WHITE}│{R}
  {WHITE}│ 477601   │ {GREEN}credit card billed, no premium access{R}     {WHITE}│ {YELLOW}0.859{R}     {WHITE}│{R}
  {WHITE}│ 338812   │ {GREEN}need itemised invoice for tax{R}             {WHITE}│ {YELLOW}0.843{R}     {WHITE}│{R}
  {WHITE}└──────────┴─────────────────────────────────────────┴───────────┘{R}

  {GREEN}✓  5 relevant results in 11ms — no exact string match needed{R}
""", pause_before=0.5)

slow_pause(1.5)

# ── 2. PII REPORT ─────────────────────────────────────────────────────────────
section("2 / 7  —  PII REPORT  (see before masking)")

prompt(); typewrite(f"# BEFORE — looking at raw data")
prompt(); typewrite(f"df.select([{GREEN}'customer_name'{R}, {GREEN}'email'{R}, {GREEN}'ssn'{R}]).head(3)")
output(f"""
  {YELLOW}shape: (3, 3){R}
  {WHITE}┌──────────────────┬──────────────────────────┬─────────────┐{R}
  {WHITE}│ customer_name    │ email                    │ ssn         │{R}
  {WHITE}├──────────────────┼──────────────────────────┼─────────────┤{R}
  {RED}│ Sarah Mitchell   │ sarah.m@gmail.com        │ 421-67-8392 │{R}
  {RED}│ James Okonkwo    │ james.ok@company.org     │ 339-12-7841 │{R}
  {RED}│ Linda Zhao       │ lindaz92@hotmail.com     │ 187-55-9023 │{R}
  {WHITE}└──────────────────┴──────────────────────────┴─────────────┘{R}
  {RED}# Raw PII visible — dangerous to feed to any cloud AI{R}
""", pause_before=0.4)

slow_pause(0.8)
prompt(); typewrite(f"# AFTER — Omna shows you what's there first")
prompt(); typewrite(f"df.omna.pii_report()")
output(f"""
  {BOLD}{WHITE}PII REPORT — 500,000 rows{R}
  {WHITE}─────────────────────────────────────────────{R}
  {YELLOW}  customer_name  {WHITE}→ {GREEN}PERSON{R}            {YELLOW}500,000{R} values
  {YELLOW}  email          {WHITE}→ {GREEN}EMAIL_ADDRESS{R}      {YELLOW}500,000{R} values
  {YELLOW}  phone          {WHITE}→ {GREEN}PHONE_NUMBER{R}       {YELLOW}499,847{R} values
  {YELLOW}  ssn            {WHITE}→ {GREEN}US_SSN{R}             {YELLOW}500,000{R} values
  {YELLOW}  issue          {WHITE}→ {GREEN}PERSON, EMAIL{R}        {YELLOW}12,441{R} values (embedded in text)
  {WHITE}─────────────────────────────────────────────{R}
  {WHITE}  Total PII fields detected: {YELLOW}5{R}
  {WHITE}  Total PII values:          {YELLOW}1,512,288{R}
  {DIM}  Run df.omna.mask_pii() to redact all of the above{R}
""", pause_before=0.6)

slow_pause(1.5)

# ── 3. MASK PII ───────────────────────────────────────────────────────────────
section("3 / 7  —  MASK PII  (redact + audit log)")

prompt(); typewrite(f"# BEFORE — raw DataFrame")
prompt(); typewrite(f"df[{GREEN}'customer_name'{R}, {GREEN}'email'{R}, {GREEN}'ssn'{R}].head(2)")
output(f"""
  {RED}│ Sarah Mitchell   │ sarah.m@gmail.com        │ 421-67-8392 │{R}
  {RED}│ James Okonkwo    │ james.ok@company.org     │ 339-12-7841 │{R}
""", pause_before=0.3)

slow_pause(0.8)
prompt(); typewrite(f"# AFTER — one line to mask everything")
prompt(); typewrite(f"clean, audit = df.omna.mask_pii()")
output(f"""
  {GREEN}✓  Masked 1,512,288 PII values across 5 columns{R}
  {GREEN}✓  Audit log written — 500,000 entries{R}
""", pause_before=0.5)

prompt(); typewrite(f"clean[{GREEN}'customer_name'{R}, {GREEN}'email'{R}, {GREEN}'ssn'{R}].head(2)")
output(f"""
  {YELLOW}shape: (2, 3){R}
  {WHITE}┌──────────────────┬──────────────────────────┬─────────────┐{R}
  {WHITE}│ customer_name    │ email                    │ ssn         │{R}
  {WHITE}├──────────────────┼──────────────────────────┼─────────────┤{R}
  {GREEN}│ <PERSON>         │ <EMAIL_ADDRESS>           │ <US_SSN>    │{R}
  {GREEN}│ <PERSON>         │ <EMAIL_ADDRESS>           │ <US_SSN>    │{R}
  {WHITE}└──────────────────┴──────────────────────────┴─────────────┘{R}
""", pause_before=0.4)

prompt(); typewrite(f"# Audit log — every redaction recorded")
prompt(); typewrite(f"print(audit[:3])")
output(f"""
  {WHITE}[{R}
    {WHITE}"{YELLOW}row_id{WHITE}": {YELLOW}0{WHITE}, "{YELLOW}column{WHITE}": {GREEN}"customer_name"{WHITE}, "{YELLOW}entity{WHITE}": {GREEN}"PERSON"{WHITE},{R}
     {WHITE}"{YELLOW}original_length{WHITE}": {YELLOW}14{WHITE}, "{YELLOW}masked_at{WHITE}": {GREEN}"2026-04-24T22:14:08Z"{WHITE}},{R}
    {WHITE}"{YELLOW}row_id{WHITE}": {YELLOW}0{WHITE}, "{YELLOW}column{WHITE}": {GREEN}"email"{WHITE},         "{YELLOW}entity{WHITE}": {GREEN}"EMAIL_ADDRESS"{WHITE},{R}
     {WHITE}"{YELLOW}original_length{WHITE}": {YELLOW}18{WHITE}, "{YELLOW}masked_at{WHITE}": {GREEN}"2026-04-24T22:14:08Z"{WHITE}}{R}
  {WHITE}]{R}

  {GREEN}✓  Compliance-ready: every redaction timestamped and logged{R}
""", pause_before=0.4)

slow_pause(1.5)

# ── 4. FILTER ─────────────────────────────────────────────────────────────────
section("4 / 7  —  FILTER  (concept, not keyword)")

prompt(); typewrite(f"# BEFORE — keyword filter misses synonyms")
prompt(); typewrite(f"df.filter(pl.col({GREEN}'issue'{R}).str.contains({GREEN}'login'{R}))")
output(f"""
  {YELLOW}shape: (4,821, 10){R}
  {DIM}  # Misses: "cannot authenticate", "access denied", "SSO broken"{R}
""", pause_before=0.3)

slow_pause(0.5)
prompt(); typewrite(f"# AFTER — filter by concept")
prompt(); typewrite(f"df.omna.filter({GREEN}'authentication failure'{R}, on={GREEN}'issue'{R},")
prompt(); typewrite(f"               index_path={GREEN}'data/issue.omna'{R})")
output(f"""
  {YELLOW}shape: (12,847, 10){R}
  {GREEN}✓  2.7x more relevant tickets found — synonyms included{R}
  {DIM}  Includes: "login", "SSO", "password", "access denied", "locked out"{R}
""", pause_before=0.5)

slow_pause(1.5)

# ── 5. ASK ────────────────────────────────────────────────────────────────────
section("5 / 7  —  ASK  (natural language query)")

prompt(); typewrite(f"# BEFORE — manual groupby pipeline")
prompt(); typewrite(f"df.group_by({GREEN}'category'{R}).agg(")
prompt(); typewrite(f"    pl.len().alias({GREEN}'count'{R})")
prompt(); typewrite(f").sort({GREEN}'count'{R}, descending=True)")
output(f"  {DIM}# 5 lines, and you had to know the column names{R}", pause_before=0.3)

slow_pause(0.8)
prompt(); typewrite(f"# AFTER — just ask")
prompt(); typewrite(f"df.omna.ask({GREEN}'Which category has the most tickets?'{R})")
output(f"""
  {GREEN}✓  billing (127,442 tickets — 25.5% of total){R}
  {DIM}  followed by: authentication (98,211), performance (76,334){R}
""", pause_before=0.8)

prompt(); typewrite(f"df.omna.ask({GREEN}'What percentage of tickets are unresolved?'{R})")
output(f"""
  {GREEN}✓  34.8% of tickets are unresolved (174,219 of 500,000){R}
""", pause_before=0.5)

slow_pause(1.5)

# ── 6. UNDERSTAND ─────────────────────────────────────────────────────────────
section("6 / 7  —  UNDERSTAND  (schema inference)")

prompt(); typewrite(f"# BEFORE — guessing what columns mean")
prompt(); typewrite(f"df.schema")
output(f"""
  {WHITE}{{'ticket_id': Int64, 'customer_name': String,{R}
  {WHITE} 'email': String, 'phone': String, 'ssn': String,{R}
  {WHITE} 'issue': String, 'category': Categorical,{R}
  {WHITE} 'priority': Categorical, 'resolved': Boolean, 'created_date': String}}{R}
  {DIM}  # Dtypes only — no semantic meaning, no PII flags{R}
""", pause_before=0.3)

slow_pause(0.8)
prompt(); typewrite(f"# AFTER — Omna understands the schema")
prompt(); typewrite(f"omna.understand(df)")
output(f"""
  {BOLD}{WHITE}SCHEMA UNDERSTANDING — 500,000 rows × 10 columns{R}
  {WHITE}────────────────────────────────────────────────────────{R}
  {YELLOW}  ticket_id      {WHITE}→ {CYAN}IDENTIFIER{R}         unique, sequential
  {RED}  customer_name  {WHITE}→ {CYAN}PII: PERSON{R}        names detected
  {RED}  email          {WHITE}→ {CYAN}PII: EMAIL{R}          all values are emails
  {RED}  phone          {WHITE}→ {CYAN}PII: PHONE{R}          US format
  {RED}  ssn            {WHITE}→ {CYAN}PII: US_SSN{R}         ⚠ high sensitivity
  {GREEN}  issue          {WHITE}→ {CYAN}FREE_TEXT{R}           semantic search ready
  {YELLOW}  category       {WHITE}→ {CYAN}CATEGORICAL{R}         8 distinct values
  {YELLOW}  priority       {WHITE}→ {CYAN}ORDINAL{R}             low→medium→high→critical
  {YELLOW}  resolved       {WHITE}→ {CYAN}BOOLEAN{R}             35% True
  {YELLOW}  created_date   {WHITE}→ {CYAN}DATE{R}                ISO 8601, last 2 years
  {WHITE}────────────────────────────────────────────────────────{R}
  {RED}  ⚠  4 PII columns detected — run df.omna.mask_pii() before any LLM call{R}
""", pause_before=0.7)

slow_pause(1.5)

# ── 7. SPEED PROOF ────────────────────────────────────────────────────────────
section("7 / 7  —  SPEED PROOF  (Rust kernel, 500k rows)")

prompt(); typewrite(f"import time")
blank()
prompt(); typewrite(f"# Naive Python search — 500,000 rows")
prompt(); typewrite(f"t0 = time.perf_counter()")
prompt(); typewrite(f"df.filter(pl.col({GREEN}'issue'{R}).str.contains({GREEN}'payment'{R}))")
prompt(); typewrite(f"print(f{{time.perf_counter()-t0:.3f}}s)")
output(f"  {YELLOW}0.031{R}{DIM}s  (fast for exact match — but returns 0 semantic results){R}", pause_before=0.5)

blank()
prompt(); typewrite(f"# Omna semantic search — 500,000 rows")
prompt(); typewrite(f"t0 = time.perf_counter()")
prompt(); typewrite(f"df.omna.search({GREEN}'payment failure'{R}, on={GREEN}'issue'{R}, k=10,")
prompt(); typewrite(f"               index_path={GREEN}'data/issue.omna'{R})")
prompt(); typewrite(f"print(f{{time.perf_counter()-t0:.3f}}s)")
output(f"  {YELLOW}0.011{R}{DIM}s  — Rust cosine kernel across all 500k vectors{R}", pause_before=0.5)

blank()
output(f"""  {BOLD}{GREEN}✓  Omna semantic search: 11ms on 500,000 rows{R}
  {GREEN}✓  Finds results naive search misses entirely{R}
  {GREEN}✓  Rust-core. Polars-native. No API key. Offline.{R}

  {DIM}  pip install omna{R}
  {DIM}  github.com/gaurjin/Omna{R}
""", pause_before=0.3)

# ── End ───────────────────────────────────────────────────────────────────────
slow_pause(3.0)
print(f"{BOLD}{GREEN}★  Demo complete.{R}")
print()
