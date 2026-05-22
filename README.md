# Media Coverage of Zohran Mamdani (2025 NYC Mayoral Election)

A data science project studying how North American news outlets covered Zohran Mamdani during the 2025 New York City mayoral election. The election was historic: Mamdani became the city’s first Muslim and South Asian mayor-elect.

## Overview

This repository contains:

- A corpus of **810 annotated news articles** from **50+ U.S. and Canadian outlets** (September–November 2025)
- Manual labels for **topic category**, **sentiment**, and **political orientation** of each source
- Exploratory visualizations in `analysis.ipynb`
- Topic-stratified article exports in `categories/`
- LLM-generated thematic summaries per category
- A written report: [`Final_Report.pdf`](Final_Report.pdf)

## Research questions

- How is coverage distributed across topics (campaign, policy, controversy, endorsements, etc.)?
- What sentiment and framing appear in articles from left-, center-, and right-leaning outlets?
- Which sources publish most frequently, and how does coverage change over the election period?

## Repository structure

```
.
├── README.md
├── Final_Report.pdf          # Full written analysis
├── analysis.ipynb              # Visualizations and EDA
├── typology.txt                # Category definitions used for annotation
├── categories/                 # Articles split by primary topic
│   ├── campaign.csv
│   ├── civil_policy.csv
│   ├── controversy.csv
│   ├── endorsements.csv
│   ├── governance.csv
│   ├── social_cause.csv
│   ├── media.csv
│   ├── category_summaries.csv  # LLM summaries (one row per category)
│   └── category_summaries.json
└── code/
    ├── article_utils.py        # Shared parsing, filtering, CSV helpers
    ├── mediastack.py           # Fetch new articles via MediaStack API
    └── llm_summaries.py        # TF-IDF + Gemini summaries per category
```

## Dataset

| Field | Description |
|-------|-------------|
| `title`, `description` | Headline and lede/snippet |
| `url`, `published_at`, `source` | Article link, date, outlet |
| `categories` | Primary topic (see typology below) |
| `sentiment` | `Positive`, `Negative`, or `Neutral` |
| `political_orientation` | Outlet leaning: `left`, `center`, or `right` (where assigned) |

**Coverage period:** September 1 – November 15, 2025  
**Geography:** Primarily U.S. and Canada, English-language outlets

The `categories/` folder holds **topic-stratified CSV exports** (~455 unique articles; a small number appear in more than one file when they fit multiple themes). The full merged dataset used in the notebook is expected at the project root as `articles_annotated.csv` (810 rows). If that file is not present locally, combine the category files or use the exports in `categories/` directly.

## Topic typology

Articles were classified into six thematic categories. Full definitions and inclusion/exclusion rules are in [`typology.txt`](typology.txt).

| Category | Focus |
|----------|--------|
| **Endorsements** | Public support from politicians, unions, organizations, or prominent figures |
| **Civil Policy** | Substance of specific policies and legislative proposals |
| **Controversy** | Scandals, attacks, ethical allegations, or reputational conflict |
| **Campaign** | Polling, fundraising, debates, strategy, and electoral mechanics |
| **Social Cause** | Identity, immigration, religion, and broader social/historical framing |
| **Governance** | Post-election transition, appointments, and administrative organization |

## Quick start

### 1. Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pandas matplotlib seaborn numpy scikit-learn requests
```

For optional scripts:

```bash
pip install google-genai   # llm_summaries.py (Gemini)
```

### 2. Run the analysis notebook

```bash
jupyter notebook analysis.ipynb
```

Place `articles_annotated.csv` in the project root if you want to reproduce all notebook outputs against the full 810-article corpus. The notebook builds charts for category distribution, sentiment, political orientation, source frequency, and time trends.

### 3. Regenerate category summaries (optional)

Requires a Gemini API key (via `google-genai` client configuration) and `articles_annotated.csv` for global TF-IDF:

```bash
cd code
python llm_summaries.py
```

Outputs are written to `categories/category_summaries.csv` and `categories/category_summaries.json`.

### 4. Fetch additional articles (optional)

Create `code/api_keys.py` with your MediaStack token (see error message in `mediastack.py` for the expected variable name), then:

```bash
cd code
python mediastack.py
```

This fetches day-by-day results for the keyword `"Zohran Mamdani"` and saves to `articles_mediastack.json` / `.csv` without overwriting the main corpus.

## Key outputs

- **`analysis.ipynb`** — Distribution plots, sentiment heatmaps, source and orientation breakdowns, temporal trends
- **`categories/category_summaries.json`** — Per-category TF-IDF keywords and LLM narrative summaries
- **`Final_Report.pdf`** — Complete project write-up

## Methods (summary)

1. **Collection** — News articles mentioning Mamdani from North American outlets during the election window.
2. **Annotation** — Human-coded category, sentiment, and source political orientation using the typology in `typology.txt`.
3. **Analysis** — Descriptive statistics and visual exploration in Jupyter.
4. **Summarization** — Category-level TF-IDF (global IDF, category TF) plus Gemini-generated thematic summaries from title + description samples.
