# Public Health Intelligence Platform — Complete Project Documentation

## Table of Contents
1. [Business Problem & Questions](#1-business-problem--questions)
2. [Data Source](#2-data-source)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Phase 1: Data Ingestion](#4-phase-1-data-ingestion)
5. [Phase 2: Feature Engineering](#5-phase-2-feature-engineering)
6. [Phase 3: ML Models](#6-phase-3-ml-models)
7. [Phase 4: Cortex LLM Narratives](#7-phase-4-cortex-llm-narratives)
8. [Phase 5: Streamlit in Snowflake Dashboard](#8-phase-5-streamlit-in-snowflake-dashboard)
9. [Data Split Strategy](#9-data-split-strategy)
10. [Snowflake Object Inventory](#10-snowflake-object-inventory)
11. [Hackathon Requirements vs Implementation](#11-hackathon-requirements-vs-implementation)
12. [Fairness & Bias Documentation](#12-fairness--bias-documentation)
13. [Key Epidemiological Concepts](#13-key-epidemiological-concepts)
14. [Cortex AI Briefs — Design & Implementation](#14-cortex-ai-briefs--design--implementation)

---

## 1. Business Problem & Questions

### Problem Statement

During a pandemic, health officials are overwhelmed with millions of rows of raw case, death, and vaccination data but need quick, actionable answers without requiring them to be data scientists. The system transforms raw epidemiological data into a decision-support intelligence platform that answers: **what is happening, what is coming, and what to do about it**.

**Target Users**: Government health ministers, WHO officials, public health analysts, hospital administrators, UN humanitarian aid teams.

---

### The 9 Business Questions

#### Original 4 — Core Questions

| # | Business Question | Decision It Drives | How the System Answers |
|---|---|---|---|
| **Q1** | Is this country getting better or worse right now? | Escalate or stand down the national response | `EPIDEMIC_PHASE` (6-class) + `ACCELERATION` (2nd derivative) + `TREND_DIRECTION` on the Epi Deep-Dive tab |
| **Q2** | What will case counts look like 30 days from now? | Hospital bed allocation, medical supply procurement, staffing plans | `SNOWFLAKE.ML.FORECAST` → 30-day projection with `LOWER_BOUND` / `UPPER_BOUND` confidence intervals |
| **Q3** | Where are the sudden, unexpected spikes happening right now? | Trigger emergency field investigation before it becomes a crisis | `SNOWFLAKE.ML.ANOMALY_DETECTION` → `IS_ANOMALY`, `ANOMALY_DIRECTION`, `DEVIATION_PCT` |
| **Q4** | Which of these 15 countries needs immediate international intervention? | WHO resource deployment, international aid prioritization | 8-factor composite Risk Score (0–100) → HIGH 🔴 / MODERATE 🟡 / LOW 🟢 tier classification |

#### 5 New High-Value Questions

| # | Business Question | Decision It Drives | How the System Answers |
|---|---|---|---|
| **Q5** | How fast is this outbreak doubling, and is it faster or slower than the last wave? | Declare public health emergency — most governments trigger at <14-day doubling time | `DOUBLING_TIME_DAYS` current value vs. its 90-day historical average per country |
| **Q6** | Which countries have Rt > 1 AND a worsening 30-day forecast simultaneously — the "double danger zone"? | Highest-priority intervention list: not just bad now, but getting worse | JOIN `RISK_TIERS` (Rt > 1) with `FORECASTS` (forecast end > current 7D avg) |
| **Q7** | What is the projected peak case load and when exactly does it hit? | ICU bed planning — hospitals need 2–3 weeks lead time to staff up | `MAX(FORECAST)` + peak `DATE` from Snowflake ML output per country |
| **Q8** | Can I trust this country's forecast, or is the data too noisy to act on? | Whether to publish model output in an official government report | MAPE from `model!SHOW_EVALUATION_METRICS()` cross-referenced with `SIGNAL_TO_NOISE` ratio |
| **Q9** | Give me a plain-English 30-second brief I can read before a press conference. | Public communication — what the minister says to the press today | `SNOWFLAKE.CORTEX.COMPLETE()` 3-sentence narrative: current phase + trajectory + recommended action |

---

## 2. Data Source

### Primary Dataset

| Field | Value |
|---|---|
| **Source** | Starschema COVID-19 Epidemiological Data |
| **Origin** | Snowflake Marketplace (free listing) |
| **Underlying Data** | Johns Hopkins University CSSE COVID-19 Dataset |
| **License** | CC BY 4.0 (public domain) |
| **Format** | Long-format SQL tables — already query-ready in Snowflake |
| **Time Span** | January 22, 2020 → March 9, 2023 (~1,143 days) |
| **Granularity** | One row per country per day |
| **Coverage** | 180+ countries |

### Key Tables Used

| Table | Source | Primary Use |
|---|---|---|
| `JHU_COVID_19` | Johns Hopkins CSSE | **Primary time-series**: daily cases + deaths per country |
| `WHO_SITUATION_REPORTS` | WHO | Cortex LLM summarization input — narrative situation text |
| `GOOG_GLOBAL_MOBILITY_REPORT` | Google | Exogenous feature for ML model + Risk Tier Factor 7 (Mobility) |
| `DATABANK_DEMOGRAPHICS` | World Bank | Population normalization for per-capita case rates |
| `HDX_ACAPS` | ACAPS | Policy restriction events by date (lockdown lag features) |
| `METADATA` | Starschema | Column-level documentation for all tables |

> **Note on Vaccinations**: The Starschema dataset does not include a unified vaccination time-series table post-2022. Oxford/OurWorldInData vaccination data was discontinued from this listing. Where vaccination context is required, the system uses `DAYS_SINCE_FIRST_CASE` as an epidemic maturity proxy and references policy restriction events from `HDX_ACAPS` as a signal for intervention timing. This limitation is documented explicitly in the Fairness section.

### Primary Profiling Targets

The three core series that are profiled, engineered, and forecast:

| Series | Source Column | Transformation |
|---|---|---|
| **Cases** | `CONFIRMED` (cumulative) | Differenced → `DAILY_NEW_CASES`, then smoothed to `ROLLING_AVG_7D_CASES` |
| **Deaths** | `DEATHS` (cumulative) | Differenced → `DAILY_NEW_DEATHS`, then smoothed to `ROLLING_AVG_7D_DEATHS` |
| **Vaccination (proxy)** | `HDX_ACAPS` policy events | Vaccination start date extracted as a structural break feature — `DAYS_SINCE_VAX_POLICY` |

### 15 Target Countries (All 6 WHO Regions)

| WHO Region | Countries |
|---|---|
| Americas | US, Brazil, Mexico |
| Europe | United Kingdom, France, Germany, Italy |
| South-East Asia | India |
| Western Pacific | Japan, Korea (South), Australia |
| Africa | South Africa, Nigeria |
| Eastern Mediterranean | Turkey, Iran |

**Total rows after ingestion**: ~17,145 (15 countries × ~1,143 days)

---

## 3. Pipeline Architecture

```
┌──────────────────────────────────────────────┐
│  Snowflake Marketplace                        │
│  Starschema COVID-19 Epidemiological Data     │
│  Tables: JHU_COVID_19, WHO_REPORTS,           │
│          GOOG_MOBILITY, HDX_ACAPS             │
└──────────────────┬───────────────────────────┘
                   │  SQL query — no download
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 1: Ingestion — Snowflake SQL           │
│  → Filter 15 countries                        │
│  → Cumulative → daily deltas (LAG)            │
│  → Join cases + deaths + mobility             │
│  → TABLE: COVID_RAW (17,145 rows)             │
└──────────────────┬───────────────────────────┘
                   │  SQL Window Functions
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Feature Engineering — SQL Notebook  │
│  → 25+ epidemiological features               │
│  → EPIDEMIC_PHASE, Rt, Doubling Time          │
│  → Endemic flag for tail-period countries     │
│  → TABLE: COVID_FEATURES                      │
└──────────┬───────────────────┬───────────────┘
           │                   │
           ▼                   ▼
┌──────────────────┐  ┌────────────────────────┐
│  Phase 3A:        │  │  Phase 3B:              │
│  SNOWFLAKE.ML     │  │  SNOWFLAKE.ML           │
│  .FORECAST        │  │  .ANOMALY_DETECTION     │
│  Train: 2020–2022 │  │  Full series            │
│  Validate: 2022–  │  │  → IS_ANOMALY           │
│  2023 (MAPE)      │  │  → ANOMALY_DIRECTION    │
│  Forecast: 30 days│  │  → DEVIATION_PCT        │
│  TABLE: FORECASTS │  │  TABLE: ANOMALIES       │
└────────┬─────────┘  └──────────┬─────────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
┌──────────────────────────────────────────────┐
│  Phase 3C: Risk Tier — SQL CASE WHEN          │
│  → 8-factor composite score (0–100)           │
│  → HIGH / MODERATE / LOW classification       │
│  → TABLE: RISK_TIERS (15 rows)                │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 4: Cortex LLM                          │
│  → CORTEX.COMPLETE() — country briefs         │
│  → CORTEX.COMPLETE() — anomaly explanations   │
│  → CORTEX.SUMMARIZE() — WHO report summaries  │
│  → TABLE: CORTEX_NARRATIVES                   │
│  → TABLE: ANOMALY_EXPLANATIONS                │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│  Phase 5: Streamlit in Snowflake (SiS)        │
│  → 7 interactive tabs                         │
│  → Deployed inside Snowflake — no localhost   │
└──────────────────────────────────────────────┘
```

---

## 4. Phase 1: Data Ingestion

### What Happens

The Starschema `JHU_COVID_19` table is already in long format inside Snowflake — no CSV download, no pandas melt, no file caching required. The ingestion phase is a pure SQL operation.

**Step 1 — Filter to 15 target countries**
Query `JHU_COVID_19` filtering `COUNTRY_REGION IN (target list)` and `PROVINCE_STATE IS NULL` to keep country-level aggregates only.

**Step 2 — Convert cumulative → daily new values**
Use `LAG()` window functions to difference cumulative confirmed and deaths into daily new counts. Apply `GREATEST(..., 0)` to clip negative corrections (countries sometimes revise counts downward; negative daily cases are not epidemiologically meaningful).

**Step 3 — Join supplementary tables**
- LEFT JOIN `GOOG_GLOBAL_MOBILITY_REPORT` on `(COUNTRY_REGION, DATE)` to bring in retail + recreation mobility index as an exogenous signal
- LEFT JOIN `HDX_ACAPS` to flag policy intervention events (used for structural break detection and vaccination proxy)

**Step 4 — Profile null coverage**
Run `COUNT(*) / SUM(CASE WHEN CONFIRMED IS NULL THEN 1 ELSE 0 END)` grouped by country to quantify data gaps. This output directly feeds the Fairness documentation — African countries will show the highest null rates.

**Output**: `TABLE COVID_RAW` — ~17,145 rows × 8 columns

| Column | Type | Description |
|---|---|---|
| `COUNTRY_REGION` | VARCHAR | Country name |
| `DATE` | DATE | Report date |
| `CUMULATIVE_CONFIRMED` | INT | Running total cases |
| `CUMULATIVE_DEATHS` | INT | Running total deaths |
| `DAILY_NEW_CASES` | INT | Differenced daily cases (clipped ≥ 0) |
| `DAILY_NEW_DEATHS` | INT | Differenced daily deaths (clipped ≥ 0) |
| `MOBILITY_RETAIL` | FLOAT | Google retail + recreation mobility index |
| `POLICY_RESTRICTION_LEVEL` | INT | ACAPS restriction level (0–4) |

---

## 5. Phase 2: Feature Engineering

All features are computed using **SQL window functions** in a `CREATE TABLE COVID_FEATURES AS SELECT ...` statement. Every pandas operation from the original pipeline maps 1:1 to a SQL equivalent.

### A. Daily Deltas

| Feature | SQL Pattern | Purpose |
|---|---|---|
| `DAILY_NEW_CASES` | `GREATEST(CONFIRMED - LAG(CONFIRMED,1) OVER (...), 0)` | New cases today |
| `DAILY_NEW_DEATHS` | `GREATEST(DEATHS - LAG(DEATHS,1) OVER (...), 0)` | New deaths today |

### B. Rolling Averages (Smoothing)

| Feature | Window | Purpose |
|---|---|---|
| `ROLLING_AVG_7D_CASES` | Last 7 days | **Primary trend** — removes day-of-week noise |
| `ROLLING_AVG_14D_CASES` | Last 14 days | Two-week trend for comparison |
| `ROLLING_AVG_28D_CASES` | Last 28 days | Monthly baseline |
| `ROLLING_AVG_7D_DEATHS` | Last 7 days | Current death trend |
| `ROLLING_AVG_14D_DEATHS` | Last 14 days | Two-week death trend |

**Why rolling averages?** Raw daily counts are noisy: weekends see fewer tests (underreporting), Mondays see backlog dumps (artificial spikes), holidays cause major gaps. A 7-day window averages over exactly one week, eliminating day-of-week effects.

### C. Rolling Sums

| Feature | Window | Purpose |
|---|---|---|
| `SUM_7D_CASES` | Last 7 days | This week's total case burden |
| `PREV_SUM_7D_CASES` | `LAG(SUM_7D, 7)` | Last week's total for WoW comparison |
| `SUM_30D_CASES` | Last 30 days | Monthly case burden |

### D. Growth Metrics

| Feature | Formula | Interpretation |
|---|---|---|
| `WEEKLY_GROWTH_RATE` | `(today - 7d_ago) / 7d_ago × 100` | +50% = cases doubled vs same day last week |
| `DOUBLING_TIME_DAYS` | `7 × ln(2) / ln(cumul_today / cumul_7d_ago)` | Days for total cases to double. <14 days = very dangerous |
| `WOW_CHANGE_PCT` | `(this_week - last_week) / last_week × 100` | Week-over-week % change — most stable growth metric |

### E. Epidemiological Indicators

| Feature | Formula | Interpretation |
|---|---|---|
| `RT_EFFECTIVE` | `SUM(cases, days 0–4) / SUM(cases, days 5–9)` | Reproduction number: >1 = growing, <1 = shrinking |
| `CASE_FATALITY_RATE` | `total_deaths / total_cases × 100` | % of confirmed cases that died |
| `ACCELERATION` | `7D_avg_today − 7D_avg_7_days_ago` | 2nd derivative: positive = worsening, negative = improving |
| `VOLATILITY_14D` | `STDDEV(daily_cases) over 14 days` | How noisy/unpredictable the data is |
| `SIGNAL_TO_NOISE` | `14D_mean / 14D_stddev` | Data quality indicator — used to flag low-confidence forecasts |

**Rt Estimation Method**: Simplified Wallinga-Teunis approximation using a 5.2-day serial interval (Nishiura et al. 2020). A 5-day rolling sum window approximates the serial interval.

### F. Classifications

| Feature | Logic | Values |
|---|---|---|
| `TREND_DIRECTION` | Compare 7D avg vs 14D avg | `ACCELERATING` (7D > 14D × 1.1), `STABLE`, `DECELERATING` (7D < 14D × 0.9) |
| `EPIDEMIC_PHASE` | Multi-threshold 7D/14D/28D comparison | `EXPONENTIAL_GROWTH`, `LINEAR_GROWTH`, `PLATEAU`, `LINEAR_DECLINE`, `EXPONENTIAL_DECLINE`, `TRANSITION` |
| `ENDEMIC_FLAG` | Last 60 days all in `EXPONENTIAL_DECLINE` or `LINEAR_DECLINE` | `TRUE` / `FALSE` — triggers "Low Forecast Confidence" badge |

**Epidemic Phase Logic** (in priority order):
```
7D > 14D × 1.2 AND 7D > 28D × 1.3  → EXPONENTIAL_GROWTH
7D > 14D × 1.05                     → LINEAR_GROWTH
|7D - 14D| ≤ 14D × 0.05            → PLATEAU
7D < 14D × 0.95 AND 7D > 28D × 0.7 → LINEAR_DECLINE
7D < 28D × 0.7                      → EXPONENTIAL_DECLINE
Otherwise                            → TRANSITION
```

### G. Context

| Feature | Purpose |
|---|---|
| `DAYS_SINCE_FIRST_CASE` | Epidemic maturity — older epidemics behave differently than new ones |
| `DAYS_SINCE_VAX_POLICY` | Days since ACAPS recorded a national vaccination policy event — vaccination proxy |
| `MOBILITY_RETAIL_14D_AVG` | 14-day average of Google retail mobility — population movement context |

---

## 6. Phase 3: ML Models

### Data Split Strategy

Time series data cannot use random splits — using future data to predict the past is data leakage that produces artificially low MAPE scores. All splits are **strictly chronological**.

```
Jan 2020 ────────── Mar 2022 ────────── Mar 9, 2023 ───── Apr 8, 2023
│                         │                   │                  │
│     TRAINING SET        │   VALIDATION SET  │   FORECAST       │
│  (wave dynamics)        │  (known outcomes) │   (deliverable)  │
│  ~800 days              │    ~365 days      │     30 days      │
│  Jan 2020 – Mar 2022    │  Mar 2022–Mar2023 │  Mar 10–Apr 8    │
└─────────────────────────┘───────────────────┘──────────────────┘
```

**Why this split:**
- Training on Jan 2020 – Mar 2022 captures full wave dynamics: Wave 1, Wave 2, Delta, and Omicron peak — real epidemic behavior the model must learn
- Validation on Mar 2022 – Mar 2023 provides 365 days of known outcomes for a statistically robust MAPE calculation
- Forecasting on the last 30 days of JHU data is the hackathon deliverable — projections into a period with no ground truth

**Endemic tail problem**: If the model were trained on all 1,143 days through the declining endemic tail (late 2022 – Mar 2023), the dominant recent signal would be "cases are falling toward zero." The 30-day forecast would project continued decline for all countries, risk tiers would all show LOW, and Cortex would generate "everything is improving" narratives — making the system useless as a public health tool. The training cutoff at Mar 2022 deliberately avoids this recency bias.

**Endemic Phase Warning**: Countries where `ENDEMIC_FLAG = TRUE` receive a "⚠️ Low Forecast Confidence — Endemic Phase" badge in the dashboard instead of a raw forecast projection.

---

### Model 3A: Time-Series Forecasting

| Item | Detail |
|---|---|
| **Snowflake Object** | `SNOWFLAKE.ML.FORECAST` |
| **Target Variable** | `ROLLING_AVG_7D_CASES` (primary) and `ROLLING_AVG_7D_DEATHS` (secondary) |
| **Training Window** | Jan 22, 2020 → Mar 31, 2022 |
| **Validation Window** | Apr 1, 2022 → Mar 9, 2023 (MAPE computed here) |
| **Forecast Horizon** | 30 days (Mar 10 – Apr 8, 2023) |
| **Confidence Intervals** | `LOWER_BOUND` and `UPPER_BOUND` — native from Snowflake ML |
| **Models Trained** | 15 (one per country) for cases + 15 for deaths = 30 total |
| **Output** | 450 rows (15 countries × 30 days) with point forecast + bounds |
| **Evaluation** | `model!SHOW_EVALUATION_METRICS()` returns MAPE per country |
| **Walk-Forward CV** | `n_splits=3`, `test_size=30`, `gap=0` — 3-fold rolling window evaluation |

**Forecast output columns:**

| Column | Description |
|---|---|
| `TS` | Forecast date |
| `FORECAST` | Point forecast value |
| `LOWER_BOUND` | Lower confidence bound |
| `UPPER_BOUND` | Upper confidence bound |
| `COUNTRY_REGION` | Country identifier |

---

### Model 3B: Anomaly Detection

| Item | Detail |
|---|---|
| **Snowflake Object** | `SNOWFLAKE.ML.ANOMALY_DETECTION` |
| **Training Window** | Full series (Jan 2020 – Mar 2023) |
| **Features** | `DAILY_NEW_CASES`, `ROLLING_AVG_7D_CASES`, `ROLLING_AVG_14D_CASES` |
| **Models Trained** | 15 (one per country) |
| **Output** | `IS_ANOMALY` (boolean), `ANOMALY_SCORE`, `ANOMALY_DIRECTION` (spike/drop), `DEVIATION_PCT` |
| **Supervised Mode** | Known holiday reporting dips (Christmas, New Year) passed as labeled non-anomaly examples to reduce false positives |
| **Total Expected Flags** | ~765 across all 15 countries |

**Anomaly interpretation**:
- `ANOMALY_DIRECTION = SPIKE` + `DEVIATION_PCT > 300%` on a Monday after a public holiday = likely reporting backlog, not a real outbreak
- `ANOMALY_DIRECTION = SPIKE` + `DEVIATION_PCT > 200%` mid-week = genuine surge event requiring field investigation

---

### Model 3C: Risk Tier Classification

| Item | Detail |
|---|---|
| **Method** | Weighted composite scoring — deterministic SQL CASE WHEN rules |
| **Input** | Latest day's metrics per country + 30-day forecast endpoint |
| **Output** | Score (0–100) and tier (HIGH 🔴 / MODERATE 🟡 / LOW 🟢) |

**8-Factor Scoring Breakdown**:

| Factor | Max Points | Threshold Logic |
|---|---|---|
| Rt Value | 25 pts | Rt > 1.5 → 25; Rt > 1.0 → 15; Rt > 0.8 → 5; Rt ≤ 0.8 → 0 |
| Forecast Trend | 20 pts | Forecast > current by >50% → 20; >10% → 10; else → 0 |
| Acceleration | 15 pts | Acceleration > 0 AND direction = ACCELERATING → 15; else → 0 |
| Doubling Time | 15 pts | < 14 days → 15; < 30 days → 8; ≥ 30 days → 0 |
| Week-over-Week Change | 10 pts | > 50% → 10; > 20% → 5; ≤ 20% → 0 |
| Case Fatality Rate | 10 pts | > 3% → 10; > 1.5% → 5; ≤ 1.5% → 0 |
| Mobility | 5 pts | Retail mobility > 0% baseline (more movement = higher risk) → 5; else → 0 |
| Anomaly Recency | 5 pts | `IS_ANOMALY = TRUE` in last 7 days → 5; else → 0 |

**Tier thresholds**: Score ≥ 50 → HIGH 🔴 | Score ≥ 25 → MODERATE 🟡 | Score < 25 → LOW 🟢

---

## 7. Phase 4: Cortex LLM Narratives

This phase is entirely net-new versus the original local implementation and requires Snowflake Cortex, unavailable in any local Python environment.

### Three Narrative Types

**Type 1 — Country Health Brief** (per country, 15 total)
- **Function**: `SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', prompt)`
- **Prompt inputs**: Country name, current 7D avg cases, 7D avg deaths, Rt, doubling time, epidemic phase, risk tier, 30-day forecast direction, MAPE confidence flag
- **Output format**: 3 sentences — (1) current situation, (2) 30-day trajectory, (3) one recommended action for a non-technical government official
- **Stored in**: `TABLE CORTEX_NARRATIVES`

**Type 2 — Anomaly Explanation** (per flagged event)
- **Function**: `SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', prompt)`
- **Prompt inputs**: Country, date, actual cases, expected baseline (14D avg), deviation %, anomaly direction, day of week, proximity to known holiday
- **Output format**: 2 sentences — (1) what the anomaly looks like statistically, (2) most likely epidemiological or reporting explanation
- **Stored in**: `TABLE ANOMALY_EXPLANATIONS`

**Type 3 — WHO Report Summary** (from raw WHO situation reports)
- **Function**: `SNOWFLAKE.CORTEX.SUMMARIZE(report_text)`
- **Input**: Raw narrative text from `WHO_SITUATION_REPORTS` table
- **Output**: 2–3 sentence plain-English summary per WHO report entry
- **Stored in**: `TABLE WHO_SUMMARIES`

---

## 8. Phase 5: Streamlit in Snowflake Dashboard

The dashboard runs as a **Streamlit in Snowflake (SiS)** application — deployed entirely inside Snowflake with no external hosting, no localhost, and no credential management required.

**Connection pattern**: Uses native `get_active_session()` SiS object. All data queries run as `session.sql("SELECT ...").to_pandas()`.

### 7-Tab Structure

| Tab | Title | Primary Content | Key New Element vs. Original |
|---|---|---|---|
| 1 | 📊 Situational Awareness | Global KPIs, risk ranking bar chart, Rt vs Risk scatter (bubble size = case load), all-countries table | Mobility score now included in risk bar |
| 2 | 📈 Epi Deep-Dive | Per-country: Rt timeline, cases (7D/14D), deaths (7D), acceleration bar chart, epidemic phase timeline | `ENDEMIC_FLAG` badge on declining countries |
| 3 | 🔮 ML Observatory | 30-day forecast chart with native Snowflake ML confidence bands, forecast peak + date KPIs, MAPE by country | Native confidence bands (vs. ±30% heuristic in v1) |
| 4 | 🔬 Comparative Lab | Multi-country cases comparison, Rt comparison, log-scale view, CFR comparison | Deaths profiling chart added |
| 5 | 🚨 Alert Center | Anomaly timeline, top anomalies by deviation %, `ANOMALY_DIRECTION` (spike vs drop) column | Direction flag distinguishes real surges from reporting artifacts |
| 6 | 🤖 Cortex AI Briefs | Per-country LLM health narratives, anomaly explanations, WHO report summaries, regional triage ranking | Entire tab is net-new |
| 7 | 📋 Methodology & Fairness | Data gaps, geographic bias, MAPE by country, model confidence levels, vaccination data absence note | Expanded from original single tab |

---

## 9. Data Split Strategy

| Zone | Date Range | Size | Purpose |
|---|---|---|---|
| **Training** | Jan 22, 2020 → Mar 31, 2022 | ~800 days | Model learns wave dynamics: Wave 1, 2, Delta, Omicron |
| **Validation** | Apr 1, 2022 → Mar 9, 2023 | ~365 days | Compute MAPE against known outcomes |
| **Forecast** | Mar 10, 2023 → Apr 8, 2023 | 30 days | True forward prediction — hackathon deliverable |

**Walk-forward cross-validation inside Snowflake ML**: `n_splits=3`, `test_size=30 days`, `gap=0` — MAPE reported as average across 3 folds for statistical robustness.

**Step 1 (validation demo)**: Train on Jan 2020 – Feb 6, 2023 → forecast 30 days → compare against Feb 7 – Mar 9 actuals → compute and display MAPE on dashboard.

**Step 2 (true deliverable)**: Retrain on all 1,143 days → forecast Mar 10 – Apr 8 → this is the 30-day projection shown in the ML Observatory tab.

---

## 10. Snowflake Object Inventory

All objects live inside a single database `PUBLIC_HEALTH_DB`:

```
PUBLIC_HEALTH_DB
├── SCHEMA: RAW
│   └── TABLE: COVID_RAW              (~17,145 rows, 8 cols)
│
├── SCHEMA: FEATURES
│   └── TABLE: COVID_FEATURES         (~17,145 rows, 27+ cols)
│
├── SCHEMA: ML
│   ├── ML.FORECAST: cases_forecast_{country}    (15 objects)
│   ├── ML.FORECAST: deaths_forecast_{country}   (15 objects)
│   ├── ML.ANOMALY_DETECTION: anomaly_{country}  (15 objects)
│   ├── TABLE: FORECASTS              (450 rows + confidence bounds)
│   └── TABLE: ANOMALIES              (~765 flagged rows + direction)
│
├── SCHEMA: ANALYTICS
│   ├── TABLE: RISK_TIERS             (15 rows, 8-factor score)
│   ├── TABLE: CORTEX_NARRATIVES      (15 country briefs)
│   ├── TABLE: ANOMALY_EXPLANATIONS   (LLM text per flagged event)
│   └── TABLE: WHO_SUMMARIES          (Cortex-summarized WHO reports)
│
└── STREAMLIT APP: public_health_dashboard   (7 tabs, SiS)
```

### Ordered Build Sequence

| Step | Task | Snowflake Object |
|---|---|---|
| 1 | Mount Starschema from Marketplace | External table reference |
| 2 | Create `COVID_RAW` — filter 15 countries + daily deltas | SQL Worksheet |
| 3 | Create `COVID_FEATURES` — 25+ window function features | SQL Worksheet |
| 4 | Add `EPIDEMIC_PHASE`, `ENDEMIC_FLAG`, `SIGNAL_TO_NOISE` | SQL CASE WHEN |
| 5 | Train `SNOWFLAKE.ML.FORECAST` on Jan 2020 – Mar 2022 | Snowflake ML |
| 6 | Validate MAPE against Mar 2022 – Mar 2023 holdout | `model!SHOW_EVALUATION_METRICS()` |
| 7 | Generate true 30-day forward forecast | `model!FORECAST(30)` |
| 8 | Train `SNOWFLAKE.ML.ANOMALY_DETECTION` per country | Snowflake ML |
| 9 | Compute 8-factor Risk Tiers | SQL CASE WHEN on FORECASTS + FEATURES |
| 10 | Generate Cortex narratives (briefs + anomaly explanations) | `CORTEX.COMPLETE()` |
| 11 | Summarize WHO situation reports | `CORTEX.SUMMARIZE()` |
| 12 | Build Streamlit in Snowflake — 7 tabs | SiS app |

---

## 11. Hackathon Requirements vs Implementation

| Hackathon Requirement | Points | Status | Implementation |
|---|---|---|---|
| Snowflake Marketplace data | — | ✅ | Direct Starschema `JHU_COVID_19` table query — no external download |
| Feature engineering in Snowpark | 30 pts (Technical) | ✅ Exceeded | 27+ features including Rt, doubling time, epidemic phase, acceleration, vaccination proxy, endemic flag |
| `SNOWFLAKE.ML.FORECAST` | 25 pts (Model Quality) | ✅ | 30 models (cases + deaths per country), walk-forward CV, MAPE per country |
| `SNOWFLAKE.ML.ANOMALY_DETECTION` | 25 pts (Model Quality) | ✅ | 15 models, supervised mode with holiday labels, direction flag |
| `CORTEX.COMPLETE()` narratives | 20 pts (Social Impact) | ✅ | Country briefs + anomaly explanations in plain English |
| Risk tier classification | — | ✅ Upgraded | 8-factor composite (added mobility + anomaly recency vs. original 7) |
| Streamlit dashboard | 15 pts (Presentation) | ✅ | 7-tab SiS app with all ML outputs |
| 10+ countries | — | ✅ | 15 countries across all 6 WHO regions |
| Fairness documentation | 20 pts (Social Impact) | ✅ | MAPE-by-country, Africa null analysis, vaccination gap, endemic tail bias |

---

## 12. Fairness & Bias Documentation

| Bias Type | Description | Mitigation |
|---|---|---|
| **Africa underrepresentation** | Nigeria and South Africa have the highest null rates in `JHU_COVID_19` — testing infrastructure limits case detection | Flagged in dashboard with null-rate badge; MAPE shown explicitly so users know forecast is less reliable |
| **Reporting lag bias** | Weekend/holiday reporting dips create artificial "recovery" signals and Monday backlog dumps create artificial spikes | 7-day rolling average corrects for day-of-week effects; holiday labels in anomaly detection reduce false positives |
| **Vaccination data absence** | No unified vaccination time-series in Starschema dataset post-2022 | `DAYS_SINCE_VAX_POLICY` from ACAPS used as structural break proxy; absence noted explicitly in Methodology tab |
| **Endemic tail recency bias** | Training on the full dataset through Mar 2023 causes the model to over-weight the declining endemic tail, producing trivially declining forecasts | Training cutoff at Mar 2022; `ENDEMIC_FLAG` badge warns users when forecast confidence is low |
| **Data freeze** | JHU CSSE stopped updating March 2023 — all forecasts are historical simulations, not live predictions | Clearly stated in dashboard header and Methodology tab |
| **High-income country bias** | Wealthier countries (US, Germany, Japan) have more consistent, complete testing data → lower MAPE → model appears more reliable for them | MAPE displayed per country; low-SIGNAL_TO_NOISE countries flagged separately |

---

## 13. Key Epidemiological Concepts

### What is Rt?
The **effective reproduction number** — the average number of new infections caused by each infected person at a given point in time. If Rt > 1, each person infects more than one other person and the epidemic grows. If Rt < 1, the epidemic shrinks. It is the single most important metric for pandemic decision-making and the primary input for lockdown and intervention decisions. Computed here using a simplified Wallinga-Teunis method with a 5.2-day serial interval.

### What is Doubling Time?
How many days it would take for total cases to double at the current growth rate. A doubling time of 7 days means cases are growing very fast. A doubling time of 100+ days means growth is almost flat. Most governments use a 14-day threshold as a trigger for emergency declarations.

### What is Case Fatality Rate (CFR)?
Deaths ÷ confirmed cases × 100. Measures how deadly the disease is within confirmed cases, but is biased by testing rates — more testing finds more mild cases, lowering CFR without the disease actually becoming less deadly. Useful for comparing healthcare system effectiveness across countries with similar testing rates.

### What is Acceleration?
The **second derivative** of the epidemic curve — it measures whether the growth rate itself is increasing or decreasing. Positive acceleration means the situation is worsening faster. Negative acceleration means improvement is speeding up. A country can have both declining case counts (good) and positive acceleration (concerning) if the rate of decline is slowing.

### Why Rolling Averages?
Raw daily case counts are noisy due to reporting delays, weekend effects, and batch uploads. A 7-day rolling average smooths these artifacts by averaging over exactly one complete week, eliminating day-of-week effects while still capturing genuine epidemic trends. The comparison of 7-day vs. 14-day vs. 28-day averages reveals whether the epidemic is accelerating, stable, or in sustained decline.

### What is the Endemic Phase Warning?
When a country has been in continuous `EXPONENTIAL_DECLINE` or `LINEAR_DECLINE` for 60+ consecutive days, its epidemic has transitioned from active outbreak to endemic circulation. Forecasting an endemic trend produces trivially declining projections with no public health utility. The system flags these countries with a "⚠️ Low Forecast Confidence — Endemic Phase" badge rather than showing a misleading near-zero forecast.


---

## 14. Cortex AI Briefs — Design & Implementation

### Two Patterns in One Tab

The Cortex AI Briefs tab uses **both** a pre-generation pattern and a live Q&A pattern for different content types. They serve different purposes and together form the strongest demonstration of Cortex capability in the dashboard.

| Content Type | Pattern | Reason |
|---|---|---|
| Country health briefs (3-sentence summary) | Pre-generated | Same for every user — generated once during pipeline, stored in `CORTEX_NARRATIVES`, fast to load |
| Anomaly explanations | Pre-generated | Generated once per detected anomaly event, stored in `ANOMALY_EXPLANATIONS` permanently |
| WHO report summaries | Pre-generated | Source text does not change — summarize once via `CORTEX.SUMMARIZE()` |
| "Ask anything about this country" Q&A | Live call | Ad-hoc natural language interface — the key innovation differentiator |

---

### Section 1 — Country Health Brief (Pre-Generated)

Displayed immediately when the user selects a country. Reads directly from the `CORTEX_NARRATIVES` table — no live Cortex call on page load.

**Visual layout**:
```
┌─────────────────────────────────────────────────────────┐
│  🌍 Select Country: [ United States ▼ ]                 │
├─────────────────────────────────────────────────────────┤
│  📋 AI Health Brief                    🟡 MODERATE RISK │
│  ─────────────────────────────────────────────────────  │
│  The United States is currently in a LINEAR_DECLINE      │
│  phase with a 7-day average of 12,400 new cases and      │
│  an Rt of 0.87, indicating the outbreak is shrinking     │
│  but not yet resolved.                                   │
│                                                          │
│  Over the next 30 days, cases are projected to fall      │
│  to approximately 8,200/day (±1,800), with no major      │
│  acceleration signals detected in the last 14 days.      │
│                                                          │
│  Recommended action: Maintain current surveillance       │
│  intensity and prepare for potential seasonal uptick     │
│  in 6–8 weeks based on historical wave patterns.         │
│                                                          │
│  ⚠️ Model MAPE: 8.3% — High Confidence                  │
│  Generated: March 9, 2023                               │
└─────────────────────────────────────────────────────────┘
```

**Prompt structure** used during pipeline execution:

```
You are a public health analyst. Based on this COVID-19 data for {COUNTRY}:
- Current 7-day avg cases: {CASES_7D}
- Current 7-day avg deaths: {DEATHS_7D}
- Case Fatality Rate: {CFR}%
- Rt (effective reproduction number): {RT}
- Doubling time: {DOUBLING_TIME} days
- Epidemic phase: {EPIDEMIC_PHASE}
- Risk tier: {RISK_TIER}
- 30-day forecast direction: {FORECAST_DIRECTION}
- Forecast model MAPE: {MAPE}%

Write a 3-sentence plain-English health summary for a non-technical
government official. Sentence 1: current situation. Sentence 2: 30-day
trajectory. Sentence 3: one specific recommended action.
```

---

### Section 2 — Anomaly Explanation (Pre-Generated)

Displayed below the health brief, showing the most recent flagged anomaly for the selected country. Reads from `ANOMALY_EXPLANATIONS` table.

**Visual layout**:
```
┌─────────────────────────────────────────────────────────┐
│  🚨 Recent Anomaly — United States                      │
│  ─────────────────────────────────────────────────────  │
│  📅 Jan 3, 2022  |  +847% deviation  |  📈 SPIKE        │
│  ─────────────────────────────────────────────────────  │
│  Cases jumped to 1.35M vs. a 14-day baseline of         │
│  142K — an 847% deviation occurring on a Tuesday        │
│  following the New Year holiday weekend.                 │
│                                                          │
│  This pattern is consistent with a reporting backlog     │
│  rather than a genuine single-day surge; the 3-day       │
│  post-holiday accumulation was released simultaneously.  │
└─────────────────────────────────────────────────────────┘
```

**Prompt structure** used during pipeline execution:

```
You are an epidemiological data analyst. A COVID-19 anomaly was detected
for {COUNTRY} on {DATE}:
- Actual cases: {ACTUAL}
- 14-day baseline expected: {EXPECTED}
- Deviation: {DEVIATION_PCT}%
- Anomaly direction: {ANOMALY_DIRECTION} (spike or drop)
- Day of week: {DAY_OF_WEEK}
- Days since nearest public holiday: {HOLIDAY_PROXIMITY}

Write 2 sentences. Sentence 1: describe what the anomaly looks like
statistically. Sentence 2: the most likely epidemiological or reporting
explanation for a public health official.
```

---

### Section 3 — Live Q&A Box (Pattern 2 — Innovation Differentiator)

The ad-hoc question interface allows any user — regardless of technical skill — to ask natural language questions about any country and receive a Cortex-generated answer grounded in that country's live dashboard data.

**Visual layout**:
```
┌─────────────────────────────────────────────────────────┐
│  💬 Ask About United States                             │
│  ─────────────────────────────────────────────────────  │
│  Suggested questions (click to auto-fill):              │
│  [How does the US compare to Europe?]                   │
│  [Should we extend travel restrictions?]                │
│  [What drove the January 2022 spike?]                   │
│                                                          │
│  ┌─────────────────────────────────────────────┐  [Ask] │
│  │ Type your question here...                  │        │
│  └─────────────────────────────────────────────┘        │
│                                                          │
│  📝 Answer:                                             │
│  ─────────────────────────────────────────────────────  │
│  [ Cortex response appears here after clicking Ask ]    │
└─────────────────────────────────────────────────────────┘
```

**Suggested questions** (pre-loaded as clickable buttons that auto-fill the text box):
- "How does this country compare to Europe right now?"
- "Should we extend travel restrictions to this country?"
- "What drove the most recent anomaly spike?"
- "Is this country likely to need international aid in the next 30 days?"
- "What is the current situation summary for a press conference?"

**Why suggested questions matter for the demo**: During a live hackathon presentation, pre-loaded suggested questions let the presenter click one button and immediately show a polished Cortex response without fumbling to type a question under pressure.

**SiS implementation pattern**:

```python
# Pre-load suggested questions as clickable buttons
suggestions = [
    "How does this country compare to Europe right now?",
    "Should we extend travel restrictions to this country?",
    "What drove the most recent anomaly spike?",
    "Is this country likely to need international aid in 30 days?",
    "Summarize the current situation for a press conference."
]

cols = st.columns(len(suggestions))
user_question = st.session_state.get("prefill_q", "")

for i, q in enumerate(suggestions):
    if cols[i].button(q[:35] + "...", key=f"sugg_{i}"):
        st.session_state["prefill_q"] = q

user_question = st.text_input("Or type your own question:", value=user_question)

if st.button("Ask Cortex") and user_question:
    prompt = f"""
    You are a public health analyst. Here is the current COVID-19 data for {selected_country}:
    - 7-day avg cases: {cases_7d}
    - 7-day avg deaths: {deaths_7d}
    - Rt: {rt_value}
    - Risk Tier: {risk_tier}
    - 30-day forecast direction: {forecast_direction}
    - Epidemic Phase: {epidemic_phase}
    - Doubling Time: {doubling_time} days
    - Model MAPE: {mape}%

    Answer this question for a non-technical government official in under 4 sentences.
    Be specific and actionable. Question: {user_question}
    """

    result = session.sql(
        f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', $${prompt}$$)"
    ).collect()[0][0]

    st.info(result)
```

---

### Section 4 — Regional Triage Summary (Pre-Generated)

A global-level Cortex output shown at the top of the tab before the country selector. Generated once using all 15 countries' risk scores as input.

**Visual layout**:
```
┌─────────────────────────────────────────────────────────┐
│  🌐 Global Triage Summary — All 15 Countries            │
│  ─────────────────────────────────────────────────────  │
│  🔴 Urgent Aid Needed:   India, Nigeria, Brazil         │
│  🟡 Monitor Closely:     US, Mexico, Turkey, Iran       │
│  🟢 Stable:              Germany, Japan, Australia...   │
│                                                          │
│  [AI-generated 2-sentence regional narrative]           │
└─────────────────────────────────────────────────────────┘
```

---

### Why This Design Wins on Hackathon Points

| Feature | Scoring Category | Why It Scores |
|---|---|---|
| Pre-generated country briefs | Social Impact (20 pts) | Non-technical minister reads a 3-sentence answer with zero data literacy required |
| Anomaly plain-English explanations | Technical Depth (30 pts) | Connects ML anomaly detection output → human-readable root cause analysis |
| Live Q&A box | Innovation (10 pts) | Ad-hoc natural language interface on top of an epidemiological ML pipeline — no competing team will have this |
| Suggested question buttons | Presentation (15 pts) | Enables smooth, polished live demo without typing under pressure |
| Regional triage summary | Social Impact (20 pts) | Answers the UN aid allocation question (Q15) directly in plain English |

